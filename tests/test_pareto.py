"""Tests for shared.pareto — dominance, would_add, feasible."""

from shared.database import DataElement
from shared.pareto import ParetoFront


def _elem(metric, exec_time=100, memory=1000, name="test", code="x"):
    return DataElement(
        name=name, code=code, success=True, metric=metric,
        objectives={"exec_time": exec_time, "memory_mb": memory},
    )


def test_add_non_dominated():
    pf = ParetoFront(max_size=10)
    assert pf.update(_elem(0.9, 100, 1000))
    assert pf.size == 1


def test_dominated_not_added():
    pf = ParetoFront(max_size=10)
    pf.update(_elem(0.8, 100, 1000))
    # Worse on all objectives
    assert not pf.update(_elem(0.9, 200, 2000))
    assert pf.size == 1


def test_dominates_removes_existing():
    pf = ParetoFront(max_size=10)
    pf.update(_elem(0.9, 200, 2000))
    # Better on all objectives — should remove the first
    assert pf.update(_elem(0.8, 100, 1000))
    assert pf.size == 1
    assert pf.best.metric == 0.8


def test_non_dominated_tradeoff():
    pf = ParetoFront(max_size=10)
    pf.update(_elem(0.8, 200, 1000))  # better metric, worse time
    pf.update(_elem(0.9, 100, 1000))  # worse metric, better time
    assert pf.size == 2


def test_would_add():
    pf = ParetoFront(max_size=10)
    pf.update(_elem(0.8, 100, 1000))
    # Dominated — should not add
    assert not pf.would_add(_elem(0.9, 200, 2000))
    # Non-dominated tradeoff — would add
    assert pf.would_add(_elem(0.9, 50, 500))
    # Better on all — would add
    assert pf.would_add(_elem(0.7, 50, 500))


def test_would_add_failed():
    pf = ParetoFront(max_size=10)
    pf.update(_elem(0.8, 100, 1000))
    failed = DataElement(success=False, metric=None)
    assert not pf.would_add(failed)


def test_max_size_evicts_worst_primary():
    """When front exceeds max_size, evict worst primary objective."""
    pf = ParetoFront(max_size=2)
    pf.update(_elem(0.8, 200, 2000, name="a"))
    pf.update(_elem(0.9, 50, 500, name="b"))
    assert pf.size == 2
    # Adding a third — should evict worst primary (0.9)
    pf.update(_elem(0.85, 100, 1000, name="c"))
    assert pf.size == 2
    names = {c.element.name for c in pf.candidates}
    assert "b" not in names, "Worst primary objective should be evicted"


def test_count_dominated_by():
    pf = ParetoFront(max_size=10)
    pf.update(_elem(0.9, 200, 2000, name="a"))
    pf.update(_elem(0.85, 150, 1500, name="b"))
    pf.update(_elem(0.95, 50, 500, name="c"))

    # Something that dominates a and b but not c
    dominator = _elem(0.8, 100, 1000)
    count = pf.count_dominated_by(dominator)
    assert count >= 1


def test_get_feasible():
    pf = ParetoFront(max_size=10)
    # Use non-dominated tradeoff so both stay on front
    e1 = DataElement(
        name="a", code="a", success=True, metric=0.85,
        objectives={"exec_time": 200, "memory_mb": 1000, "flops_equivalent_size": 200_000},
    )
    e2 = DataElement(
        name="b", code="b", success=True, metric=0.9,
        objectives={"exec_time": 50, "memory_mb": 500, "flops_equivalent_size": 2_000_000},
    )
    pf.update(e1)
    pf.update(e2)
    assert pf.size == 2  # Both on front (tradeoff: metric vs time/memory)

    # Only e1 is in the tiny bucket
    feasible = pf.get_feasible(100_000, 500_000)
    assert len(feasible) == 1
    assert feasible[0].element.name == "a"

    # Both in a wide range
    feasible = pf.get_feasible(100_000, 10_000_000)
    assert len(feasible) == 2

    # None in a narrow range
    feasible = pf.get_feasible(50_000_000, 100_000_000)
    assert len(feasible) == 0
