"""Tests for validator work-splitting (get_my_assignments)."""

from types import SimpleNamespace

from validator.neuron import compute_live_validator_uids, get_my_assignments


def _mg(permits, last_update=None, n=None):
    """Build a stub metagraph with permits and optional last_update."""
    n = n if n is not None else len(permits)
    return SimpleNamespace(
        n=n,
        validator_permit=list(permits),
        last_update=list(last_update) if last_update is not None else None,
    )


class TestComputeLiveValidatorUids:
    def test_permit_only(self):
        """Without miner/block info, returns all permit holders."""
        mg = _mg([True, False, True, True])
        assert compute_live_validator_uids(mg) == [0, 2, 3]

    def test_excludes_miners(self):
        """UIDs with commitments are miners, not validators, even with permit."""
        mg = _mg([True, True, True, True])
        result = compute_live_validator_uids(mg, miner_uids={2, 3})
        assert result == [0, 1]

    def test_excludes_stale_last_update(self):
        """Validators whose last_update is older than stale_blocks are skipped."""
        # current_block=1000, stale_blocks=100. UID 2's last_update=500 -> stale.
        mg = _mg([True, True, True], last_update=[950, 999, 500])
        result = compute_live_validator_uids(
            mg, current_block=1000, stale_blocks=100,
        )
        assert result == [0, 1]

    def test_last_update_zero_is_bootstrap(self):
        """last_update=0 (never updated) is not treated as stale."""
        mg = _mg([True, True], last_update=[0, 0])
        result = compute_live_validator_uids(
            mg, current_block=1000, stale_blocks=100,
        )
        assert result == [0, 1]

    def test_combined_filters(self):
        """All three filters compose: permit + not-miner + fresh."""
        mg = _mg(
            [True, True, True, False, True],
            last_update=[990, 990, 100, 990, 990],
        )
        result = compute_live_validator_uids(
            mg,
            miner_uids={1},          # UID 1 is a miner
            current_block=1000,
            stale_blocks=100,        # UID 2 stale
        )
        # UID 0 kept, 1 excluded (miner), 2 excluded (stale),
        # 3 excluded (no permit), 4 kept
        assert result == [0, 4]

    def test_no_permits_returns_empty(self):
        mg = SimpleNamespace(n=3, validator_permit=None, last_update=None)
        assert compute_live_validator_uids(mg) == []


class TestGetMyAssignments:
    def test_deterministic(self):
        """Same inputs -> same output."""
        uids = list(range(10))
        validators = [0, 1, 2]
        a1 = get_my_assignments(uids, validators, 0, seed=42)
        a2 = get_my_assignments(uids, validators, 0, seed=42)
        assert a1 == a2

    def test_different_seeds_differ(self):
        """Different seeds -> different assignments."""
        uids = list(range(20))
        validators = [0, 1]
        a1 = get_my_assignments(uids, validators, 0, seed=42)
        a2 = get_my_assignments(uids, validators, 0, seed=99)
        assert a1 != a2

    def test_full_coverage(self):
        """All UIDs are assigned to exactly one validator."""
        uids = list(range(10))
        validators = [0, 1, 2]
        all_assigned = []
        for v in validators:
            all_assigned.extend(get_my_assignments(uids, validators, v, seed=42))
        assert sorted(all_assigned) == sorted(uids)

    def test_no_overlaps(self):
        """No UID assigned to multiple validators."""
        uids = list(range(15))
        validators = [0, 1, 2]
        sets = []
        for v in validators:
            assigned = get_my_assignments(uids, validators, v, seed=42)
            sets.append(set(assigned))

        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert sets[i].isdisjoint(sets[j]), f"Overlap between validator {i} and {j}"

    def test_single_validator_gets_all(self):
        """With one validator, it gets all UIDs."""
        uids = list(range(5))
        result = get_my_assignments(uids, [0], 0, seed=42)
        assert sorted(result) == sorted(uids)

    def test_empty_uids(self):
        """Empty UID list returns empty."""
        result = get_my_assignments([], [0, 1], 0, seed=42)
        assert result == []

    def test_validator_not_in_list_gets_all(self):
        """If my_uid not in validator_uids, return all."""
        uids = list(range(5))
        result = get_my_assignments(uids, [0, 1], 99, seed=42)
        assert sorted(result) == sorted(uids)

    def test_balanced_distribution(self):
        """UIDs are roughly evenly distributed."""
        uids = list(range(30))
        validators = [10, 20, 30]
        counts = {}
        for v in validators:
            counts[v] = len(get_my_assignments(uids, validators, v, seed=42))
        assert all(c == 10 for c in counts.values())
