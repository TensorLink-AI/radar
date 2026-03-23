"""Tests for validator work-splitting (get_my_assignments)."""

from validator.neuron import get_my_assignments


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
