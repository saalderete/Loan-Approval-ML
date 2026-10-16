import numpy as np

from loan_approval.data.split import SPLIT_NAMES, SplitSpec, assign_splits, split_hash


def test_split_is_deterministic_disjoint_and_stratified(synthetic_frame):
    frame, _ = synthetic_frame
    a = assign_splits(frame, SplitSpec(seed=11))
    b = assign_splits(frame, SplitSpec(seed=11))
    assert (a == b).all()
    assert set(a.unique()) == set(SPLIT_NAMES)
    assert split_hash(frame["_row_id"], a) == split_hash(frame["_row_id"], b)
    overall = frame["denied"].mean()
    for name in SPLIT_NAMES:
        assert abs(frame.loc[a == name, "denied"].mean() - overall) < 0.03
    shares = a.value_counts(normalize=True)
    assert np.isclose(shares["test"], 0.2, atol=0.01)
    assert np.isclose(shares["train"], 0.6, atol=0.01)


def test_different_seed_changes_hash(synthetic_frame):
    frame, _ = synthetic_frame
    h1 = split_hash(frame["_row_id"], assign_splits(frame, SplitSpec(seed=1)))
    h2 = split_hash(frame["_row_id"], assign_splits(frame, SplitSpec(seed=2)))
    assert h1 != h2
