"""The held-out calls, fixed once, that no vintage may train on.

The retrieval benchmark in `diagnostics/bow_vs_dense.py` builds its queries from the same structure the
contrastive stage trains on, an analyst question and the executive answer that follows it. Training on a
call and then evaluating on that same call measures memorization, and it would look like success: the
model would score well on the one number Phase 4 is steering by.

So the split lives in its own module, because the pair builder and the evaluator both depend on it and
neither may define it. Three properties make it safe to rely on.

**It is a function of the call id, not of a shuffle.** `is_holdout(keydevid)` needs no stored list, no
ordering and no random state, so the same call is held out in every process, on every machine, at every
vintage, forever. A stored list would drift the moment the corpus was refreshed with new calls.

**It is by call, not by chunk.** Two chunks of one call share speakers, topic and boilerplate, so
splitting at chunk level would put near-duplicates on both sides and the evaluation would be
contaminated while looking disjoint.

**The salt is versioned and recorded.** Changing `SALT` reassigns every call, which invalidates every
comparison made before the change, so the value is pinned here and written into the artifacts that
depend on it.
"""
import hashlib

SALT = "ecalls-holdout-v1"   # changing this reassigns every call and invalidates prior comparisons
# 10% rather than 5%, and the choice is one-way. Raising the fraction later moves calls from trainable
# to held out, so any model already trained on them is contaminated, which means the number has to be
# generous from the start. **Measured** on the real corpus: 5% gives 234 held-out calls per quarterly
# shard against the 200 the retrieval benchmark samples, which leaves no room for a wider evaluation or
# for several seeds. 10% gives roughly 468 per shard, and the cost is 10% of the training text.
DEFAULT_FRACTION = 0.10

_SCALE = 1 << 32


def call_bucket(keydevid: int, salt: str = SALT) -> float:
    """A stable number in [0, 1) for one call.

    sha1 of the salted id rather than Python's `hash`, because `hash` is randomized per process for
    strings and would put a call in the training set in one run and the evaluation set in the next.
    """
    h = hashlib.sha1(f"{salt}:{int(keydevid)}".encode()).digest()
    return int.from_bytes(h[:4], "big") / _SCALE


def is_holdout(keydevid: int, fraction: float = DEFAULT_FRACTION, salt: str = SALT) -> bool:
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    return call_bucket(keydevid, salt) < fraction


def split_ids(keydevids, fraction: float = DEFAULT_FRACTION, salt: str = SALT):
    """(train_ids, holdout_ids) for an iterable of call ids, preserving the input order."""
    train, held = [], []
    for k in keydevids:
        (held if is_holdout(k, fraction, salt) else train).append(k)
    return train, held


def describe(fraction: float = DEFAULT_FRACTION, salt: str = SALT) -> dict:
    """What to record in an artifact so a later reader can reproduce the split."""
    return {"holdout_salt": salt, "holdout_fraction": fraction,
            "holdout_rule": "sha1(salt:keydevid)[:4] / 2**32 < fraction"}
