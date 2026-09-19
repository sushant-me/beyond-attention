"""A keyed, in-process store: memory that is *addressed*, not only decayed.

The rest of this repository carries a fixed-size state forward: the S6 recurrence
keeps the last thing it was told and forgets the rest, and the agent loop's eight
registers are that state decoded. That is working memory, and it is bounded by
construction — ``streaming.py`` measures the model's state at **19,456 bytes at
every length from 16,384 to 1,048,576 tokens** and the same flatness holds for the
agent's 64-byte register file. What it cannot do is hold a fact from a thousand
steps ago, because there is nowhere to put it.

This module is the other half of that trade, and it is deliberately the least
interesting implementation that works: an explicit table the loop writes facts
into and reads facts out of **by key**.

* **Direct-mapped.** ``slot = key % capacity``. There is no search, no eviction
  policy to tune, and no similarity — a key either is in its slot or it is not.
* **Tagged by default.** The slot holds the key *and* the value, so a retrieval
  either returns the fact that was written under that key or reports a miss. The
  tag costs 8 of the 17 bytes per slot and is what makes "it remembered" mean
  "it remembered the right thing" rather than "it returned whatever was in that
  slot"; ``verify_key=False`` is the control that measures the difference, and it
  is cheaper and wrong.
* **Replace on overwrite.** Writing a key twice leaves the second value, which is
  the property the stale-fact control checks: a store that returns the first
  value is worse than one that returns nothing, because a caller cannot tell.
* **Bounded or not.** A bounded store costs a constant number of bytes no matter
  how long the episode runs; ``capacity=None`` keeps everything and its bytes grow
  with the number of writes, which is the unbounded baseline this module is
  measured against rather than a recommended configuration.

What it is **not**, stated here rather than left to be discovered:

* **It is not associative memory and it is not learned.** Retrieval is integer
  equality on a key that the caller supplies. Nothing decides *what* is worth
  remembering or *which* key it belongs under; the agent's plan says both.
* **It is not the SSM's state and has nothing to do with it.** The recurrence in
  ``agent.py`` carries one value between adjacent steps; this array is what makes
  the distance between the fact and the query stop mattering. The measurements
  separate the two, and the state-space model does not contribute to the
  long-range result.
* **It is not a vector store, a cache, or a database.** No disk, no network, no
  clock, no serialisation — only ``numpy`` arrays in this process, and a test
  reads this module's imports to keep it that way.

The one place a random number enters is the ``random_retrieval`` control, which
is how the random-retrieval floor is measured. It is seeded, and its seed is part
of the store's ``spec()`` so a run can be replayed exactly.
"""

from __future__ import annotations

import numpy as np

# Bytes per slot, element by element. These are numpy's own element sizes rather
# than a convention: the measured `bytes()` below is `nbytes` on the allocated
# arrays, and the arithmetic in the results file has to agree with it.
KEY_BYTES = 8       # int64
VALUE_BYTES = 8     # int64
PRESENT_BYTES = 1   # bool
TAGGED_SLOT_BYTES = KEY_BYTES + VALUE_BYTES + PRESENT_BYTES   # 17
UNTAGGED_SLOT_BYTES = VALUE_BYTES + PRESENT_BYTES             # 9

# The capacity the agent loop uses by default: eight slots, matching the eight
# named registers of its working state, because the comparison between the two is
# the point and equal-width is the fair version of it.
DEFAULT_CAPACITY = 8
MIN_CAPACITY = 1


def store_bytes(capacity: int, verify_key: bool = True) -> int:
    """Bytes an ``capacity``-slot store occupies, analytically.

    The same number ``MemoryStore.bytes()`` measures from the live arrays, so the
    published figure can be checked against the object rather than trusted.
    """
    return capacity * (TAGGED_SLOT_BYTES if verify_key else UNTAGGED_SLOT_BYTES)


class MemoryStore:
    """A bounded, keyed, in-process store. Deterministic except for one control.

    Parameters, each of which exists to make one claim measurable:

    ``capacity``
        Number of slots. ``None`` means unbounded: the store grows by doubling and
        its byte cost grows with the number of writes. That is the baseline, not
        the recommendation.
    ``overwrite``
        ``True`` (default): a later write under the same key replaces the earlier
        value. ``False`` is the stale-fact control — a first-write-wins store,
        which is precisely the failure mode "recall that returns stale data is
        worse than no recall" names.
    ``verify_key``
        ``True`` (default): the slot is tagged, so retrieval confirms the key it
        finds. ``False`` drops the tag (9 bytes a slot instead of 17) and returns
        whatever occupies the slot, which is how a retrieval returns a *different*
        fact's value. Untagged retrieval is only defined for a bounded store,
        because without the tag there is nothing to check a growing log against.
    ``retrieval``
        ``False`` disables reading entirely: writes still happen and still cost
        their bytes, and every read is a miss. That separates "the capability is
        the retrieval" from "the capability is the storage".
    ``random_retrieval``
        ``True`` returns a uniformly random *live* value instead of the one under
        the key, seeded by ``seed``. The floor a keyed retrieval has to beat.
    """

    def __init__(
        self,
        capacity: int | None = DEFAULT_CAPACITY,
        *,
        overwrite: bool = True,
        verify_key: bool = True,
        retrieval: bool = True,
        random_retrieval: bool = False,
        seed: int = 0,
    ) -> None:
        if capacity is not None:
            if isinstance(capacity, bool) or not isinstance(capacity, int):
                raise ValueError("capacity must be an int or None")
            if capacity < MIN_CAPACITY:
                raise ValueError(f"capacity must be >= {MIN_CAPACITY}")
        if not retrieval and random_retrieval:
            raise ValueError("retrieval disabled and random retrieval are exclusive")
        if not verify_key:
            if capacity is None:
                raise ValueError(
                    "untagged retrieval is only defined for a bounded store: "
                    "without the tag there is nothing to check a growing log "
                    "against, and slots move when the array grows"
                )
            if not overwrite:
                raise ValueError(
                    "an untagged store cannot tell a write from a collision, so "
                    "first-write-wins is undefined"
                )

        self.bounded = capacity is not None
        self.initial_capacity = MIN_CAPACITY if capacity is None else capacity
        self.capacity = self.initial_capacity
        self.overwrite = overwrite
        self.verify_key = verify_key
        self.retrieval = retrieval
        self.random_retrieval = random_retrieval
        self.seed = seed
        self._rng = np.random.default_rng(seed) if random_retrieval else None

        self.keys: np.ndarray | None = None
        self.values = np.zeros(self.capacity, dtype=np.int64)
        self.present = np.zeros(self.capacity, dtype=bool)
        if verify_key:
            self.keys = np.zeros(self.capacity, dtype=np.int64)

        # Counters. Kept because "it remembered" is a claim that needs to be
        # countable in both directions: a hit is only a hit next to the misses
        # and the wrong-fact answers it is being compared with.
        self.writes = 0
        self.overwritten = 0
        self.evictions = 0
        self.hits = 0
        self.misses = 0
        self.random_draws = 0
        self.count = 0  # live entries for an unbounded store

    # -- bytes ---------------------------------------------------------------

    def bytes(self) -> int:
        """Bytes actually allocated, measured from the arrays themselves.

        Constant for a bounded store, whatever is written into it; a staircase for
        an unbounded one, because a growing ``numpy`` array over-allocates by
        doubling. Both are reported, because the arithmetic floor (17 bytes a
        write) and the allocated bytes are different numbers and only one of them
        is what a process would actually hold.
        """
        total = int(self.values.nbytes) + int(self.present.nbytes)
        if self.keys is not None:
            total += int(self.keys.nbytes)
        return total

    def bytes_per_slot(self) -> int:
        return TAGGED_SLOT_BYTES if self.verify_key else UNTAGGED_SLOT_BYTES

    def spec(self) -> dict:
        """The constructor arguments, so a run can be replayed from its record."""
        return {
            "capacity": None if not self.bounded else self.initial_capacity,
            "overwrite": self.overwrite,
            "verify_key": self.verify_key,
            "retrieval": self.retrieval,
            "random_retrieval": self.random_retrieval,
            "seed": self.seed,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "MemoryStore":
        return cls(**spec)

    # -- internals -----------------------------------------------------------

    def slot(self, key: int) -> int:
        """Where a key lives: ``key % capacity``, and nothing cleverer."""
        return int(key) % self.capacity

    def _grow(self) -> None:
        """Double an unbounded store. Slots are unstable here, so writes append."""
        if self.bounded:  # pragma: no cover - defensive
            raise ValueError("a bounded store does not grow")
        new_capacity = max(MIN_CAPACITY, self.capacity * 2)
        values = np.zeros(new_capacity, dtype=np.int64)
        present = np.zeros(new_capacity, dtype=bool)
        values[: self.capacity] = self.values
        present[: self.capacity] = self.present
        self.values, self.present = values, present
        if self.keys is not None:
            keys = np.zeros(new_capacity, dtype=np.int64)
            keys[: self.capacity] = self.keys
            self.keys = keys
        self.capacity = new_capacity

    def _live(self) -> np.ndarray:
        """Indices of occupied slots."""
        return np.flatnonzero(self.present)

    # -- writing -------------------------------------------------------------

    def write(self, key: int, value: int) -> int | None:
        """Store ``value`` under ``key``. Returns the key that was evicted, if any.

        Bounded: the slot is ``key % capacity``. Writing a key that already holds
        its own value replaces it in place (``overwrite=True``) or is ignored
        (``overwrite=False``, the stale control). Writing a *different* key into an
        occupied slot evicts it — the moment a bounded store starts forgetting,
        and the reason the capacity sweep has a curve in it.
        """
        self.writes += 1
        if self.bounded:
            index = self.slot(key)
            if self.keys is None:  # untagged: the slot is all there is
                self.evictions += int(self.present[index])
                self.values[index] = value
                self.present[index] = True
                return None
            if self.present[index]:
                held = int(self.keys[index])
                if held == key:
                    if not self.overwrite:
                        self.overwritten += 1
                        return None
                    self.values[index] = value
                    return None
                self.evictions += 1
                self.keys[index] = key
                self.values[index] = value
                return held
            self.keys[index] = key
            self.values[index] = value
            self.present[index] = True
            return None

        # Unbounded: append, and let retrieval scan. The log is what costs bytes.
        if self.count >= self.capacity:
            self._grow()
        if self.keys is not None:
            self.keys[self.count] = key
        self.values[self.count] = value
        self.present[self.count] = True
        self.count += 1
        return None

    # -- reading -------------------------------------------------------------

    def retrieve(self, key: int) -> int | None:
        """The value stored under ``key``, or ``None`` for a miss.

        ``None`` is a *miss* and never a guess: a tagged store returns nothing
        rather than another key's value, which is the difference the precision
        table measures.
        """
        if not self.retrieval:
            self.misses += 1
            return None

        if self._rng is not None:
            live = self._live()
            if live.size == 0:
                self.misses += 1
                return None
            self.random_draws += 1
            self.hits += 1
            return int(self.values[int(live[int(self._rng.integers(live.size))])])

        if self.bounded:
            index = self.slot(key)
            if not self.present[index]:
                self.misses += 1
                return None
            if self.keys is None:
                # Untagged: the slot answers for whoever asked for it.
                self.hits += 1
                return int(self.values[index])
            if int(self.keys[index]) != int(key):
                # The fact under this key was evicted by a later one. This is a
                # miss, not a wrong answer -- the tag is what buys that.
                self.misses += 1
                return None
            self.hits += 1
            return int(self.values[index])

        # Unbounded: last write wins (or first, for the stale control).
        assert self.keys is not None
        matching = np.flatnonzero(self.present & (self.keys == int(key)))
        if matching.size == 0:
            self.misses += 1
            return None
        self.hits += 1
        # Last write wins; the stale control keeps the first instead.
        chosen = int(matching[-1]) if self.overwrite else int(matching[0])
        return int(self.values[chosen])

    def summary(self) -> dict:
        """The store's accounting, for a results file rather than a claim."""
        return {
            "capacity": None if not self.bounded else self.initial_capacity,
            "slots_allocated": int(self.capacity),
            "verify_key": self.verify_key,
            "overwrite": self.overwrite,
            "retrieval": self.retrieval,
            "random_retrieval": self.random_retrieval,
            "bytes": self.bytes(),
            "bytes_per_slot": self.bytes_per_slot(),
            "writes": self.writes,
            "overwritten": self.overwritten,
            "evictions": self.evictions,
            "hits": self.hits,
            "misses": self.misses,
            "random_draws": self.random_draws,
        }
