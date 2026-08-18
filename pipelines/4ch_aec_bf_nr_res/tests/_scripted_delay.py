"""Scripted shared-delay shim used by test_delay_parity.py and test_pipeline.py.

Publishes a scripted ``(delay, solid)`` per hop in place of a live DelayAec3.
Scripted rather than provoked, for the same reason the C side drives its
admission state machine directly (tests/test_4aec_nr_res.c): a live estimator
re-offers a movement on every hop once it has one, so a held candidate is
always resolved on the very next eligible hop and its TTL never runs out;
and the hop a lock is PUBLISHED on must be chosen by the scene, not by the
estimator, when the test is about what the pipeline does on that hop.
"""


class ScriptedDelay:
    def __init__(self, script):
        self._script = list(script)
        self._index = -1

    def accumulate(self, capture, render):
        self._index += 1

    def reset(self):
        self._index = -1

    @property
    def _current(self):
        return self._script[min(max(self._index, 0), len(self._script) - 1)]

    @property
    def estimated_delay(self):
        return self._current[0]

    @property
    def confidence(self):
        return 1.0 if self._current[1] else 0.5

    @property
    def is_solid(self):
        return bool(self._current[1])

    @property
    def _n_updates(self):
        return 3
