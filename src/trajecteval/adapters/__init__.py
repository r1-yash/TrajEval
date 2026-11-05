# Offline conversion from already-loaded external agent runs to TrajEval
# trajectories. Format-specific adapters (for example a Debuggernaut adapter
# in a later milestone) live beside this foundation; this package knows no
# external field names and performs no evaluation.

from trajecteval.adapters.offline import (
    UNAVAILABLE_STATE_KEY,
    ExternalStep,
    OfflineRunAdapter,
    is_unavailable_state,
    unavailable_state,
)

__all__ = [
    "ExternalStep",
    "OfflineRunAdapter",
    "UNAVAILABLE_STATE_KEY",
    "is_unavailable_state",
    "unavailable_state",
]
