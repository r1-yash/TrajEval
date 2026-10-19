"""Step 11: the travel_booking environment -- physics, not rules.

Two contracts: the world must NOT enforce the task spec (booking first
class and paying with points must work -- TrajEval's graders judge those
moves), and it must reject only what is physically impossible, leaving the
state untouched when it does.
"""

from __future__ import annotations

from trajecteval.envs import ActionRejected, TravelBookingEnvironment
from trajecteval.models import Action


def test_world_does_not_enforce_the_task_rules() -> None:
    # cabin 'first' and method 'points' break the spec's rules on purpose;
    # here they must simply work, because the graders are the judge.
    env = TravelBookingEnvironment()
    env.run_action(Action("search_flights", {"origin": "SFO", "destination": "NRT"}))
    env.run_action(
        Action("book_flight", {"flight": "TR420", "cabin": "first", "passengers": 1})
    )
    state = env.run_action(Action("pay", {"method": "points"}))
    assert state["status"] == "confirmed"
    assert state["cabin"] == "first"
    assert state["payment_method"] == "points"


def test_physically_impossible_actions_are_rejected_with_unchanged_state() -> None:
    # Two kinds of impossible: a precondition (pay before anything is
    # booked) and an unknown action name. Both raise, both change nothing.
    env = TravelBookingEnvironment()
    for action in (
        Action("pay", {"method": "card"}),
        Action("teleport", {}),
    ):
        before = env.current_state()
        try:
            env.run_action(action)
        except ActionRejected:
            pass
        else:  # pragma: no cover - a missing raise is the bug this guards
            raise AssertionError(f"{action.name!r} should have been rejected")
        assert env.current_state() == before
