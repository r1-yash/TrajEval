#basically its a abstract base class for other graders, defines structure rules etc for the graders 

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import RULE_SECTIONS, TaskSpec



# This block is basically defining a contract/interface for anything that acts as a Grader.
# runtime_checkable allows us to use isinstance() to check if an object implements this protocol at runtime.
# protocol is like set of rules/methods/behaviour that an object is expected to provide so now pyuthon knows how to treat soemthing that inherits this class, what rule it must follow 
''''so basically this is a protocol that defines the structure and behavior that any Grader class should adhere to. It specifies that any Grader must have a dimension attribute and
a grade method that takes a TaskSpec and a Trajectory as input and returns a GraderResult. This allows for consistent grading behavior across different implementations of Graders.'''
@runtime_checkable
class Grader(Protocol):

    dimension: Dimension

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        # Judge one dimension of one trajectory against one task's rules.
        ...  # pragma: no cover - protocol body, never executed



# # a safe check befora grading starts  It receives:
# - spec → information about which task is supposed to be graded
# - trajectory → information about what the agent actually did
# - dimension → what aspect we're grading, e.g. correctness/security
# - GraderResult | None → it can return either a GraderResult or nothing (None)
'''here its mainly about that whether the task id matches the task id of trajectory coz if it dosnt then we return the verdit along with evidence'''

def task_id_mismatch_result(
    spec: TaskSpec, dimension: Dimension, trajectory: Trajectory
) -> GraderResult | None:
    # ERROR when spec(task name) and trajectory name are different tasks 
    if spec.task_id == trajectory.task_id:
        return None
    return GraderResult(
        dimension=dimension,
        verdict=Verdict.ERROR, #return that verdict is error if task id doesnt match trajectory name
        reason=(
            f"spec is for task {spec.task_id!r} but trajectory is for "
            f"{trajectory.task_id!r}; wrong pairing -- "
            f"cannot judge the {dimension.value} dimension"
        ),
        evidence=(      #gives evidence if it doesnt match 
            Evidence(
                kind="task_id_mismatch",
                step=None,
                field_name="task_id",
                value={"spec": spec.task_id, "trajectory": trajectory.task_id},
            ),
        ),
    )

#Check whether the TaskSpec actually contains the section that this particular grader needs.
'''liek for example if the grader is for correctness and the spec doesnt have a correctness section then we return an error along with evidence'''
def missing_section_result(
    spec: TaskSpec, dimension: Dimension, section: str
) -> GraderResult | None:
    if section not in RULE_SECTIONS:
        raise ValueError(
            f"invalid section {section!r}; expected one of {', '.join(RULE_SECTIONS)}"
        )
    if getattr(spec, section) is not None:
        return None
    return GraderResult(
        dimension=dimension,
        verdict=Verdict.ERROR,
        reason=(
            f"spec for task {spec.task_id!r} has no {section!r} section; "
            f"cannot judge the {dimension.value} dimension"
        ),
        evidence=(
            Evidence(kind="missing_spec_field", step=None, field_name=section),
        ),
    )
