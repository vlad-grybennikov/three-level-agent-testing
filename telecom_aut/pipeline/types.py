"""Typed interfaces between pipeline stages.

These models ARE the Level-3 contract: each stage consumes and produces
exactly these types, so frozen inputs and expected outputs serialise cleanly.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Intent labels follow the pilot-fiber event taxonomy where a counterpart
# exists (cancel_order_*, service_update_*, reschedule_appointment_*).
# subscriber_info covers record/field questions ("what is Erin's active
# order id?"). view_appointments is the read-only appointment listing.
# Both are negative controls for the Level-1 oracle (no state change).
INTENTS = (
    "subscriber_info",
    "view_appointments",
    "reschedule_appointment",
    "service_update",
    "cancel_order",
    "unsupported",
)
Intent = Literal[
    "subscriber_info", "view_appointments", "reschedule_appointment",
    "service_update", "cancel_order", "unsupported",
]


class IntentResult(BaseModel):
    """Stage 1 output."""

    intent: Intent


class SlotSet(BaseModel):
    """Stage 2 output: typed slots, all optional (null = not mentioned).

    extra="ignore": a model that invents slot names does not crash the
    pipeline. Invented slots are simply dropped (and visible only as their
    absence, a classic extraction defect signature).
    """

    model_config = ConfigDict(extra="ignore")

    subscriber_name: Optional[str] = None
    subscriber_email: Optional[str] = None
    order_id: Optional[str] = None          # ORD-XXXX if the customer named one
    appointment_id: Optional[str] = None    # APT-XXXX if the customer named one
    target_plan: Optional[str] = None       # plan code or name mentioned
    target_date: Optional[str] = None       # YYYY-MM-DD
    target_time_window: Optional[str] = None
    requested_field: Optional[str] = None   # for field-level info questions
    notes: Optional[str] = None


class SelectionInput(BaseModel):
    """Frozen input for stages 3 and 4: intent + slots + working history."""

    intent: Intent
    slots: SlotSet
    history: list[str] = Field(default_factory=list)


class ToolCandidate(BaseModel):
    tool: str
    score: float


class SelectionResult(BaseModel):
    """Stage 3 output: ranked top-k candidates (Recall@k is measured on
    `candidates`, while the loop only ever executes `top1`)."""

    candidates: list[ToolCandidate]

    @property
    def top1(self) -> Optional[str]:
        return self.candidates[0].tool if self.candidates else None


class BoundCall(BaseModel):
    """Stage 4 output: a schema-validated tool call, ready to execute."""

    tool: str
    args: dict
