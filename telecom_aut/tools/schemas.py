"""Pydantic argument schemas for every pseudo-API tool.

These schemas define *structural* validity only: field names, types, enums,
and id formats. They deliberately encode no policy: a schema-valid call may
still be a policy violation, and the API layer will execute it anyway.

Entity ids are prefixed hex strings. The pattern constraints make a
wrong-entity or malformed id fail fast as invalid_args instead of silently
hitting the wrong row. `extra="forbid"` so a binder hallucinating an
argument name is a visible structural failure, not a silently ignored key.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_DATE = r"^\d{4}-\d{2}-\d{2}$"
SUB_ID = r"^SUB-[0-9A-F]{4}$"
ORD_ID = r"^ORD-[0-9A-F]{4}$"
APT_ID = r"^APT-[0-9A-F]{4}$"
SLT_ID = r"^SLT-[0-9A-F]{4}$"


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- read tools -------------------------------------------------------------

class FindSubscriberArgs(ToolArgs):
    name_or_email: str = Field(min_length=1)


class ListOrdersArgs(ToolArgs):
    subscriber_id: str = Field(pattern=SUB_ID)


class GetOrderArgs(ToolArgs):
    order_id: str = Field(pattern=ORD_ID)


class ListPlansArgs(ToolArgs):
    pass


class ListAppointmentsArgs(ToolArgs):
    subscriber_id: str = Field(pattern=SUB_ID)
    status: Optional[Literal["pending", "completed", "cancelled"]] = None


class GetAppointmentArgs(ToolArgs):
    appointment_id: str = Field(pattern=APT_ID)


class ListAvailableSlotsArgs(ToolArgs):
    region: Optional[str] = None
    date: Optional[str] = Field(default=None, pattern=_DATE)
    technician_id: Optional[int] = None


class ListInvoicesArgs(ToolArgs):
    order_id: str = Field(pattern=ORD_ID)
    status: Optional[Literal["paid", "unpaid"]] = None


class SearchPolicyArgs(ToolArgs):
    query: str = Field(min_length=1)


# -- write tools ------------------------------------------------------------

class BookSlotArgs(ToolArgs):
    slot_id: str = Field(pattern=SLT_ID)


class ReleaseSlotArgs(ToolArgs):
    slot_id: str = Field(pattern=SLT_ID)


class UpdateAppointmentArgs(ToolArgs):
    """Partial update: only fields the caller actually passed are applied.

    Passing an explicit ``"slot_id": null`` detaches the appointment from its
    slot (used after releasing it); omitting ``slot_id`` leaves it unchanged.
    The distinction is read from ``model_fields_set``.
    """

    appointment_id: str = Field(pattern=APT_ID)
    slot_id: Optional[str] = Field(default=None, pattern=SLT_ID)
    status: Optional[Literal["pending", "completed", "cancelled"]] = None
    notes: Optional[str] = None


class UpdateOrderPlanArgs(ToolArgs):
    order_id: str = Field(pattern=ORD_ID)
    new_plan_code: str = Field(min_length=1)  # plans are referenced by code


class CancelOrderArgs(ToolArgs):
    order_id: str = Field(pattern=ORD_ID)


class RequestConfirmationArgs(ToolArgs):
    action_summary: str = Field(min_length=1)
