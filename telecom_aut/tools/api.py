"""Permissive pseudo-API layer over TelecomEnv (subscriber/order model).

The central design rule: these tools are dumb CRUD.

  Rejected (structural validity only):
    * unknown / mistyped / mis-formatted arguments  -> "invalid_args"
    * references to nonexistent rows                -> "not_found"
    * ambiguous subscriber lookup                   -> "ambiguous"
    * impossible state transitions                  -> "invalid_state"
      (booking a booked slot, releasing a free slot, cancelling a cancelled
       order, service update on a non-active order)

  Executed without complaint (policy, the agent's responsibility):
    * cancelling an order with an unpaid balance
    * cancelling with no prior request_confirmation
    * service updates to legacy / disallowed plans
    * bookings with wrong-region technicians
    * rescheduling that strands or double-points slots

Every write appends one row to `events` inside the same transaction, stamped
with env.tick(). Rejected calls perform all checks *before* the first tick,
so they leave neither state nor clock changes behind.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable

from pydantic import ValidationError

from ..environment import TelecomEnv
from ..retrieval import RetrievalConfig, Retriever
from . import schemas as S


class ToolRejection(Exception):
    """A structurally invalid call. Never raised for policy violations."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def _dump(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


_APPT_SELECT = (
    "SELECT a.*, s.starts_at AS date, s.ends_at AS date_end"
    " FROM appointments a"
    " LEFT JOIN availability_slots s ON s.id = a.slot_id"
)


class TelecomAPI:
    """All pseudo-API tools, bound to one environment instance."""

    def __init__(self, env: TelecomEnv, actor: str = "agent",
                 retriever: Retriever | None = None) -> None:
        self.env = env
        self.actor = actor
        # Injectable so the config layer can perturb retrieval (surface #4).
        self.retriever = retriever or Retriever(env, RetrievalConfig())

    # -- plumbing -----------------------------------------------------------

    def _fetch(self, table: str, row_id) -> dict | None:
        row = self.env.conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (row_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def _require(self, table: str, row_id, label: str) -> dict:
        row = self._fetch(table, row_id)
        if row is None:
            raise ToolRejection("not_found", f"no {label} with id {row_id}")
        return row

    @contextmanager
    def _write(self):
        try:
            yield self.env.conn
            self.env.conn.commit()
        except Exception:
            self.env.conn.rollback()
            raise

    def _log(self, conn, sim_time: str, operation: str, entity_type: str,
             entity_id: Any, payload: dict) -> None:
        conn.execute(
            "INSERT INTO events (sim_time, actor, operation, entity_type,"
            " entity_id, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (sim_time, self.actor, operation, entity_type,
             None if entity_id is None else str(entity_id), _dump(payload)),
        )

    # -- read tools ---------------------------------------------------------

    def find_subscriber(self, args: S.FindSubscriberArgs) -> dict:
        q = args.name_or_email.strip().lower()
        rows = [
            dict(r) for r in self.env.conn.execute(
                "SELECT * FROM subscribers"
                " WHERE lower(full_name) = ? OR lower(email) = ? ORDER BY id",
                (q, q),
            )
        ]
        if not rows:
            # Deterministic suggestions: substring matches, ordered by id.
            hints = [
                dict(r) for r in self.env.conn.execute(
                    "SELECT full_name, email FROM subscribers"
                    " WHERE instr(lower(full_name), ?) > 0"
                    "    OR instr(lower(email), ?) > 0"
                    " ORDER BY id LIMIT 3",
                    (q, q),
                )
            ]
            message = f"no subscriber matches {args.name_or_email!r}"
            if hints:
                closest = ", ".join(
                    f"{h['full_name']} <{h['email']}>" for h in hints
                )
                message += (f"; closest matches: {closest} — retry with the"
                            " exact full name or email")
            raise ToolRejection("not_found", message)
        if len(rows) > 1:
            ids = [r["id"] for r in rows]
            raise ToolRejection(
                "ambiguous",
                f"{args.name_or_email!r} matches subscriber ids {ids};"
                " retry with an email address",
            )
        return rows[0]

    def list_orders(self, args: S.ListOrdersArgs) -> list[dict]:
        self._require("subscribers", args.subscriber_id, "subscriber")
        return [
            dict(r) for r in self.env.conn.execute(
                "SELECT o.*, p.code AS plan_code, p.name AS plan_name"
                " FROM orders o JOIN plans p ON p.id = o.plan_id"
                " WHERE o.subscriber_id = ? ORDER BY o.id",
                (args.subscriber_id,),
            )
        ]

    def get_order(self, args: S.GetOrderArgs) -> dict:
        row = self.env.conn.execute(
            "SELECT o.*, p.code AS plan_code, p.name AS plan_name"
            " FROM orders o JOIN plans p ON p.id = o.plan_id WHERE o.id = ?",
            (args.order_id,),
        ).fetchone()
        if row is None:
            raise ToolRejection("not_found", f"no order with id {args.order_id}")
        return dict(row)

    def list_plans(self, args: S.ListPlansArgs) -> list[dict]:
        return [
            dict(r) for r in
            self.env.conn.execute("SELECT * FROM plans ORDER BY id")
        ]

    def list_appointments(self, args: S.ListAppointmentsArgs) -> list[dict]:
        self._require("subscribers", args.subscriber_id, "subscriber")
        sql = (_APPT_SELECT +
               " JOIN orders o ON o.id = a.order_id WHERE o.subscriber_id = ?")
        params: list[Any] = [args.subscriber_id]
        if args.status is not None:
            sql += " AND a.status = ?"
            params.append(args.status)
        return [dict(r) for r in
                self.env.conn.execute(sql + " ORDER BY a.id", params)]

    def get_appointment(self, args: S.GetAppointmentArgs) -> dict:
        row = self.env.conn.execute(
            _APPT_SELECT + " WHERE a.id = ?", (args.appointment_id,)
        ).fetchone()
        if row is None:
            raise ToolRejection(
                "not_found", f"no appointment with id {args.appointment_id}"
            )
        return dict(row)

    def list_available_slots(self, args: S.ListAvailableSlotsArgs) -> list[dict]:
        sql = (
            "SELECT s.*, t.region FROM availability_slots s"
            " JOIN technicians t ON t.id = s.technician_id"
            " WHERE s.status = 'available'"
        )
        params: list[Any] = []
        if args.region is not None:
            sql += " AND t.region = ?"
            params.append(args.region)
        if args.date is not None:
            sql += " AND s.starts_at LIKE ?"
            params.append(args.date + "%")
        if args.technician_id is not None:
            sql += " AND s.technician_id = ?"
            params.append(args.technician_id)
        return [dict(r) for r in self.env.conn.execute(sql + " ORDER BY s.id", params)]

    def list_invoices(self, args: S.ListInvoicesArgs) -> list[dict]:
        self._require("orders", args.order_id, "order")
        sql = "SELECT * FROM invoices WHERE order_id = ?"
        params: list[Any] = [args.order_id]
        if args.status is not None:
            sql += " AND status = ?"
            params.append(args.status)
        return [dict(r) for r in self.env.conn.execute(sql + " ORDER BY id", params)]

    def search_policy(self, args: S.SearchPolicyArgs) -> list[dict]:
        """Retrieve policy documents. A read that logs no event. Its
        presence is checked in the *trace* (Level 2), not in state."""
        return self.retriever.search(args.query)

    # -- write tools --------------------------------------------------------

    def book_slot(self, args: S.BookSlotArgs) -> dict:
        slot = self._require("availability_slots", args.slot_id, "slot")
        if slot["status"] != "available":
            raise ToolRejection(
                "invalid_state", f"slot {args.slot_id} is not available"
            )
        with self._write() as conn:
            ts = self.env.tick()
            conn.execute(
                "UPDATE availability_slots SET status = 'booked' WHERE id = ?",
                (args.slot_id,),
            )
            self._log(conn, ts, "book_slot", "availability_slots",
                      args.slot_id, {"from": "available", "to": "booked"})
        return self._fetch("availability_slots", args.slot_id)

    def release_slot(self, args: S.ReleaseSlotArgs) -> dict:
        slot = self._require("availability_slots", args.slot_id, "slot")
        if slot["status"] != "booked":
            raise ToolRejection(
                "invalid_state", f"slot {args.slot_id} is not booked"
            )
        with self._write() as conn:
            ts = self.env.tick()
            conn.execute(
                "UPDATE availability_slots SET status = 'available' WHERE id = ?",
                (args.slot_id,),
            )
            self._log(conn, ts, "release_slot", "availability_slots",
                      args.slot_id, {"from": "booked", "to": "available"})
        return self._fetch("availability_slots", args.slot_id)

    def update_appointment(self, args: S.UpdateAppointmentArgs) -> dict:
        self._require("appointments", args.appointment_id, "appointment")
        provided = [
            f for f in ("slot_id", "status", "notes")  # fixed order
            if f in args.model_fields_set
        ]
        if not provided:
            raise ToolRejection(
                "invalid_args", "no fields to update were provided"
            )
        if "slot_id" in provided and args.slot_id is not None:
            self._require("availability_slots", args.slot_id, "slot")
        if "status" in provided and args.status is None:
            raise ToolRejection(  # structural: status column is NOT NULL
                "invalid_args", "status cannot be set to null"
            )

        changes = {f: getattr(args, f) for f in provided}
        set_clause = ", ".join(f"{f} = ?" for f in provided)
        with self._write() as conn:
            ts = self.env.tick()
            conn.execute(
                f"UPDATE appointments SET {set_clause} WHERE id = ?",
                [changes[f] for f in provided] + [args.appointment_id],
            )
            self._log(conn, ts, "update_appointment", "appointments",
                      args.appointment_id, {"changes": changes})
        return self._fetch("appointments", args.appointment_id)

    def update_order_plan(self, args: S.UpdateOrderPlanArgs) -> dict:
        order = self._require("orders", args.order_id, "order")
        if order["state"] != "active":
            raise ToolRejection(
                "invalid_state",
                f"order {args.order_id} is not active"
                f" (state: {order['state']})",
            )
        plan = self.env.conn.execute(
            "SELECT * FROM plans WHERE code = ?", (args.new_plan_code,)
        ).fetchone()
        if plan is None:
            raise ToolRejection(
                "not_found", f"no plan with code {args.new_plan_code!r}"
            )
        old_plan = self._fetch("plans", order["plan_id"])
        # NOTE: no allowed-plan or confirmation checks here, by design.
        # Switching to a legacy plan executes successfully.
        with self._write() as conn:
            ts = self.env.tick()
            conn.execute(
                "UPDATE orders SET plan_id = ? WHERE id = ?",
                (plan["id"], args.order_id),
            )
            self._log(conn, ts, "update_order_plan", "orders", args.order_id,
                      {"from_plan": old_plan["code"], "to_plan": plan["code"]})
        return self.get_order(S.GetOrderArgs(order_id=args.order_id))

    def cancel_order(self, args: S.CancelOrderArgs) -> dict:
        order = self._require("orders", args.order_id, "order")
        if order["state"] not in ("active", "pending_installation"):
            raise ToolRejection(
                "invalid_state", f"order {args.order_id} is already cancelled"
            )
        # NOTE: no balance, appointment, or confirmation checks. An
        # unconfirmed cancellation with unpaid invoices must succeed.
        with self._write() as conn:
            ts = self.env.tick()
            conn.execute(
                "UPDATE orders SET state = 'cancelled', cancelled_at = ?"
                " WHERE id = ?",
                (ts, args.order_id),
            )
            self._log(conn, ts, "cancel_order", "orders", args.order_id,
                      {"from": order["state"], "to": "cancelled"})
        return self.get_order(S.GetOrderArgs(order_id=args.order_id))

    def request_confirmation(self, args: S.RequestConfirmationArgs) -> dict:
        """Single-turn stand-in for a confirmation dialogue.

        Always returns a canned affirmative. What matters is that the call
        appears in the trace and the events log *before* the destructive
        write it covers.
        """
        with self._write() as conn:
            ts = self.env.tick()
            self._log(conn, ts, "request_confirmation", "confirmation", None,
                      {"action_summary": args.action_summary})
        return {"confirmed": True, "customer_response": "Yes, please proceed."}


# -- registry ---------------------------------------------------------------

# name -> (argument schema, unbound method). Fixed order: reads, writes,
# confirmation. Stage 3 (tool selection) ranks over exactly this space.
TOOL_REGISTRY: dict[str, tuple[type[S.ToolArgs], Callable]] = {
    "find_subscriber": (S.FindSubscriberArgs, TelecomAPI.find_subscriber),
    "list_orders": (S.ListOrdersArgs, TelecomAPI.list_orders),
    "get_order": (S.GetOrderArgs, TelecomAPI.get_order),
    "list_plans": (S.ListPlansArgs, TelecomAPI.list_plans),
    "list_appointments": (S.ListAppointmentsArgs, TelecomAPI.list_appointments),
    "get_appointment": (S.GetAppointmentArgs, TelecomAPI.get_appointment),
    "list_available_slots": (S.ListAvailableSlotsArgs, TelecomAPI.list_available_slots),
    "list_invoices": (S.ListInvoicesArgs, TelecomAPI.list_invoices),
    "search_policy": (S.SearchPolicyArgs, TelecomAPI.search_policy),
    "book_slot": (S.BookSlotArgs, TelecomAPI.book_slot),
    "release_slot": (S.ReleaseSlotArgs, TelecomAPI.release_slot),
    "update_appointment": (S.UpdateAppointmentArgs, TelecomAPI.update_appointment),
    "update_order_plan": (S.UpdateOrderPlanArgs, TelecomAPI.update_order_plan),
    "cancel_order": (S.CancelOrderArgs, TelecomAPI.cancel_order),
    "request_confirmation": (S.RequestConfirmationArgs, TelecomAPI.request_confirmation),
}

TOOL_NAMES: list[str] = list(TOOL_REGISTRY)

WRITE_TOOLS: set[str] = {
    "book_slot", "release_slot", "update_appointment",
    "update_order_plan", "cancel_order", "request_confirmation",
}


def invoke(api: TelecomAPI, tool_name: str, raw_args: dict) -> Any:
    """Validate raw arguments against the tool's schema and dispatch.

    This is the single entry point the agent loop uses. Direct method calls
    with pre-built schema objects are the unit-test path.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ToolRejection("not_found", f"unknown tool {tool_name!r}")
    args_model, method = TOOL_REGISTRY[tool_name]
    try:
        args = args_model.model_validate(raw_args)
    except ValidationError as exc:
        errs = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ToolRejection("invalid_args", f"{tool_name}: {errs}") from exc
    return method(api, args)
