"""Pseudo-API tests: permissive CRUD, events log, structural-only rejection.

The load-bearing test class is TestPolicyViolationsExecute: the API layer
must happily perform policy-violating writes. If any of those tests ever
needs an API-side guard to pass, the study design is broken.
"""

import pytest

from telecom_aut.environment import TelecomEnv
from telecom_aut.tools import TelecomAPI, ToolRejection, invoke
from telecom_aut.tools import schemas as S


@pytest.fixture()
def env():
    e = TelecomEnv.fresh()
    yield e
    e.close()


@pytest.fixture()
def api(env):
    return TelecomAPI(env)


def events(env):
    return env.snapshot()["tables"]["events"]


# -- reads ------------------------------------------------------------------

class TestReads:
    def test_find_subscriber_by_name_and_email(self, api):
        by_name = api.find_subscriber(
            S.FindSubscriberArgs(name_or_email="Alice Nguyen")
        )
        by_email = api.find_subscriber(
            S.FindSubscriberArgs(name_or_email="ALICE.NGUYEN@example.com")
        )
        assert by_name["id"] == by_email["id"] == "SUB-0001"
        assert by_name["state"] == "active"

    def test_find_subscriber_not_found_without_similar_names(self, api):
        with pytest.raises(ToolRejection) as exc:
            api.find_subscriber(S.FindSubscriberArgs(name_or_email="Zzz Qqq"))
        assert exc.value.code == "not_found"
        assert "closest matches" not in exc.value.message

    def test_find_subscriber_suggests_closest_matches(self, api):
        with pytest.raises(ToolRejection) as exc:
            api.find_subscriber(S.FindSubscriberArgs(name_or_email="erin"))
        assert exc.value.code == "not_found"  # contract stays strict
        assert "Erin Walsh <erin.walsh@example.com>" in exc.value.message

    def test_find_subscriber_ambiguous(self, api, env):
        env.conn.execute(
            "INSERT INTO subscribers (id, full_name, email, state, created_at)"
            " VALUES ('SUB-0099', 'Alice Nguyen', 'alice2@example.com',"
            " 'active', '2025-01-01T00:00:00Z')"
        )
        env.conn.commit()
        with pytest.raises(ToolRejection) as exc:
            api.find_subscriber(S.FindSubscriberArgs(name_or_email="Alice Nguyen"))
        assert exc.value.code == "ambiguous"

    def test_list_orders_multi_order_subscriber(self, api):
        orders = api.list_orders(S.ListOrdersArgs(subscriber_id="SUB-0004"))
        assert [o["id"] for o in orders] == ["ORD-0103", "ORD-0106"]
        states = {o["id"]: o["state"] for o in orders}
        assert states["ORD-0103"] == "active"
        assert states["ORD-0106"] == "pending_installation"
        assert orders[0]["plan_code"] == "fiber-2gig"

    def test_get_order_includes_plan(self, api):
        order = api.get_order(S.GetOrderArgs(order_id="ORD-0100"))
        assert order["plan_code"] == "fiber-1gig"
        assert order["region"] == "north"

    def test_list_appointments_includes_visit_date(self, api):
        pending = api.list_appointments(
            S.ListAppointmentsArgs(subscriber_id="SUB-0002", status="pending")
        )
        assert [a["id"] for a in pending] == ["APT-0405"]
        assert pending[0]["date"] == "2026-08-04T14:00:00Z"
        assert pending[0]["kind"] == "maintenance"

    def test_appointments_span_past_and_future(self, api):
        all_erin = api.list_appointments(
            S.ListAppointmentsArgs(subscriber_id="SUB-0005")
        )
        assert [a["id"] for a in all_erin] == ["APT-0402"]
        assert all_erin[0]["status"] == "completed"
        assert all_erin[0]["date"] == "2026-07-10T09:00:00Z"  # past visit

    def test_list_available_slots_filters(self, api):
        north = api.list_available_slots(S.ListAvailableSlotsArgs(region="north"))
        assert {s["technician_id"] for s in north} == {200}
        assert "SLT-0303" not in {s["id"] for s in north}  # booked -> excluded

        aug4 = api.list_available_slots(S.ListAvailableSlotsArgs(date="2026-08-04"))
        assert aug4 and all(s["starts_at"].startswith("2026-08-04") for s in aug4)

    def test_list_invoices_unpaid_filter(self, api):
        unpaid = api.list_invoices(
            S.ListInvoicesArgs(order_id="ORD-0101", status="unpaid")
        )
        assert [i["id"] for i in unpaid] == ["INV-0502"]

    def test_reads_log_no_events_and_change_nothing(self, api, env):
        before = env.snapshot()
        api.list_plans(S.ListPlansArgs())
        api.get_appointment(S.GetAppointmentArgs(appointment_id="APT-0400"))
        api.list_available_slots(S.ListAvailableSlotsArgs())
        api.search_policy(S.SearchPolicyArgs(query="cancellation"))
        assert env.snapshot() == before
        assert events(env) == []


# -- write lifecycle and events log -----------------------------------------

class TestWrites:
    def test_book_and_release_lifecycle(self, api, env):
        booked = api.book_slot(S.BookSlotArgs(slot_id="SLT-0300"))
        assert booked["status"] == "booked"
        released = api.release_slot(S.ReleaseSlotArgs(slot_id="SLT-0300"))
        assert released["status"] == "available"

        ops = [(e["operation"], e["entity_id"]) for e in events(env)]
        assert ops == [("book_slot", "SLT-0300"), ("release_slot", "SLT-0300")]

    def test_events_are_ordered_and_clock_stamped(self, api, env):
        api.book_slot(S.BookSlotArgs(slot_id="SLT-0300"))
        api.book_slot(S.BookSlotArgs(slot_id="SLT-0301"))
        api.release_slot(S.ReleaseSlotArgs(slot_id="SLT-0300"))
        evs = events(env)
        assert [e["seq"] for e in evs] == [1, 2, 3]
        times = [e["sim_time"] for e in evs]
        assert times == sorted(times) and len(set(times)) == 3
        assert times[0] == "2026-07-28T09:00:01Z"

    def test_update_appointment_is_partial(self, api):
        appt = api.update_appointment(
            S.UpdateAppointmentArgs(appointment_id="APT-0400", notes="Rebooked")
        )
        assert appt["notes"] == "Rebooked"
        assert appt["slot_id"] == "SLT-0303" and appt["status"] == "pending"

    def test_update_appointment_explicit_null_detaches_slot(self, api):
        appt = invoke(api, "update_appointment",
                      {"appointment_id": "APT-0400", "slot_id": None})
        assert appt["slot_id"] is None

    def test_update_appointment_no_fields_rejected(self, api):
        with pytest.raises(ToolRejection) as exc:
            api.update_appointment(
                S.UpdateAppointmentArgs(appointment_id="APT-0400")
            )
        assert exc.value.code == "invalid_args"

    def test_request_confirmation_is_canned_and_logged(self, api, env):
        out = api.request_confirmation(
            S.RequestConfirmationArgs(action_summary="Cancel order ORD-0100")
        )
        assert out == {"confirmed": True,
                       "customer_response": "Yes, please proceed."}
        ev = events(env)[0]
        assert ev["operation"] == "request_confirmation"
        assert "ORD-0100" in ev["payload"]


# -- structural rejection ---------------------------------------------------

class TestStructuralRejection:
    @pytest.mark.parametrize("tool,args,code", [
        ("book_slot", {"slot_id": "SLT-9999"}, "not_found"),
        ("book_slot", {"slot_id": "SLT-0303"}, "invalid_state"),  # booked
        ("release_slot", {"slot_id": "SLT-0300"}, "invalid_state"),
        ("cancel_order", {"order_id": "ORD-0105"}, "invalid_state"),  # cancelled
        ("update_order_plan",
         {"order_id": "ORD-0105", "new_plan_code": "fiber-500"}, "invalid_state"),
        # The user-specified structural rule: service update requires an
        # ACTIVE order, pending_installation does not qualify.
        ("update_order_plan",
         {"order_id": "ORD-0106", "new_plan_code": "fiber-2gig"}, "invalid_state"),
        ("update_order_plan",
         {"order_id": "ORD-0100", "new_plan_code": "fiber-9000"}, "not_found"),
        ("book_slot", {"slot_id": 300}, "invalid_args"),        # bare int
        ("book_slot", {"slot_id": "ORD-0100"}, "invalid_args"),  # wrong prefix
        ("update_appointment",
         {"appointment_id": "SUB-0001", "notes": "x"}, "invalid_args"),
        ("book_slot", {"slot_id": "SLT-0300", "extra": 1}, "invalid_args"),
        ("no_such_tool", {}, "not_found"),
    ])
    def test_rejection_codes(self, api, tool, args, code):
        with pytest.raises(ToolRejection) as exc:
            invoke(api, tool, args)
        assert exc.value.code == code

    def test_rejected_calls_leave_no_trace_at_all(self, api, env):
        before = env.snapshot()
        clock_before = env.sim_now
        for tool, args in [
            ("book_slot", {"slot_id": "SLT-0303"}),
            ("cancel_order", {"order_id": "ORD-0105"}),
            ("update_appointment", {"appointment_id": "APT-9999", "notes": "x"}),
        ]:
            with pytest.raises(ToolRejection):
                invoke(api, tool, args)
        assert env.snapshot() == before
        assert env.sim_now == clock_before


# -- THE requirement: policy violations execute -----------------------------

class TestPolicyViolationsExecute:
    """Correct behaviour = the API performs these. Policy is the agent's job."""

    def test_unconfirmed_unpaid_cancellation_succeeds(self, api, env):
        # Bruno: unpaid invoice, no request_confirmation, pending visit.
        # Three policy violations at once: the storage layer must not care.
        order = api.cancel_order(S.CancelOrderArgs(order_id="ORD-0101"))
        assert order["state"] == "cancelled"
        assert order["cancelled_at"] == "2026-07-28T09:00:01Z"
        assert [e["operation"] for e in events(env)] == ["cancel_order"]
        # And the events log shows NO prior confirmation, visible in state.

    def test_service_update_to_legacy_plan_succeeds(self, api):
        # Policy doc 602 forbids switching any order TO fiber-100 (legacy).
        # The API executes it anyway.
        order = api.update_order_plan(
            S.UpdateOrderPlanArgs(order_id="ORD-0100", new_plan_code="fiber-100")
        )
        assert order["plan_code"] == "fiber-100"

    def test_cancellation_with_pending_appointment_strands_slot(self, api):
        # Alice has pending APT-0400 on booked SLT-0303. Cancelling without
        # cleanup strands the slot, which is allowed here. Only the agent
        # should prevent it (policy doc 603). Level 1 sees the stranded slot.
        order = api.cancel_order(S.CancelOrderArgs(order_id="ORD-0100"))
        assert order["state"] == "cancelled"
        slot = api.env.conn.execute(
            "SELECT status FROM availability_slots WHERE id = 'SLT-0303'"
        ).fetchone()
        assert slot["status"] == "booked"  # stranded

    def test_cancelling_pending_installation_order_is_allowed(self, api):
        order = api.cancel_order(S.CancelOrderArgs(order_id="ORD-0106"))
        assert order["state"] == "cancelled"
        # Its installation visit is now orphaned unless the agent cleans up.
        appt = api.get_appointment(
            S.GetAppointmentArgs(appointment_id="APT-0406")
        )
        assert appt["status"] == "pending"  # policy violation, visible


# -- chain necessity --------------------------------------------------------

class TestChainNecessity:
    def test_single_update_cannot_accomplish_a_reschedule(self, api, env):
        """update_appointment alone leaves slot states wrong: the release/
        book steps are load-bearing, not conventional."""
        api.update_appointment(
            S.UpdateAppointmentArgs(appointment_id="APT-0400",
                                    slot_id="SLT-0304")
        )
        slots = {s["id"]: s["status"]
                 for s in env.snapshot()["tables"]["availability_slots"]}
        assert slots["SLT-0303"] == "booked"      # old slot stranded
        assert slots["SLT-0304"] == "available"   # new slot never booked

    def test_full_reschedule_chain_produces_consistent_state(self, api, env):
        api.release_slot(S.ReleaseSlotArgs(slot_id="SLT-0303"))
        api.book_slot(S.BookSlotArgs(slot_id="SLT-0304"))
        api.update_appointment(
            S.UpdateAppointmentArgs(appointment_id="APT-0400",
                                    slot_id="SLT-0304")
        )
        snap = env.snapshot()["tables"]
        slots = {s["id"]: s["status"] for s in snap["availability_slots"]}
        appt = next(a for a in snap["appointments"] if a["id"] == "APT-0400")
        assert slots["SLT-0303"] == "available" and slots["SLT-0304"] == "booked"
        assert appt["slot_id"] == "SLT-0304"
        assert [e["operation"] for e in snap["events"]] == [
            "release_slot", "book_slot", "update_appointment"
        ]


# -- determinism under tool traffic -----------------------------------------

def test_identical_call_sequences_produce_identical_snapshots():
    def run():
        env = TelecomEnv.fresh()
        api = TelecomAPI(env)
        invoke(api, "release_slot", {"slot_id": "SLT-0303"})
        invoke(api, "book_slot", {"slot_id": "SLT-0306"})
        invoke(api, "update_appointment",
               {"appointment_id": "APT-0400", "slot_id": "SLT-0306"})
        invoke(api, "request_confirmation", {"action_summary": "cancel order"})
        invoke(api, "cancel_order", {"order_id": "ORD-0100"})
        snap = env.snapshot()
        env.close()
        return snap

    assert run() == run()
