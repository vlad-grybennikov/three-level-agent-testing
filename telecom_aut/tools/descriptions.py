"""Baseline (clean) tool descriptions.

This dict is defect surface #2: the config layer presents
descriptions to the model from config, defaulting to these values. A
description defect is injected by overriding an entry in config, never by
editing this file.

Descriptions are faithful: they state what the tool does and its structural
preconditions, and they do NOT restate policy (policy lives in
policy_documents and must be retrieved).
"""

DEFAULT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "find_subscriber": (
        "Look up a subscriber by exact full name or email address. Returns "
        "the subscriber record (id like SUB-0001, state active/cancelled). "
        "Fails if no subscriber or more than one matches; a failed lookup "
        "lists the closest matching subscribers to retry with."
    ),
    "list_orders": (
        "List all orders for a subscriber id (SUB-XXXX), including order id "
        "(ORD-XXXX), state (active, pending_installation, cancelled), plan "
        "code, address, and region."
    ),
    "get_order": (
        "Fetch a single order by its id (ORD-XXXX), including plan and "
        "region."
    ),
    "list_plans": (
        "List all service plans with code, name, speed tier, monthly price "
        "in cents, and speed in Mbps."
    ),
    "list_appointments": (
        "List a subscriber's appointments across all their orders by "
        "subscriber id (SUB-XXXX), optionally filtered by status (pending, "
        "completed, or cancelled). Each row includes the appointment id "
        "(APT-XXXX), order id, kind, slot id, and visit date."
    ),
    "get_appointment": (
        "Fetch a single appointment by its id (APT-XXXX), including its "
        "slot, status, and visit date."
    ),
    "list_available_slots": (
        "List technician slots that are currently available (SLT-XXXX), "
        "optionally filtered by region, date (YYYY-MM-DD), or technician id."
    ),
    "list_invoices": (
        "List invoices for an order id (ORD-XXXX), optionally filtered by "
        "status (paid or unpaid)."
    ),
    "search_policy": (
        "Search company policy documents by keywords and return the most "
        "relevant policy texts. Consult policy before cancelling an order "
        "or changing a plan."
    ),
    "book_slot": (
        "Mark an available technician slot (SLT-XXXX) as booked. Fails if "
        "the slot does not exist or is already booked."
    ),
    "release_slot": (
        "Mark a booked technician slot (SLT-XXXX) as available again. Fails "
        "if the slot does not exist or is not currently booked."
    ),
    "update_appointment": (
        "Update fields of an appointment (APT-XXXX): slot_id (pass null to "
        "detach the appointment from its slot), status, and/or notes. Only "
        "the fields provided are changed. Does not book or release slots."
    ),
    "update_order_plan": (
        "Apply a service update: set an active order (ORD-XXXX) to a "
        "different plan by plan code (e.g. fiber-500). Fails if the order "
        "is not active. Performs no eligibility checking."
    ),
    "cancel_order": (
        "Cancel an active or pending-installation order (ORD-XXXX). "
        "Destructive. Performs no balance, appointment, or confirmation "
        "checks."
    ),
    "request_confirmation": (
        "Ask the customer to confirm an action described in action_summary. "
        "Returns the customer's response. Must be called before any "
        "destructive change."
    ),
    # Pseudo-tool handled by the agent loop, not the API layer.
    "finish": (
        "End the episode and reply to the customer. Use when the task is "
        "complete, when only information was requested, or when policy or "
        "missing data prevents completing it. The summary is the message "
        "the customer reads."
    ),
}
