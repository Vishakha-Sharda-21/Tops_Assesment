"""
Section D - Step 2: TEST & DEBUG (without AI)
This is my CORRECTED version. It fixes the crash that happens when a tool
returns {'error': ...} instead of the expected fields, by checking for an
error observation and feeding it back into the next reasoning step instead
of indexing into it blindly.
"""

orders_db = {
    "FD4521": {"status": "delivered", "delivery_time_minutes": 40, "expected_time_minutes": 30},
}


def lookup_order_status(order_id):
    order = orders_db.get(order_id)
    if order is None:
        return {"error": "Order not found"}
    return order


def check_refund_eligibility(order_id):
    order = orders_db.get(order_id)
    if order is None:
        return {"error": "Order not found"}
    delay = order["delivery_time_minutes"] - order["expected_time_minutes"]
    eligible = delay >= 10
    return {"eligible": eligible, "delay_minutes": delay}


def react_agent(user_query, order_id):
    print(f"User query: {user_query}\n")

    # Cycle 1
    print("Cycle 1")
    print("Thought: I need to check the order status first.")
    observation1 = lookup_order_status(order_id)
    print(f"Action: lookup_order_status('{order_id}')")
    print(f"Observation: {observation1}\n")

    # FIX: check for an error observation instead of assuming 'status' exists.
    if "error" in observation1:
        print("Thought: The lookup returned an error, so I cannot proceed to check "
              "refund eligibility for an order that doesn't exist.")
        print(f"Final response: I couldn't find order '{order_id}'. Could you double-check "
              f"the order ID and try again?")
        return

    # Cycle 2
    print("Cycle 2")
    print("Thought: Order exists, now I need to check refund eligibility.")
    observation2 = check_refund_eligibility(order_id)
    print(f"Action: check_refund_eligibility('{order_id}')")
    print(f"Observation: {observation2}\n")

    if "error" in observation2:
        print(f"Final response: I couldn't verify refund eligibility for order '{order_id}' "
              f"due to an internal lookup error. Escalating to a human agent.")
        return

    if observation2["eligible"]:
        print("Final response: You're eligible for a refund due to the delay.")
    else:
        print("Final response: Sorry, this order does not qualify for a refund.")


if __name__ == "__main__":
    # Works fine for a valid order:
    react_agent("My order was late, can I get a refund?", "FD4521")

    print("\n" + "=" * 70 + "\n")

    # No longer crashes for an order that doesn't exist / was mistyped:
    react_agent("My order was late, can I get a refund?", "FD9999")
