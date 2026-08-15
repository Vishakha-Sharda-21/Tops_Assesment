"""
Section D - Step 1: BUILD WITH AI
This is the ORIGINAL code produced by the AI tool from the prompt below.
It contains a bug that surfaces when a tool returns a 'not found' / error result.
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
    # BUG: assumes the 'status' key always exists, crashes with KeyError
    # whenever the tool returns an error/not-found result instead.
    print(f"Observation: order status is {observation1['status']}\n")

    # Cycle 2
    print("Cycle 2")
    print("Thought: Now I need to check refund eligibility.")
    observation2 = check_refund_eligibility(order_id)
    print(f"Action: check_refund_eligibility('{order_id}')")
    print(f"Observation: eligible = {observation2['eligible']}, delay = {observation2['delay_minutes']} min\n")

    if observation2["eligible"]:
        print("Final response: You're eligible for a refund due to the delay.")
    else:
        print("Final response: Sorry, this order does not qualify for a refund.")


if __name__ == "__main__":
    # Works fine for a valid order:
    react_agent("My order was late, can I get a refund?", "FD4521")

    print("\n" + "=" * 70 + "\n")

    # Crashes for an order that doesn't exist / was mistyped:
    react_agent("My order was late, can I get a refund?", "FD9999")
