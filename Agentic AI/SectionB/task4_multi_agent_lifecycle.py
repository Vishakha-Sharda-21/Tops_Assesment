"""
Task 4: Multi-Agent Order Lifecycle System
Module 19 + Module 20 combined
Food Delivery Platform - Data Science M20-A1
"""

import time
import random


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def check_restaurant_status(name):
    statuses = {"saravana bhavan": "open", "pizza point": "closed", "spice hub": "open"}
    return statuses.get(name.lower(), "unknown")


def get_estimated_delivery_time(order_id):
    return random.choice([20, 25, 30, 40])


def apply_discount(order_id, reason):
    return f"20% discount applied to {order_id} ({reason})"


def file_complaint(order_id, issue):
    return f"Complaint ticket TCK-{order_id}-{random.randint(100, 999)} opened for: {issue}"


TOOL_SCHEMAS = [
    {"name": "check_restaurant_status", "description": "Checks if a restaurant is open.",
     "parameters": {"name": "string"}},
    {"name": "get_estimated_delivery_time", "description": "Gets ETA in minutes for an order.",
     "parameters": {"order_id": "string"}},
    {"name": "apply_discount", "description": "Applies a discount to an order.",
     "parameters": {"order_id": "string", "reason": "string"}},
    {"name": "file_complaint", "description": "Files a complaint ticket for an order.",
     "parameters": {"order_id": "string", "issue": "string"}},
]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class OrderAgent:
    def process(self, data):
        """data: raw user input dict -> order summary dict"""
        order_id = f"FD{random.randint(1000, 9999)}"
        summary = {
            "order_id": order_id,
            "restaurant": data["restaurant"],
            "items": data["items"],
            "restaurant_status": check_restaurant_status(data["restaurant"]),
        }
        return summary


class DispatchAgent:
    PARTNERS = ["Ravi K.", "Anita S.", "Mohammed I.", "Priya N."]

    def process(self, data):
        """data: order summary dict -> dispatch details dict"""
        order_id = data["order_id"]
        partner = random.choice(self.PARTNERS)
        eta = get_estimated_delivery_time(order_id)
        return {"order_id": order_id, "delivery_partner": partner, "eta_minutes": eta}


class SupportAgent:
    def process(self, data):
        """data: dict with order_id + feedback text -> classification + technique + action result"""
        order_id = data["order_id"]
        feedback = data["feedback"].lower()

        if any(w in feedback for w in ["great", "good", "on time", "delicious", "thanks"]):
            classification = "positive"
            technique = "prompt"
            action_result = "Thank-you message sent to customer."
        elif any(w in feedback for w in ["cold", "late", "wrong item", "missing"]):
            classification = "complaint"
            technique = "rag"
            action_result = file_complaint(order_id, data["feedback"])
        else:
            classification = "escalate"
            technique = "fine_tuned"
            action_result = apply_discount(order_id, reason="escalated dissatisfaction")

        return {
            "order_id": order_id,
            "classification": classification,
            "technique": technique,
            "action_result": action_result,
        }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    def __init__(self):
        self.order_agent = OrderAgent()
        self.dispatch_agent = DispatchAgent()
        self.support_agent = SupportAgent()

    def run_lifecycle(self, customer_input, feedback):
        start = time.time()

        order_summary = self.order_agent.process(customer_input)
        dispatch_details = self.dispatch_agent.process(order_summary)
        support_result = self.support_agent.process(
            {"order_id": order_summary["order_id"], "feedback": feedback}
        )

        elapsed = round(time.time() - start, 4)

        self._print_report(order_summary, dispatch_details, support_result, elapsed)
        return {
            "order_summary": order_summary,
            "dispatch_details": dispatch_details,
            "support_result": support_result,
            "processing_time_seconds": elapsed,
        }

    @staticmethod
    def _print_report(order_summary, dispatch_details, support_result, elapsed):
        print("=" * 90)
        print("ORDER LIFECYCLE REPORT")
        print("-" * 90)
        print(f"Order Summary     : {order_summary}")
        print(f"Dispatch Details  : {dispatch_details}")
        print(f"Feedback Class.   : {support_result['classification']}")
        print(f"Technique Selected: {support_result['technique']}")
        print(f"Support Action    : {support_result['action_result']}")
        print(f"Total Processing Time: {elapsed} s")
        print("=" * 90)


if __name__ == "__main__":
    print("=== Registered Tool Schemas (MCP-style) ===")
    for schema in TOOL_SCHEMAS:
        print(schema)
    print()

    coordinator = Coordinator()

    # Scenario 1: smooth delivery
    coordinator.run_lifecycle(
        customer_input={"restaurant": "Saravana Bhavan", "items": ["Masala Dosa x2"]},
        feedback="The food was delicious and arrived on time, great service!",
    )

    print()

    # Scenario 2: complaint scenario
    coordinator.run_lifecycle(
        customer_input={"restaurant": "Spice Hub", "items": ["Chicken Biryani x1"]},
        feedback="The delivery was very late and the food arrived cold.",
    )
