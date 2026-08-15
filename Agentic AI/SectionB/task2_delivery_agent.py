"""
Task 2: Tool-Calling Delivery Agent
Module 20 - Agentic AI
Food Delivery Platform - Data Science M20-A1
"""

import random

# ---------------------------------------------------------------------------
# Tool functions (outside the class, as required)
# ---------------------------------------------------------------------------

def check_restaurant_status(name):
    statuses = {"saravana bhavan": "open", "pizza point": "closed", "spice hub": "open"}
    status = statuses.get(name.lower(), "unknown restaurant")
    return f"Restaurant '{name}' status: {status}"


def get_estimated_delivery_time(order_id):
    eta = random.choice([20, 25, 30, 40])
    return f"Order {order_id} estimated delivery time: {eta} minutes"


def apply_discount(order_id, reason):
    return f"Applied 20% discount to order {order_id} (reason: {reason})"


def file_complaint(order_id, issue):
    ticket_id = f"TCK-{order_id}-{random.randint(100, 999)}"
    return f"Complaint filed for order {order_id} (issue: {issue}). Ticket: {ticket_id}"


# ---------------------------------------------------------------------------
# MCP-style schemas for each tool
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "check_restaurant_status",
        "description": "Check whether a restaurant is currently open and accepting orders.",
        "parameters": {"name": "string - the restaurant name"},
    },
    {
        "name": "get_estimated_delivery_time",
        "description": "Get the estimated delivery time (in minutes) for an existing order.",
        "parameters": {"order_id": "string - the order identifier"},
    },
    {
        "name": "apply_discount",
        "description": "Apply a discount to a customer's order for a stated reason.",
        "parameters": {
            "order_id": "string - the order identifier",
            "reason": "string - why the discount is being applied",
        },
    },
    {
        "name": "file_complaint",
        "description": "File a formal complaint ticket against an order.",
        "parameters": {
            "order_id": "string - the order identifier",
            "issue": "string - description of the problem",
        },
    },
]


# ---------------------------------------------------------------------------
# Tool-calling agent
# ---------------------------------------------------------------------------

class FoodDeliveryAgent:
    """Simulates a simple tool-calling agent that routes customer queries to
    the correct internal tool via keyword matching and tracks session history."""

    def __init__(self):
        self.log = []  # list of (query, tool_called, result) tuples

    def think(self, query):
        q = query.lower()

        # very naive keyword-based intent routing
        if "status" in q or "open" in q or "closed" in q:
            restaurant = self._extract_restaurant(q)
            result = check_restaurant_status(restaurant)
            tool_called = "check_restaurant_status"

        elif "eta" in q or "delivery time" in q or "how long" in q:
            order_id = self._extract_order_id(q)
            result = get_estimated_delivery_time(order_id)
            tool_called = "get_estimated_delivery_time"

        elif "discount" in q or "refund" in q or "compensat" in q:
            order_id = self._extract_order_id(q)
            result = apply_discount(order_id, reason="customer request")
            tool_called = "apply_discount"

        elif "complain" in q or "issue" in q or "problem" in q or "cold" in q or "wrong" in q:
            order_id = self._extract_order_id(q)
            result = file_complaint(order_id, issue=query)
            tool_called = "file_complaint"

        else:
            result = "Sorry, I couldn't map this query to a known tool."
            tool_called = None

        self.log.append((query, tool_called, result))
        return result

    @staticmethod
    def _extract_order_id(text):
        for token in text.split():
            cleaned = token.strip(".,?!").upper()
            if cleaned.startswith("FD") or cleaned.isdigit():
                return cleaned
        return "UNKNOWN_ORDER"

    @staticmethod
    def _extract_restaurant(text):
        known = ["saravana bhavan", "pizza point", "spice hub"]
        for r in known:
            if r in text:
                return r.title()
        return "Unknown Restaurant"


if __name__ == "__main__":
    print("=== Registered Tool Schemas (MCP-style) ===")
    for schema in TOOL_SCHEMAS:
        print(schema)
    print()

    agent = FoodDeliveryAgent()

    test_queries = [
        "Is Saravana Bhavan open right now?",
        "What's the ETA for order FD4521?",
        "Can I get a discount on order FD4521, it was late?",
        "My order FD9981 arrived cold, I want to file a complaint",
        "How long will order FD1122 take to arrive?",
    ]

    print("=== Test Queries ===")
    for q in test_queries:
        response = agent.think(q)
        print(f"Query   : {q}")
        print(f"Response: {response}")
        print("-" * 80)

    print("\n=== Session Log ===")
    for entry in agent.log:
        print(entry)
