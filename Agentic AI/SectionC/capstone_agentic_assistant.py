"""
Section C - Mini Capstone Project
Agentic Food Delivery Assistant
Data Science M20-A1

Combines: technique-selection logic (Module 19), tool calling, MCP-style
schemas, per-session memory, and menu-driven agent interaction (Module 20)
into one console program.

Run interactively:   python3 capstone_agentic_assistant.py
Run scripted demo:   python3 capstone_agentic_assistant.py --demo
"""

import sys
import random


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def place_order(restaurant, items):
    order_id = f"FD{random.randint(1000, 9999)}"
    return {"order_id": order_id, "restaurant": restaurant, "items": items, "status": "placed"}


def track_order(order_id):
    eta = random.choice([15, 20, 30, 40])
    return {"order_id": order_id, "eta_minutes": eta, "status": "on the way"}


def file_complaint(order_id, issue):
    ticket_id = f"TCK-{order_id}-{random.randint(100, 999)}"
    return {"ticket_id": ticket_id, "order_id": order_id, "issue": issue, "status": "open"}


def get_recommendations(preferred_cuisine):
    catalog = {
        "south indian": ["Masala Dosa", "Idli Sambar", "Uttapam"],
        "north indian": ["Butter Chicken", "Paneer Tikka", "Chole Bhature"],
        "italian": ["Margherita Pizza", "Pasta Alfredo"],
        "chinese": ["Veg Manchurian", "Hakka Noodles"],
        None: ["Chef's Special Thali", "Today's Combo Meal"],
    }
    picks = catalog.get(preferred_cuisine, catalog[None])
    return {"preferred_cuisine": preferred_cuisine, "recommendations": picks}


TOOL_SCHEMAS = [
    {
        "name": "place_order",
        "description": "Places a new food order for the customer.",
        "parameters": {"restaurant": "string", "items": "list[string]"},
    },
    {
        "name": "track_order",
        "description": "Returns live tracking / ETA information for an existing order.",
        "parameters": {"order_id": "string"},
    },
    {
        "name": "file_complaint",
        "description": "Files a complaint ticket against an order.",
        "parameters": {"order_id": "string", "issue": "string"},
    },
    {
        "name": "get_recommendations",
        "description": "Returns personalised dish recommendations based on the customer's preferred cuisine.",
        "parameters": {"preferred_cuisine": "string or null"},
    },
]


# ---------------------------------------------------------------------------
# Session memory + strategy logic
# ---------------------------------------------------------------------------

session_memory = {
    "orders_placed": 0,
    "complaints_filed": 0,
    "preferred_cuisine": None,
    "interaction_count": 0,
}

session_log = []  # list of dicts: {action, tool, args, strategy, result}


def decide_response_strategy(action_type, memory):
    """
    Decision logic:
      - 'complaint' action after 2+ prior complaints -> fine_tuned
        (enough recurring signal to justify a specialised complaint-handling response)
      - 'recommend' action -> rag (needs live/current menu grounding)
      - 'track' action -> rag (needs live/current order-status grounding)
      - everything else (place_order, first-time complaint) -> prompt
    """
    if action_type == "complaint":
        if memory["complaints_filed"] >= 2:
            return "fine_tuned"
        return "prompt"
    if action_type in ("recommend", "track"):
        return "rag"
    return "prompt"


def _update_memory(action_type, cuisine=None):
    session_memory["interaction_count"] += 1
    if action_type == "order":
        session_memory["orders_placed"] += 1
    if action_type == "complaint":
        session_memory["complaints_filed"] += 1
    if cuisine:
        session_memory["preferred_cuisine"] = cuisine


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def action_place_order(restaurant, items):
    strategy = decide_response_strategy("order", session_memory)
    result = place_order(restaurant, items)
    _update_memory("order")
    session_log.append({"action": "Place Order", "tool": "place_order",
                         "args": {"restaurant": restaurant, "items": items},
                         "strategy": strategy, "result": result})
    print(f"[{strategy}] Order placed: {result}")


def action_track_order(order_id):
    strategy = decide_response_strategy("track", session_memory)
    result = track_order(order_id)
    _update_memory("track")
    session_log.append({"action": "Track Order", "tool": "track_order",
                         "args": {"order_id": order_id},
                         "strategy": strategy, "result": result})
    print(f"[{strategy}] Tracking info: {result}")


def action_file_complaint(order_id, issue):
    strategy = decide_response_strategy("complaint", session_memory)
    result = file_complaint(order_id, issue)
    _update_memory("complaint")
    session_log.append({"action": "File Complaint", "tool": "file_complaint",
                         "args": {"order_id": order_id, "issue": issue},
                         "strategy": strategy, "result": result})
    print(f"[{strategy}] Complaint filed: {result}")


def action_get_recommendations(cuisine):
    strategy = decide_response_strategy("recommend", session_memory)
    result = get_recommendations(cuisine)
    _update_memory("recommend", cuisine=cuisine)
    session_log.append({"action": "Get Recommendations", "tool": "get_recommendations",
                         "args": {"preferred_cuisine": cuisine},
                         "strategy": strategy, "result": result})
    print(f"[{strategy}] Recommendations: {result}")


def print_session_report():
    print("\n" + "=" * 90)
    print("SESSION REPORT")
    print("-" * 90)
    for i, entry in enumerate(session_log, start=1):
        print(f"{i}. Action: {entry['action']:<20} Tool: {entry['tool']:<20} "
              f"Strategy: {entry['strategy']:<12} Result: {entry['result']}")
    print("-" * 90)
    print(f"Final session memory: {session_memory}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Menu loop
# ---------------------------------------------------------------------------

MENU_TEXT = """
========== Agentic Food Delivery Assistant ==========
1. Place Order
2. Track Order
3. File Complaint
4. Get Personalised Recommendations
5. Exit
=======================================================
"""


def run_interactive():
    print("=== Registered Tool Schemas (MCP-style) ===")
    for schema in TOOL_SCHEMAS:
        print(schema)

    while True:
        print(MENU_TEXT)
        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            restaurant = input("Restaurant name: ").strip()
            items = [i.strip() for i in input("Items (comma-separated): ").split(",")]
            action_place_order(restaurant, items)
        elif choice == "2":
            order_id = input("Order ID: ").strip()
            action_track_order(order_id)
        elif choice == "3":
            order_id = input("Order ID: ").strip()
            issue = input("Describe the issue: ").strip()
            action_file_complaint(order_id, issue)
        elif choice == "4":
            cuisine = input("Preferred cuisine (or leave blank): ").strip().lower() or None
            action_get_recommendations(cuisine)
        elif choice == "5":
            print_session_report()
            break
        else:
            print("Invalid option, please choose 1-5.")


def run_demo():
    """Scripted run so the program's behaviour can be verified non-interactively."""
    print("=== Registered Tool Schemas (MCP-style) ===")
    for schema in TOOL_SCHEMAS:
        print(schema)
    print()

    action_get_recommendations(None)
    action_place_order("Saravana Bhavan", ["Masala Dosa x2"])
    action_get_recommendations("south indian")
    action_track_order("FD4521")
    action_file_complaint("FD4521", "Order arrived 40 minutes late")
    action_file_complaint("FD4521", "Food was cold on arrival")
    action_file_complaint("FD4521", "Still waiting on a refund")  # 3rd complaint -> fine_tuned

    print_session_report()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()
