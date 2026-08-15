"""
Task 3: Memory-Aware Strategy Selector
Module 19 + Module 20 combined
Food Delivery Platform - Data Science M20-A1
"""

# ---------------------------------------------------------------------------
# Strategy functions
# ---------------------------------------------------------------------------

def prompt_template_response(query):
    return f"[Prompt-Template] Thanks for reaching out! Regarding '{query}', here is a quick, templated answer."


def rag_lookup_response(query):
    return f"[RAG-Lookup] Retrieved current menu/item data relevant to '{query}' from the live restaurant catalog."


def fine_tuned_response(query):
    return f"[Fine-Tuned] Our specialised complaint-handling model has analysed '{query}' and drafted a resolution."


# ---------------------------------------------------------------------------
# Per-user memory store
# ---------------------------------------------------------------------------

memory = {}  # user_id -> {interaction_count, preferred_cuisine, last_complaint}


def _ensure_user(user_id):
    if user_id not in memory:
        memory[user_id] = {
            "interaction_count": 0,
            "preferred_cuisine": None,
            "last_complaint": None,
        }


def _update_memory(user_id, query):
    _ensure_user(user_id)
    mem = memory[user_id]
    mem["interaction_count"] += 1

    q = query.lower()
    cuisines = ["south indian", "north indian", "chinese", "italian", "mexican"]
    for c in cuisines:
        if c in q:
            mem["preferred_cuisine"] = c.title()

    if any(word in q for word in ["complain", "cold", "late", "wrong", "issue"]):
        mem["last_complaint"] = query


# ---------------------------------------------------------------------------
# Strategy decision logic
# ---------------------------------------------------------------------------

def decide_strategy(user_id, query):
    _update_memory(user_id, query)
    mem = memory[user_id]
    q = query.lower()

    is_complaint = any(word in q for word in ["complain", "cold", "late", "wrong", "issue", "refund"])
    is_menu_query = any(word in q for word in ["menu", "dish", "item", "recommend", "order"])

    if is_complaint and mem["interaction_count"] >= 3:
        strategy = "fine_tuned"
        response = fine_tuned_response(query)
    elif is_menu_query:
        strategy = "rag_lookup"
        response = rag_lookup_response(query)
    else:
        strategy = "prompt_template"
        response = prompt_template_response(query)

    return strategy, response


if __name__ == "__main__":
    user_id = "user_42"

    interactions = [
        "What South Indian dishes do you recommend?",
        "Can you show me the menu for Italian food?",
        "My order arrived cold, this is a complaint",
        "This is the third time my order was late, I want to complain",
        "What's your refund policy for a late delivery complaint?",
    ]

    for i, query in enumerate(interactions, start=1):
        strategy, response = decide_strategy(user_id, query)
        print(f"Interaction {i}: \"{query}\"")
        print(f"  Strategy chosen : {strategy}")
        print(f"  Response        : {response}")
        print(f"  Memory state    : {memory[user_id]}")
        print("-" * 90)
