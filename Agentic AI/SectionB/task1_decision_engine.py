"""
Task 1: AI Technique Decision Engine
Module 19 - Fine-Tuning & Model Customization
Food Delivery Platform - Data Science M20-A1
"""


def recommend_technique(update_frequency, data_in_external_docs, needs_custom_behaviour, latency_budget):
    """
    Recommends the most suitable AI technique (Prompt Engineering, RAG, or
    Fine-Tuning) for a food-delivery use case.

    Parameters
    ----------
    update_frequency : str        'high' | 'low'   - how often the underlying data changes
    data_in_external_docs : str   'yes' | 'no'      - is the needed knowledge available in a
                                                       retrievable document/database source?
    needs_custom_behaviour : str  'yes' | 'no'      - does the task need a deep change in
                                                       tone/format/domain reasoning style?
    latency_budget : str          'low' | 'high'    - 'low' means the response must be fast
                                                       (little tolerance for an extra retrieval hop)

    Returns
    -------
    dict with keys 'technique' and 'justification'
    """
    uf = update_frequency.strip().lower()
    doc = data_in_external_docs.strip().lower()
    custom = needs_custom_behaviour.strip().lower()
    lat = latency_budget.strip().lower()

    technique = None
    justification = ""

    if uf == "high":
        if doc == "yes":
            technique = "RAG"
            justification = (
                "Data changes frequently and already lives in a retrievable source, so RAG "
                "fetches the latest facts at query time instead of memorising a moving target. "
                "Fine-tuning would go stale almost immediately and require constant retraining."
            )
        else:
            technique = "Prompt Engineering"
            justification = (
                "The data changes daily but there is no external document store to retrieve from, "
                "so instructions/templates in the prompt are the cheapest way to steer behaviour. "
                "Fine-tuning is wasted effort here because there is no stable pattern to bake into weights."
            )
    else:  # low update frequency
        if custom == "yes":
            if doc == "yes":
                technique = "RAG"
                justification = (
                    "The behaviour is specialised but the supporting knowledge already exists in "
                    "documents, so retrieval plus well-crafted prompts can deliver the customisation "
                    "without the cost of training. This avoids fine-tuning when grounding data is available."
                )
            else:
                technique = "Fine-Tuning"
                justification = (
                    "The task needs a genuinely different behaviour/style, the data is stable, and there is "
                    "no external corpus to retrieve from, so the only way to reliably shift the model's "
                    "behaviour is to bake it into the weights via fine-tuning."
                )
        else:
            if doc == "yes":
                technique = "RAG"
                justification = (
                    "No deep behavioural change is needed and relevant facts already live in a document "
                    "source, so retrieval augmented generation supplies accurate grounding cheaply. "
                    "Fine-tuning would be overkill for a task that is really just a lookup problem."
                )
            else:
                if lat == "low":
                    technique = "Fine-Tuning"
                    justification = (
                        "The data is stable, there is no document source to retrieve from, and the latency "
                        "budget is tight, so an extra retrieval hop is undesirable. A fine-tuned model can "
                        "respond directly and quickly since the pattern is baked into its weights."
                    )
                else:
                    technique = "Prompt Engineering"
                    justification = (
                        "The task is simple, stable, needs no special behaviour, and has no external data "
                        "source, so a well-designed prompt template is sufficient. Investing in fine-tuning "
                        "or a retrieval pipeline for such a simple task is not cost-effective."
                    )

    return {"technique": technique, "justification": justification}


if __name__ == "__main__":
    scenarios = [
        {
            "name": "Real-time menu recommendations (menu changes daily, pulled from partner DB)",
            "args": dict(update_frequency="high", data_in_external_docs="yes",
                         needs_custom_behaviour="no", latency_budget="high"),
        },
        {
            "name": "Complaint categorisation into fixed categories (rules stable, no docs)",
            "args": dict(update_frequency="low", data_in_external_docs="no",
                         needs_custom_behaviour="yes", latency_budget="high"),
        },
        {
            "name": "Order confirmation message generation (simple, stable, fast reply needed)",
            "args": dict(update_frequency="low", data_in_external_docs="no",
                         needs_custom_behaviour="no", latency_budget="low"),
        },
        {
            "name": "Domain-specific FAQ answering (answers live in a stable knowledge base)",
            "args": dict(update_frequency="low", data_in_external_docs="yes",
                         needs_custom_behaviour="no", latency_budget="high"),
        },
        {
            "name": "Live delivery-partner availability chat (changes minute to minute, no doc store)",
            "args": dict(update_frequency="high", data_in_external_docs="no",
                         needs_custom_behaviour="yes", latency_budget="high"),
        },
    ]

    for s in scenarios:
        result = recommend_technique(**s["args"])
        print(f"Scenario: {s['name']}")
        print(f"  Inputs      : {s['args']}")
        print(f"  Technique   : {result['technique']}")
        print(f"  Justification: {result['justification']}")
        print("-" * 90)
