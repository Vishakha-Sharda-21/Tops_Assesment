"""
M16-A1 - Section C - Mini Capstone
QuickBite AI - Intelligent Food Delivery Assistant

Combines:
  - Module 13: a deployed ML model, called through the Flask /predict API
    from Task B2 (must be running at http://127.0.0.1:5000)
  - Module 15: prompt engineering / LLM integration with a few-shot system
    prompt, built into a Streamlit chat UI
  - Module 16: a LangChain agent with memory and a custom tool
    (get_delivery_estimate), plus a menu-search feature that injects
    personalised context into the prompt

Run:
    1) python task1_save_load_model.py      (creates delivery_model.joblib)
    2) python task2_flask_api.py            (starts the /predict API)
    3) streamlit run capstone_quickbite_app.py

Requires:
    pip install streamlit langchain langchain-openai requests
"""

import json
import requests
import streamlit as st

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

FLASK_API_URL = "http://127.0.0.1:5000/predict"
MENU_PATH = "menu.json"

st.set_page_config(page_title="QuickBite AI", page_icon="🛵")
st.title("🛵 QuickBite AI - Food Delivery Assistant")

# ---------------------------------------------------------
# Sidebar: address + dietary preference, persisted for the session
# ---------------------------------------------------------
with st.sidebar:
    st.header("Your Delivery Info")
    address = st.text_input("Delivery Address", key="address", placeholder="e.g. 42 MG Road, Surat")
    dietary_pref = st.selectbox(
        "Dietary Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian"], key="dietary_pref"
    )
    clear_clicked = st.button("Clear Chat")

# ---------------------------------------------------------
# Load menu data
# ---------------------------------------------------------
@st.cache_data
def load_menu():
    with open(MENU_PATH, "r") as f:
        return json.load(f)

menu_items = load_menu()


def search_menu(dietary_pref, keyword=""):
    """Filter the menu by dietary preference and an optional cuisine keyword,
    return the top 3 matches."""
    results = menu_items

    if dietary_pref == "Vegetarian" or dietary_pref == "Vegan":
        results = [item for item in results if item["veg"]]
    # "Non-vegetarian" and "No preference" don't filter by veg flag

    if keyword:
        keyword_lower = keyword.lower()
        results = [item for item in results if keyword_lower in item["cuisine"].lower()]

    return results[:3]


# ---------------------------------------------------------
# Custom LangChain tool: get_delivery_estimate
# ---------------------------------------------------------
def get_delivery_estimate(query: str) -> str:
    """
    Calls the local Flask /predict API. Expects the agent to pass a simple
    string like "distance_km=4.2, num_items=3, rain_flag=0" - we parse it
    defensively and fall back to reasonable defaults if parsing fails.
    """
    params = {"distance_km": 5.0, "num_items": 2, "rain_flag": 0}
    try:
        for part in query.split(","):
            if "=" in part:
                key, value = part.split("=")
                key = key.strip()
                value = value.strip()
                if key in params:
                    params[key] = float(value)
    except Exception:
        pass  # use defaults if the agent's input string is malformed

    try:
        resp = requests.post(FLASK_API_URL, json=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        minutes = data.get("predicted_delivery_time_min")
        return f"Estimated delivery time: about {minutes} minutes."
    except Exception as e:
        return f"Could not reach the delivery estimate service ({e}). Please try again shortly."


delivery_tool = Tool(
    name="get_delivery_estimate",
    func=get_delivery_estimate,
    description=(
        "Use this tool whenever the user asks how long their order will take, "
        "or asks for a delivery time estimate. Input should look like "
        "'distance_km=4.2, num_items=3, rain_flag=0' - estimate reasonable "
        "values for distance and item count if the user hasn't given them."
    ),
)

# ---------------------------------------------------------
# Few-shot system prompt
# ---------------------------------------------------------
FEW_SHOT_PREFIX = """You are QuickBite, a friendly and concise assistant for QuickBite Food Delivery.
Always use the customer's delivery address and dietary preference (given below) when relevant,
without asking the customer to repeat them. Keep responses short, warm, and food-app appropriate.

Example 1:
User: I'm starving, what do you recommend tonight?
QuickBite: Since you're vegetarian, how about the Paneer Tikka Masala (₹240) or the Veg Biryani (₹200)? Both are popular North Indian picks tonight!

Example 2:
User: How long will my order take?
QuickBite: Let me check that for you... your order should arrive in about 28 minutes, depending on current traffic and weather.

Customer delivery address: {address}
Customer dietary preference: {dietary_pref}
"""

# ---------------------------------------------------------
# Session state: memory + chat history
# ---------------------------------------------------------
if "memory" not in st.session_state or clear_clicked:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
    st.session_state.chat_history_display = []

if clear_clicked:
    st.rerun()

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",  # swap to https://api.openai.com/v1 for real OpenAI
    api_key="ollama",
    model="llama3",
    temperature=0.6,
)

agent = initialize_agent(
    tools=[delivery_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state.memory,
    verbose=False,
    handle_parsing_errors=True,
)

# ---------------------------------------------------------
# Chat UI
# ---------------------------------------------------------
for turn in st.session_state.chat_history_display:
    role, text = turn["role"], turn["text"]
    if role == "user":
        st.markdown(
            f"<div style='text-align:right; background:#DCF8C6; padding:8px 12px; "
            f"border-radius:10px; margin:4px 0; display:inline-block; float:right; clear:both;'>{text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:left; background:#F1F0F0; padding:8px 12px; "
            f"border-radius:10px; margin:4px 0; display:inline-block; float:left; clear:both;'>{text}</div>",
            unsafe_allow_html=True,
        )

user_message = st.chat_input("Ask QuickBite something...")

if user_message:
    st.session_state.chat_history_display.append({"role": "user", "text": user_message})

    # menu search: pull a cuisine keyword out of the message if it mentions one
    matched_items = search_menu(dietary_pref, keyword=user_message)
    menu_context = ""
    if matched_items:
        lines = [f"- {i['name']} ({i['cuisine']}, ₹{i['price']})" for i in matched_items]
        menu_context = "\nRelevant menu items you could suggest:\n" + "\n".join(lines)

    system_context = FEW_SHOT_PREFIX.format(
        address=address or "not provided yet",
        dietary_pref=dietary_pref,
    ) + menu_context

    full_input = f"{system_context}\n\nUser: {user_message}"

    with st.spinner("QuickBite is thinking..."):
        try:
            response = agent.run(input=full_input)
        except Exception as e:
            response = f"Sorry, something went wrong: {e}"

    st.session_state.chat_history_display.append({"role": "assistant", "text": response})
    st.rerun()
