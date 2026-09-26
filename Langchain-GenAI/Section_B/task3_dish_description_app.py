"""
M16-A1 - Section B - Task 3
LLM-Powered Dish Description Generator (Streamlit)

Sends a structured prompt to an LLM (works with Ollama running locally,
or any OpenAI-compatible endpoint - just change BASE_URL / API_KEY / MODEL
below) and generates an appetising promo description for a menu dish.

Run:
    streamlit run task3_dish_description_app.py

Requires:
    pip install streamlit openai
    (and either Ollama running locally with `ollama serve`, or a real
    OpenAI-compatible API key)
"""

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------
# LLM client config
# ---------------------------------------------------------
# For local Ollama (OpenAI-compatible endpoint):
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"          # Ollama ignores the key but the SDK requires one
MODEL = "llama3"            # swap for whatever model you have pulled

# For real OpenAI instead, comment the three lines above and use:
# BASE_URL = "https://api.openai.com/v1"
# API_KEY = "sk-..."
# MODEL = "gpt-4o-mini"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

st.set_page_config(page_title="QuickBite Dish Description Generator", page_icon="🍽️")
st.title("🍽️ QuickBite Dish Description Generator")
st.caption("Turn a dish name + cuisine into an appetising, customer-facing description.")

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
dish_name = st.text_input("Dish Name", placeholder="e.g. Paneer Tikka Masala")
cuisine_type = st.text_input("Cuisine Type", placeholder="e.g. North Indian")
length = st.selectbox("Description Length", ["Short", "Medium", "Long"])

LENGTH_GUIDE = {
    "Short": "1-2 sentences (around 25-35 words)",
    "Medium": "3-4 sentences (around 60-90 words)",
    "Long": "a full paragraph (around 120-160 words)",
}


def build_prompt(dish, cuisine, length_choice):
    return (
        f"You are a food copywriter for a food delivery app called QuickBite. "
        f"Write an appetising, customer-facing promotional description for the following dish.\n\n"
        f"Dish Name: {dish}\n"
        f"Cuisine Type: {cuisine}\n"
        f"Desired Length: {length_choice} ({LENGTH_GUIDE[length_choice]})\n\n"
        f"Instructions: Write in a warm, mouth-watering, customer-facing tone suitable for a "
        f"delivery app listing. Do not include ingredient lists or nutrition facts, just the "
        f"promotional description text itself, nothing else before or after it."
    )


def generate_description(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# Session state so "Regenerate" works without re-entering inputs
# ---------------------------------------------------------
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "last_description" not in st.session_state:
    st.session_state.last_description = None


def run_generation():
    if not dish_name or not cuisine_type:
        st.warning("Please fill in both Dish Name and Cuisine Type.")
        return

    prompt = build_prompt(dish_name, cuisine_type, length)
    print("---- PROMPT SENT TO LLM ----")
    print(prompt)
    print("-----------------------------")

    st.session_state.last_prompt = prompt

    with st.spinner("Generating description..."):
        try:
            description = generate_description(prompt)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return

    st.session_state.last_description = description


col1, col2 = st.columns([1, 1])
with col1:
    generate_clicked = st.button("Generate Description", type="primary")
with col2:
    regenerate_clicked = st.button(
        "Regenerate", disabled=st.session_state.last_prompt is None
    )

if generate_clicked:
    run_generation()

if regenerate_clicked and st.session_state.last_prompt:
    with st.spinner("Generating description..."):
        try:
            st.session_state.last_description = generate_description(
                st.session_state.last_prompt
            )
        except Exception as e:
            st.error(f"LLM call failed: {e}")

# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
if st.session_state.last_description:
    with st.container(border=True):
        st.subheader(f"{dish_name}")
        st.write(st.session_state.last_description)

    char_count = len(st.session_state.last_description)
    st.caption(f"Character count: {char_count}")
