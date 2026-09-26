"""
M16-A1 - Section B - Task 4
LangChain Food Order Assistant with Memory

Multi-turn ConversationChain + ConversationBufferMemory that remembers a
user's delivery address and dietary preference across a session.

Requires:
    pip install langchain langchain-community langchain-openai

Uses an OpenAI-compatible endpoint (works with local Ollama too - just point
base_url at it, e.g. http://localhost:11434/v1, and use any model name you
have pulled).
"""

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------
# 1. LLM + memory setup
# ---------------------------------------------------------
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",  # swap to https://api.openai.com/v1 for real OpenAI
    api_key="ollama",
    model="llama3",
    temperature=0.5,
)

SYSTEM_PROMPT = """You are QuickBite, a helpful assistant for QuickBite Food Delivery.
Once the user tells you their delivery address, remember it for the rest of the
conversation and use it when relevant. Once the user tells you a dietary preference
(e.g. vegetarian, vegan, no nuts), remember that too and use it to guide any
recommendations you make later in the conversation, without the user needing to
repeat themselves.

Conversation so far:
{history}
Human: {input}
QuickBite:"""

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template=SYSTEM_PROMPT,
)

# memory_key must match the {history} variable used in the prompt template above
memory = ConversationBufferMemory(memory_key="history")
assert memory.memory_key in prompt_template.input_variables, (
    "memory_key must match a variable in the prompt template"
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt_template,
    verbose=False,
)

# ---------------------------------------------------------
# 2. Simulated 5+ turn conversation
# ---------------------------------------------------------
scripted_turns = [
    "Hi, my delivery address is 42 MG Road, Surat.",
    "Just so you know, I'm vegetarian.",
    "Can you recommend something for dinner tonight?",
    "Will that dish be delivered to the address I gave you earlier?",
    "By the way, what's the QuickBite customer support number?",
]

turn_count = 0

for user_input in scripted_turns:
    print(f"User: {user_input}")

    if user_input.strip().lower() == "exit":
        break

    response = conversation.predict(input=user_input)
    print(f"QuickBite: {response}\n")
    turn_count += 1

# ---------------------------------------------------------
# 3. Simple interactive loop with an 'exit' command
#    (comment out the scripted block above if you want to type live instead)
# ---------------------------------------------------------
print("--- Live mode: type 'exit' to end the conversation ---")
while True:
    user_input = input("User: ")
    if user_input.strip().lower() == "exit":
        print(f"\nSession ended. Total turns: {turn_count}")
        print("\n--- Full memory buffer ---")
        print(memory.buffer)
        break

    response = conversation.predict(input=user_input)
    print(f"QuickBite: {response}\n")
    turn_count += 1
