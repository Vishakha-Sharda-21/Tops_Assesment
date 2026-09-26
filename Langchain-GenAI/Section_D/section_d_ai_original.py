"""
M16-A1 - Section D - Step 1: BUILD WITH AI
This is the AI tool's original output, unmodified, before manual debugging.
"""

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Simulated order status database
order_status = {
    "#101": "Preparing",
    "#102": "Out for delivery",
    "#103": "Delivered",
}

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3",
)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

queried_orders = set()

print("QuickBite Order Tracker (type 'quit' to exit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print(f"Session ended. Unique orders queried: {len(queried_orders)}")
        break

    # naive extraction: assumes the order id is always the last "word"
    order_id = user_input.split()[-1]
    queried_orders.add(order_id)

    # BUG: no check for order_id not being in the dictionary -
    # this throws a KeyError and crashes the whole program the moment
    # someone asks about an order that doesn't exist, or if the naive
    # split()[-1] extraction grabs something that isn't a real order id
    status = order_status[order_id]

    response = conversation.predict(
        input=f"The user asked about order {order_id}. Its status is {status}. Reply naturally."
    )
    print(f"Assistant: {response}")
