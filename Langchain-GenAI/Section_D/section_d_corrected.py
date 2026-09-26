"""
M16-A1 - Section D - Step 2: TEST & DEBUG (WITHOUT AI)

Corrected version of section_d_ai_original.py.
Changed lines are marked with "# FIX:" comments explaining what was wrong.
"""

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

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

    order_id = user_input.split()[-1]

    # FIX: check whether the order id actually exists in the dictionary
    # BEFORE looking it up, instead of indexing straight into it. The AI's
    # original code threw an unhandled KeyError and crashed the whole
    # program whenever the user asked about an order id that wasn't in
    # order_status (e.g. a typo, or an order that genuinely doesn't exist).
    # Now we handle the missing case gracefully and let the loop continue.
    if order_id not in order_status:
        print(f"Assistant: I couldn't find an order with ID '{order_id}'. "
              f"Could you double check the order number?")
        continue  # FIX: skip the rest of this iteration, don't crash

    # FIX: only add to queried_orders once we know it's a real, found order -
    # the AI's original version added the (possibly invalid) order_id to the
    # set before checking it existed, which would have inflated the "unique
    # orders queried" count with bad input if the crash hadn't happened first.
    queried_orders.add(order_id)
    status = order_status[order_id]

    response = conversation.predict(
        input=f"The user asked about order {order_id}. Its status is {status}. Reply naturally."
    )
    print(f"Assistant: {response}")
