# Section D — Prompt given to the AI tool (Claude)

> Write a Python program that implements a LangChain ConversationChain for a food
> delivery order-tracking assistant. The user should be able to ask for the status
> of any order by giving an order ID (e.g. "#101"). Simulate the order lookup using
> a local Python dictionary mapping order IDs to status strings like "Preparing",
> "Out for delivery", and "Delivered". Use ConversationBufferMemory so that if the
> user asks "What about order #102?" right after asking about "#101", the assistant
> understands the follow-up in context without asking which order they mean. Run it
> as a terminal input loop that exits cleanly when the user types "quit", and print
> a summary at the end showing how many unique order IDs were queried during the
> session.
