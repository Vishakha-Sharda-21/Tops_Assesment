# Section D — Notes on what I changed and why

The AI's original code looked up `order_status[order_id]` directly with no check for
whether that key actually existed, so asking about any order ID not in the dictionary
(a typo, or a real-but-unknown order) threw an unhandled `KeyError` and crashed the
whole terminal loop — confirmed this by testing with `"#999"`, which isn't in the dict.
I added an `if order_id not in order_status:` check before the lookup that prints a
friendly "couldn't find that order" message and `continue`s the loop instead of crashing,
and moved the `queried_orders.add(order_id)` line to only run after we've confirmed the
order actually exists, so the "unique orders queried" summary at the end isn't inflated
by bad/invalid input.
