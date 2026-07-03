import re

with open("neurona.py", "r") as f:
    code = f.read()

# Fix the steps_per_epoch off-by-one bug to avoid OneCycleLR crash
code = code.replace(
"""    steps = len(loader) // grad_accum
    if steps == 0: steps = 1""",
"""    steps = (len(loader) + grad_accum - 1) // grad_accum
    if steps == 0: steps = 1"""
)

with open("neurona.py", "w") as f:
    f.write(code)
