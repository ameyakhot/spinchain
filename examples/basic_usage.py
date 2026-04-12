"""Test SpinChain optimization with synthetic reasoning chains.

Run: uv run python examples/basic_usage.py

No API key needed — uses hardcoded reasoning chains to validate
the fragment extraction → QUBO → SA → stability ranking pipeline.
"""

import json

from spinchain.server import optimize_reasoning

# Simulate 5 diverse reasoning chains for: "A farmer has 17 sheep. All but 9 die."
completions = [
    (
        "The farmer starts with 17 sheep. 'All but 9 die' means 9 survive. "
        "So the farmer has 9 sheep left. The answer is 9."
    ),
    (
        "Let me think step by step. Total sheep: 17. 'All but 9 die' is a classic "
        "trick question. It means every sheep except 9 dies. 17 - 9 = 8 sheep die. "
        "The farmer still has 9 sheep."
    ),
    (
        "17 sheep total. 'All but 9' means 9 are excluded from dying. "
        "So 17 - 9 = 8 die, and 9 remain alive. The answer is 9 sheep."
    ),
    (
        "This is a language trick. 'All but 9 die' — the 'but 9' means except 9. "
        "So 9 sheep survive. The farmer has 9 sheep left."
    ),
    (
        "Starting with 17 sheep. If all but 9 die, that means 17 - 9 = 8 sheep die. "
        "Wait, let me re-read. 'All but 9 die' means only 9 don't die. "
        "So the farmer has 9 sheep remaining."
    ),
]

print("Testing SpinChain optimize_reasoning with 5 synthetic chains...\n")

result_json = optimize_reasoning(
    completions=completions,
    num_reads=100,
    num_sweeps=1000,
)

result = json.loads(result_json)

print(f"Solver: {result['solver']}")
print(f"Completions: {result['num_completions']}")
print(f"Total fragments: {result['num_fragments']}")
print(f"Selected fragments: {len(result['selected_indices'])}")
print(f"Min energy: {result['min_energy']}")
print(f"Fallback: {result['fallback']}")
print(f"\nSelected reasoning fragments:")
for i, frag in enumerate(result["selected_fragments"]):
    print(f"  [{i + 1}] {frag}")
