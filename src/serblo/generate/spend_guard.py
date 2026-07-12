"""Client-side spend hard-cap for PAID generation accounts (no Google budget API).

The accounts are paid (they charge the card past the $300 credit), so we enforce the
ceiling ourselves. The guarantee rests on one lever: cap maxOutputTokens per generation.
Then a batch's WORST-CASE cost is exactly computable before submit:
    worst = n_prompts * (avg_input_tok * price_in + max_output_tok * price_out) / 1e6
and we refuse any batch whose worst case would push committed spend over `budget`.

Async batches RESERVE their worst-case cost the instant they're submitted; when they
finish, the ledger swaps the reservation for the (always lower) actual cost, freeing
headroom for more. Invariant: sum of actual costs can never exceed `budget`, because at
no moment do we let committed worst-case exceed it. A margin (budget < credit) absorbs
price drift + tiny non-token costs (GCS I/O). Ledger: data/generated/spend-<account>.json
"""
import json
from pathlib import Path

# Vertex BATCH prices, USD per 1M tokens (already the 50%-off tier). Hardcoded so the cap
# is self-contained; a Google price change needs a bump here (margin covers short drift).
PRICES = {
    "gemini-2.5-flash-lite": {"in": 0.05, "out": 0.20},
    "gemini-2.5-flash": {"in": 0.15, "out": 1.25},
    "gemini-3.1-pro": {"in": 1.00, "out": 6.00},
}


def per_prompt_worst_cost(avg_input_tokens, max_output_tokens, model):
    p = PRICES[model]
    return (avg_input_tokens * p["in"] + max_output_tokens * p["out"]) / 1e6


def batch_worst_cost(n_prompts, avg_input_tokens, max_output_tokens, model):
    return n_prompts * per_prompt_worst_cost(avg_input_tokens, max_output_tokens, model)


def actual_cost(input_tokens, output_tokens, model):
    p = PRICES[model]
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1e6


class SpendGuard:
    """Per-account committed-spend ledger with a hard budget."""

    def __init__(self, path, budget):
        self.path = Path(path)
        self.data = {"budget": budget, "batches": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            self.data["budget"] = budget                 # a re-run may re-cap; never widen silently
        self._save()

    def committed(self):
        """Actual cost where known, else reserved worst-case."""
        return sum(b["actual"] if b.get("actual") is not None else b["worst"]
                   for b in self.data["batches"].values())

    def remaining(self):
        return round(self.data["budget"] - self.committed(), 6)

    def fits(self, worst_cost):
        return worst_cost <= self.remaining() + 1e-9

    def max_prompts(self, avg_input_tokens, max_output_tokens, model):
        """How many prompts the remaining budget can cover at worst case (for auto-trim).
        Never negative: over-budget returns 0 so the caller refuses cleanly."""
        per = per_prompt_worst_cost(avg_input_tokens, max_output_tokens, model)
        return max(0, int(self.remaining() / per)) if per > 0 else 0

    def reserve(self, job_name, worst_cost):
        self.data["batches"][job_name] = {"worst": worst_cost, "actual": None}
        self._save()

    def settle(self, job_name, actual):
        if job_name in self.data["batches"]:
            self.data["batches"][job_name]["actual"] = actual
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))
