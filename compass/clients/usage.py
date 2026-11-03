"""Usage from every response, including empty responses that trigger retries."""

from copy import deepcopy


class CompletionFailure(RuntimeError):
    def __init__(self, message, ledger, settings):
        super().__init__(message)
        self.audit_record = {"request_settings": settings, **ledger.fields()}


class UsageLedger:
    def __init__(self, pricing):
        self.pricing = pricing
        self.attempts = []

    def record(self, input_tokens, output_tokens, raw_response, finish_reason, known=True):
        self.attempts.append({
            "tokens_used": {"input": input_tokens, "output": output_tokens},
            "cost_usd": (input_tokens * self.pricing.input_cost_per_million +
                         output_tokens * self.pricing.output_cost_per_million) / 1_000_000,
            "raw_response": raw_response, "finish_reason": finish_reason,
            "usage_complete": known,
        })

    def fields(self):
        return {
            "billing_tokens": {key: sum(a["tokens_used"][key] for a in self.attempts)
                               for key in ("input", "output")},
            "cost_usd": sum(a["cost_usd"] for a in self.attempts),
            "usage_complete": bool(self.attempts) and all(a["usage_complete"] for a in self.attempts),
            "usage_attempts": deepcopy(self.attempts),
        }
