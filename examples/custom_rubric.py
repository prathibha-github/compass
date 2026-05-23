#!/usr/bin/env python3
"""Define and use your own custom rubric.

Shows how to create a rubric for something not in the standard library.
Custom rubrics are immutable and versioned, just like built-in ones.
"""
from compass import AnthropicClient, EvaluationCache, JudgeConfig, LLMJudge, Rubric

def main():
    client = AnthropicClient(model="claude-opus-4-7")
    cache = EvaluationCache()
    my_rubric = Rubric(
        name="code_quality",
        version="1.0",
        category="technical",
        created_at="2026-05-23",
        text=(
            "Score the code on clarity and maintainability.\n\n"
            "Score 1.0 if the code is:\n"
            "- Clear variable and function names\n"
            "- Proper error handling\n"
            "- Follows language conventions\n"
            "- Has no obvious inefficiencies\n\n"
            "Score 0.0 if the code:\n"
            "- Uses cryptic names (x, tmp, etc.)\n"
            "- Ignores errors silently\n"
            "- Has confusing control flow\n"
            "- Wastes resources unnecessarily\n\n"
            "Score 0.5 for code that's partially there."
        ),
        hit_threshold=0.7,
    )

    print(f"Custom rubric hash: {my_rubric.hash}")
    print(f"Rubric name: {my_rubric.name}")

    config = JudgeConfig(
        rubric=my_rubric,
        judge_model="claude-opus-4-7",
    )
    judge = LLMJudge(config, client, cache)

    bad_code = """
def f(x):
    y = []
    for i in range(len(x)):
        if x[i] > 0:
            y.append(i)
    return y
"""

    result = judge.evaluate(bad_code)
    print(f"\nCode quality score: {result.score:.2f}")
    print(f"Hit (good quality): {result.hit}")
    print(f"Feedback: {result.rationale}")


if __name__ == "__main__":
    main()
