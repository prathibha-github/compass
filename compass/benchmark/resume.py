"""Check saved benchmark inputs before reusing results."""

import json
from pathlib import Path

from compass.evaluation.contracts import fingerprint


class ResumeMismatch(ValueError):
    """Saved work cannot be reused for the requested inputs."""


def manifest_path(path):
    return Path(path).with_suffix(".manifest.json")


def check_resume(path, rows, expected, field):
    manifest = manifest_path(path)
    if rows and not manifest.exists():
        raise ResumeMismatch(f"{path} has no input manifest. Use a fresh output directory; saved generations can be scored there.")
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        planned = {tuple(item["identity"]): item["contract"] for item in previous["expected"]}
        if any(expected.get(key) != contract for key, contract in planned.items()):
            raise ResumeMismatch(f"Inputs changed for {path}. Use a fresh output directory; saved generations can be scored there.")
    for key, row in rows.items():
        if key not in expected or row.get(field) != expected[key]:
            raise ResumeMismatch(f"Stale or unexpected row in {path}. Use a fresh output directory.")


def write_manifest(path, expected):
    manifest = manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"expected": [{"identity": list(key), "contract": value} for key, value in expected.items()]}
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(manifest)


def check_generation_coverage(path, rows):
    manifest = manifest_path(path)
    if not manifest.exists():
        if any("generation_contract" in row for row in rows.values()):
            raise ResumeMismatch("Generation manifest is missing; restore it before scoring")
        # Archived generations can be rescored; the caller must verify the population.
        return
    expected = {tuple(item["identity"]): item["contract"] for item in json.loads(manifest.read_text())["expected"]}
    check_resume(path, rows, expected, "generation_contract")
    if set(rows) != set(expected):
        raise ResumeMismatch("Generation coverage is incomplete; resume generation before scoring")


def generation_contracts(spec, models, samples, budgets):
    if not models or len(models) != len(set(models)):
        raise ValueError("Models must be nonempty and unique")
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    expected = {}
    for rubric, prompts in spec.prompts_by_rubric.items():
        if len({p.id for p in prompts}) != len(prompts):
            raise ValueError("Prompt IDs must be unique within each rubric")
        for prompt in prompts:
            for model in models:
                contract = fingerprint({
                    "version": 1, "benchmark": spec.name, "benchmark_version": spec.version, "prompt": prompt.as_dict(),
                    "model": model, "max_tokens": budgets[model], "temperature": 0.7,
                })
                for sample in range(samples):
                    expected[(model, rubric, prompt.id, sample)] = contract
    return expected
