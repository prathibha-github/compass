"""Check saved benchmark inputs before reusing results."""

import json
from pathlib import Path

from compass.evaluation.contracts import fingerprint, manifest_contracts


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
        planned = manifest_contracts(previous)
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
    expected = manifest_contracts(json.loads(manifest.read_text()))
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


def check_evaluation_coverage(path, rows):
    """Require complete modern scores; legacy reports remain explicitly unverified."""
    from compass.benchmark.schemas import evaluation_identity
    manifest = manifest_path(path)
    if not manifest.exists():
        if any("evaluation_contract" in row for row in rows):
            raise ResumeMismatch("Evaluation manifest is missing; restore it before reporting")
        return False
    expected = manifest_contracts(json.loads(manifest.read_text()))
    seen = set()
    for row in rows:
        key = evaluation_identity(row)
        if key in seen or key not in expected or row.get("evaluation_contract") != expected[key]:
            raise ResumeMismatch("Duplicate, unexpected, or stale score; re-score before reporting")
        seen.add(key)
    if seen != set(expected):
        raise ResumeMismatch("Incomplete evaluation coverage; resume scoring before reporting")
    return True
