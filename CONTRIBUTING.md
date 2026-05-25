# Contributing to Compass

Thank you for interest in contributing! This guide explains how to develop and contribute.

## Local Setup

```bash
git clone https://github.com/prathibha-github/compass.git
cd compass

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_judges.py -v

# Run with coverage
pytest tests/ --cov=compass --cov-report=html
```

## Code Style

We use `black` and `isort` for formatting.

```bash
# Format code
black compass/ tests/ examples/

# Sort imports
isort compass/ tests/ examples/
```

## Making Changes

### Adding a New Rubric

1. Add to `compass/rubrics/library.py`:
```python
CLARITY = Rubric(
    name="clarity",
    version="1.0",
    category="constitutional",
    text="Score 1.0 when the explanation is clear and well-structured...",
    hit_threshold=0.5,
)
```

2. Add to `RubricLibrary` class:
```python
class RubricLibrary:
    clarity = CLARITY
```

3. Write tests in `tests/test_rubrics.py`

### Adding a New Judge Type

1. Extend the `Judge` base class in `compass/judges/base.py`
2. Implement the `evaluate()` method
3. Add tests to `tests/test_judges.py`
4. Document in `docs/api.md`

### Adding Utility Functions

1. Add to appropriate module (e.g., `compass/reproducibility/versioning.py`)
2. Export in module `__init__.py`
3. Export in main `compass/__init__.py` if part of public API
4. Write tests
5. Add docstring with examples

## Testing Requirements

- All new features need tests
- Aim for 85%+ code coverage
- Test both happy paths and edge cases
- Tests should be deterministic (no randomness)

Example test:

```python
def test_new_feature(self):
    """Brief description of what this tests."""
    result = my_function(input_value)
    self.assertEqual(result, expected_value)
```

## Documentation

- Update docstrings when changing function signatures
- Keep examples in `examples/` up to date
- Update `docs/api.md` for new public APIs
- Add migration notes to `CHANGELOG.md` for breaking changes

### Benchmark Changes

If your change touches benchmark logic or benchmark outputs:

- read `docs/BENCHMARK_PLAYBOOK.md` before changing runner, registry, or report code
- keep CLI wrappers thin and move reusable behavior into `compass/benchmark/`
- update at least one benchmark test when benchmark logic changes
- validate changed benchmark evaluation reports before merging:
  `python scripts/validate_changed_benchmark_reports.py path/to/evaluations.jsonl`
- validate any benchmark report directly with:
  `python scripts/validate_benchmark_report.py path/to/evaluations.jsonl`

The pull-request CI will enforce two benchmark-specific rules:
- benchmark logic changes require a benchmark test change
- changed benchmark evaluation reports must pass report validation

## Commit Messages

Write clear commit messages that explain the "why" not just the "what":

```
Add support for custom judge timeouts

Allow judges to specify custom timeout values for API calls.
This is useful for evaluating large completions that may take longer.

- Add timeout parameter to JudgeConfig
- Default to 30 seconds for compatibility
- Add tests covering timeout behavior
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes and write tests
3. Run tests locally: `pytest tests/ -v`
4. Commit with clear messages
5. Push and create a pull request
6. Link any related issues
7. Wait for CI to pass
8. Request review

For benchmark-heavy PRs, run the benchmark-focused checks locally when relevant:

```bash
pytest tests/test_benchmark_*.py tests/test_constitutional_benchmark_core.py -v
python scripts/validate_benchmark_report.py tests/fixtures/benchmark_evaluations_valid.jsonl
```

## Project Structure

```
compass/
├── __init__.py             # Main exports
├── _version.py             # Version string
├── judges/                 # Judge implementations
├── rubrics/                # Rubric definitions
├── cache/                  # Caching system
├── comparison/             # Multi-model comparison
├── reproducibility/        # Versioning and reproducibility
├── clients/                # Client abstractions
└── utils/                  # Utilities (currently empty)

tests/                       # Comprehensive test suite
examples/                    # Working examples
docs/                        # Documentation
```

## Code Principles

1. **Immutability where possible** — Rubrics, configs, and metadata are frozen
2. **Determinism** — Same inputs always produce same outputs
3. **Reproducibility** — Results are tied to rubric version, model, and seed
4. **Clarity over brevity** — Use meaningful names and avoid cryptic patterns
5. **Test-first for complex features** — Write tests before implementation

## Common Issues

### Import Cycles

If you get a circular import error:
- Check that submodules don't import from parent package
- Use `TYPE_CHECKING` guards for type hints
- Use lazy imports in `__init__.py` if needed

### Cache Inconsistency

If cached results seem wrong:
- Check that rubric hash hasn't changed
- Verify cache directory has proper permissions
- Delete `.compass_cache/` and re-run (safe to do)

## Reporting Bugs

Include:
- Python version (`python --version`)
- Compass version (`python -c "import compass; print(compass.__version__)"`)
- Minimal reproducible example
- Full error traceback
- What you expected vs. what happened

## Questions?

Open an issue for:
- Questions about the codebase
- Suggestions for improvement
- Design discussions
- General compass usage questions

We're here to help!
