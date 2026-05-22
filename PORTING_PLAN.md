# Porting Phase 2 Infrastructure: helm_eval_lex_quirks → compass

## Overview
Extract successful patterns from helm_eval_lex_quirks Phase 2 work and integrate into compass core, examples, and documentation.

---

## 1. COMPASS CORE CHANGES

### 1.1 Add OllamaClient (compass/clients/ollama.py)
**Source:** helm_eval_lex_quirks/models/ollama_client.py

**What to extract:**
- OllamaClient class with complete() interface
- ModelResponse dataclass
- Error handling pattern (raise RuntimeError on failures)
- Token estimation logic

**Changes:**
- Make it a first-class client alongside OpenAIClient, AnthropicClient
- Add to compass/clients/__init__.py
- Update compass/clients/base.py to document the interface

### 1.2 Add Checkpoint/Resume System (compass/evaluation/checkpoint.py - NEW)
**Source:** phase2_benchmark.py checkpoint logic

**What to extract:**
- `load_checkpoint()` function with legacy backward compatibility
- Sample-level identity tracking pattern: `(model, suite, detector, prompt_id, condition, sample_idx)`
- Graceful handling of missing/invalid fields
- Logging for data loss visibility

**Key patterns:**
```python
# Legacy records without sample_idx default to 0
sample_idx = result.get("sample_idx", 0)
if not isinstance(sample_idx, int):
    sample_idx = int(sample_idx)  # Handle string serialization
if missing_sample_idx:
    legacy_records += 1  # Track for logging
```

**Changes:**
- New module: compass/evaluation/checkpoint.py
- Add CheckpointManager class
- Document backward compatibility strategy in docstrings

### 1.3 Add Judge Reliability Audit (compass/judges/reliability.py - NEW)
**Source:** helm_eval_lex_quirks safety evaluation work

**What to extract:**
- Judge disagreement measurement (inter-judge correlation)
- Confidence interval calculation (Wilson intervals)
- Low-n sample detection
- Judge drift indicators (benign_control failures)

**Changes:**
- New module: compass/judges/reliability.py
- Add `audit_judge_agreement()` function
- Add `detect_judge_drift()` function
- Document assumptions and limitations

### 1.4 Enhance Pairwise Ranking (compass/comparison/pairwise.py - NEW)
**Source:** helm_eval_lex_quirks/metrics/pairwise_preference.py

**What to extract:**
- Segmented ranking (by task_type, condition, etc.)
- Win/loss/tie scoring with ties = 0.5 credit
- Confidence intervals on win rates
- Low-sample flagging

**Changes:**
- New module: compass/comparison/pairwise.py
- Add PairwiseRanker class with segmentation support
- Add statistical significance testing (optional)

---

## 2. COMPASS EXAMPLES

### 2.1 Long-Running Evaluation with Checkpoint/Resume
**File:** compass/examples/resumable_evaluation.py

**Topics:**
- Create CheckpointManager for long-running batches
- Resume from interruption
- Handle legacy data
- Log data loss explicitly

### 2.2 Multi-Judge Agreement Analysis
**File:** compass/examples/judge_reliability_audit.py

**Topics:**
- Run same completion against multiple judges
- Measure disagreement
- Detect judge drift (benign_control tests)
- Identify low-confidence regions

### 2.3 Constitutional Compliance Evaluation
**File:** compass/examples/constitutional_compliance.py

**Topics:**
- Evaluate models on task_focus, truthfulness, etc.
- Use pairwise ranking to compare models
- Segment results by task type
- Generate safety-case report

### 2.4 Local Model Evaluation with Ollama
**File:** compass/examples/ollama_evaluation.py

**Topics:**
- Use OllamaClient for free local evaluation
- Run multiple models without API keys
- Compare local vs cloud models

---

## 3. COMPASS DOCUMENTATION

### 3.1 Update README.md
- Add section: "Long-Running Evaluations with Checkpoint/Resume"
- Add section: "Judge Reliability & Disagreement Audit"
- Add section: "Pairwise Model Comparison"
- Add section: "Local Model Evaluation with Ollama"

### 3.2 Add docs/CHECKPOINT_SYSTEM.md
**Topics:**
- How checkpoint/resume works
- Backward compatibility strategy
- Legacy data handling
- Sample-level granularity
- When to use vs. alternatives

### 3.3 Add docs/JUDGE_RELIABILITY.md
**Topics:**
- Inter-judge disagreement measurement
- Detecting judge drift
- Confidence intervals on scores
- Low-n sample handling
- When judges should be audited

### 3.4 Add docs/PAIRWISE_COMPARISON.md
**Topics:**
- Head-to-head model comparison
- Segmented ranking (by task type)
- Statistical rigor (CIs, significance)
- Interpreting win rates
- Limitations & caveats

### 3.5 Add docs/CONSTITUTIONAL_COMPLIANCE.md
**Topics:**
- What we measure (task focus, truthfulness, etc.)
- Rubric design principles
- Expected results across models
- How to interpret results
- Safety-case reporting

---

## 4. COMPASS TESTS

### 4.1 test_checkpoint_system.py
- Parse legacy checkpoints
- Handle missing/invalid fields
- Backward compatibility
- Sample-level skip logic

### 4.2 test_judge_reliability.py
- Inter-judge agreement calculation
- Judge drift detection (benign_control)
- Confidence interval accuracy
- Low-n flagging

### 4.3 test_pairwise_ranking.py
- Segmented ranking correctness
- Win/loss/tie scoring
- Confidence interval coverage
- Transitive property checks

---

## 5. COMPASS VERSION & RELEASE

**Current:** v0.1.0 (core judge framework)

**After porting:**
- **v0.2.0** (Phase 2: Advanced evaluation patterns)
  - Checkpoint/resume system
  - Judge reliability auditing
  - Pairwise comparison with segmentation
  - Ollama client
  - 5 new example scripts
  - 4 new documentation files

---

## PORTING ORDER

### Phase 1: Core Infrastructure (1-2 days)
1. Add OllamaClient to compass/clients/
2. Add CheckpointManager to compass/evaluation/
3. Add JudgeReliabilityAuditor to compass/judges/
4. Add PairwiseRanker to compass/comparison/
5. Update compass/__init__.py exports

### Phase 2: Examples (1 day)
6. Create 4 example scripts in compass/examples/
7. Update compass/examples/README.md

### Phase 3: Documentation (1 day)
8. Add 4 new docs/*.md files
9. Update main README.md
10. Update API documentation

### Phase 4: Testing (1 day)
11. Add test_checkpoint_system.py
12. Add test_judge_reliability.py
13. Add test_pairwise_ranking.py
14. Verify 120+ tests still pass

### Phase 5: Release (0.5 day)
15. Update CHANGELOG.md
16. Bump version to 0.2.0
17. Update setup.py
18. Tag v0.2.0

---

## FILES TO CREATE/MODIFY

### New files (compass):
- compass/clients/ollama.py
- compass/evaluation/__init__.py
- compass/evaluation/checkpoint.py
- compass/judges/reliability.py
- compass/comparison/pairwise.py
- compass/examples/resumable_evaluation.py
- compass/examples/judge_reliability_audit.py
- compass/examples/constitutional_compliance.py
- compass/examples/ollama_evaluation.py
- docs/CHECKPOINT_SYSTEM.md
- docs/JUDGE_RELIABILITY.md
- docs/PAIRWISE_COMPARISON.md
- docs/CONSTITUTIONAL_COMPLIANCE.md
- tests/test_checkpoint_system.py
- tests/test_judge_reliability.py
- tests/test_pairwise_ranking.py

### Modified files (compass):
- compass/__init__.py (add exports)
- compass/clients/__init__.py (add OllamaClient)
- compass/clients/base.py (document client interface)
- compass/comparison/__init__.py (add PairwiseRanker)
- compass/judges/__init__.py (add reliability auditor)
- README.md (add new sections)
- setup.py (update version)
- CHANGELOG.md (add 0.2.0 notes)

---

## SUCCESS CRITERIA

✅ All phase 2 patterns extracted to compass
✅ 4 new example scripts demonstrating real use cases
✅ 4 new documentation files covering advanced topics
✅ 120+ tests passing (existing + new)
✅ compass v0.2.0 released with backward compatibility
✅ helm_eval_lex_quirks no longer imports from compass (clean separation)

