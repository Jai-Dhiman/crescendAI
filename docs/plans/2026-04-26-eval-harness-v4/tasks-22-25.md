# Tasks 22-25 (Group C — integration)

---

## Task 22: `prompts.ts` integration — voice blocks after `<style_guidance>`
**Group:** C (depends on Task 15, Task 16)

**Behavior:** `buildSynthesisFraming(...)` invokes `selectClusters` + `formatTeacherVoiceBlocks` and inserts the result after `<style_guidance>` and before `<task>`.

**Files:**
- Modify: `apps/api/src/services/prompts.ts`
- Modify: `apps/api/src/services/prompts.test.ts`

- [ ] **Step 1: Write the failing test** — append to `prompts.test.ts`:

```typescript
describe("buildSynthesisFraming + teacher_voice", () => {
	const pieceMetadata = { title: "Prelude", composer: "Chopin", skill_level: 3 };

	it("includes <teacher_voice> block after <style_guidance>", () => {
		const out = buildSynthesisFraming(
			900_000, "continuous_play",
			[
				{ dimension: "dynamics", score: 0.8, deviation_from_mean: 0.25, direction: "above_average" },
				{ dimension: "timing", score: 0.3, deviation_from_mean: -0.18, direction: "below_average" },
			],
			[],
			pieceMetadata, "", "Chopin",
		);
		const styleIdx = out.indexOf("<style_guidance");
		const voiceIdx = out.indexOf("<teacher_voice");
		const taskIdx = out.indexOf("<task>");
		expect(styleIdx).toBeGreaterThan(-1);
		expect(voiceIdx).toBeGreaterThan(styleIdx);
		expect(taskIdx).toBeGreaterThan(voiceIdx);
	});

	it("includes <also_consider> block", () => {
		const out = buildSynthesisFraming(
			900_000, "continuous_play",
			[{ dimension: "dynamics", score: 0.8, deviation_from_mean: 0.25, direction: "above_average" }],
			[],
			pieceMetadata, "", "Chopin",
		);
		expect(out).toContain("<also_consider");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/api && bun run vitest run src/services/prompts.test.ts
```
Expected: FAIL — `<teacher_voice` not found.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/prompts.ts`, add an import at top:

```typescript
import {
	deriveSignals,
	formatTeacherVoiceBlocks,
	selectClusters,
} from "./teacher_style";
```

Inside `buildSynthesisFraming`, after the existing block that pushes `<style_guidance>` (after the `if (guidance.length > 0) { ... }` block) and before the `student_memory` block, add:

```typescript
	const signals = deriveSignals(
		topMoments,
		drillingRecords,
		sessionDurationMs,
		pieceMetadata,
		practicePattern,
	);
	const voiceBlocks = formatTeacherVoiceBlocks(selectClusters(signals));
	if (voiceBlocks.length > 0) {
		parts.push("");
		parts.push(voiceBlocks);
	}
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/api && bun run vitest run src/services/prompts.test.ts
```
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```
git add apps/api/src/services/prompts.ts apps/api/src/services/prompts.test.ts
git commit -m "feat(api): inject teacher_voice + also_consider blocks in synthesis prompt"
```

---

## Task 23: `run_eval.py` integration — Python cluster injection
**Group:** C (depends on Task 10)

**Behavior:** `build_synthesis_user_msg(...)` returns a string containing both `<teacher_voice>` and `<also_consider>` blocks.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Create: `apps/evals/teaching_knowledge/test_run_eval_blocks.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/test_run_eval_blocks.py
from teaching_knowledge.run_eval import build_synthesis_user_msg


def test_user_msg_contains_teacher_voice_and_also_consider():
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.8, "timing": 0.3, "pedaling": 0.5,
                   "articulation": 0.5, "phrasing": 0.5, "interpretation": 0.5},
        duration_seconds=900,
        meta={"piece_slug": "x", "title": "Prelude", "composer": "Chopin", "skill_bucket": 3},
    )
    assert "<teacher_voice" in msg
    assert "<also_consider" in msg


def test_user_msg_blocks_appear_between_style_and_task():
    msg = build_synthesis_user_msg(
        muq_means={"dynamics": 0.8, "timing": 0.3, "pedaling": 0.5,
                   "articulation": 0.5, "phrasing": 0.5, "interpretation": 0.5},
        duration_seconds=900,
        meta={"piece_slug": "x", "title": "Prelude", "composer": "Chopin", "skill_bucket": 3},
    )
    style_idx = msg.find("<style_guidance")
    voice_idx = msg.find("<teacher_voice")
    task_idx = msg.find("<task>")
    assert -1 < style_idx < voice_idx < task_idx
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest teaching_knowledge/test_run_eval_blocks.py -v
```
Expected: FAIL — `<teacher_voice` not in output.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teaching_knowledge/run_eval.py`, add an import near other `shared` imports:

```python
from shared.teacher_style import format_teacher_voice_blocks, select_clusters
```

In `build_synthesis_user_msg`, after `guidance = get_style_guidance(meta.get("composer", ""))`, derive signals and append voice blocks. Update the function so the trailing block reads:

```python
    devs = [float(m["deviation_from_mean"]) for m in top_moments]
    negs = [-d for d in devs if d < 0]
    poss = [d for d in devs if d > 0]
    signals = {
        "max_neg_dev": max(negs) if negs else 0.0,
        "max_pos_dev": max(poss) if poss else 0.0,
        "n_significant": sum(1 for d in devs if abs(d) >= 0.1),
        "drilling_present": False,
        "drilling_improved": False,
        "duration_min": duration_seconds / 60,
        "mode_count": 1,
        "has_piece": bool(meta.get("title")) and meta.get("title") != "Unknown",
    }
    voice_blocks = format_teacher_voice_blocks(select_clusters(signals))

    parts: list[str] = [
        "<session_data>",
        json.dumps(session_data, indent=2),
        "</session_data>",
    ]
    if guidance:
        parts.append("")
        parts.append(guidance)
    if voice_blocks:
        parts.append("")
        parts.append(voice_blocks)
    parts.append("")
    parts.append(
        "<task>Write <analysis>...</analysis> first as a reasoning scratchpad "
        "(this will be stripped). Then write your teacher response: 3-6 sentences, "
        "conversational, warm, specific. Do not mention scores or numbers. Focus on "
        "what matters most for this session.</task>"
    )
    return "\n".join(parts)
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest teaching_knowledge/test_run_eval_blocks.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/teaching_knowledge/test_run_eval_blocks.py
git commit -m "feat(eval): wire cluster-based voice blocks in run_eval"
```

---

## Task 24: `run_eval.py` integration — atomic-matrix gate
**Group:** C (depends on Task 23 + Task 20)

**Behavior:** When `mean(judge_dimensions[].score) < atomic_threshold`, the row contains an `atomic_matrix` field with 8 moves; when above, the field is absent.

**Files:**
- Modify: `apps/evals/teaching_knowledge/run_eval.py`
- Create: `apps/evals/teaching_knowledge/test_run_eval_atomic_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/teaching_knowledge/test_run_eval_atomic_gate.py
from teaching_knowledge.run_eval import _maybe_atomic_judge


class FakeAtomicJudge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
    def complete(self, *, user: str, system: str, max_tokens: int) -> str:
        self.calls += 1
        return self.response


HAPPY = """{
  "moves": [
    {"move_id": "voicing_diagnosis", "attempted": false, "criteria": null},
    {"move_id": "pedal_triage", "attempted": false, "criteria": null},
    {"move_id": "rubato_coaching", "attempted": false, "criteria": null},
    {"move_id": "phrasing_arc_analysis", "attempted": false, "criteria": null},
    {"move_id": "tempo_stability_triage", "attempted": false, "criteria": null},
    {"move_id": "dynamic_range_audit", "attempted": false, "criteria": null},
    {"move_id": "articulation_clarity_check", "attempted": false, "criteria": null},
    {"move_id": "exercise_proposal", "attempted": false, "criteria": null}
  ]
}"""


def test_atomic_gate_fires_below_threshold():
    judge_dims = [{"score": 1.0}, {"score": 1.5}, {"score": 2.0}]  # mean = 1.5
    client = FakeAtomicJudge(HAPPY)
    result = _maybe_atomic_judge(
        synthesis_text="x", context={}, judge_dimensions=judge_dims,
        threshold=2.0, client=client,
    )
    assert result is not None
    assert len(result["moves"]) == 8
    assert client.calls == 1


def test_atomic_gate_skips_above_threshold():
    judge_dims = [{"score": 2.5}, {"score": 2.5}, {"score": 2.5}]
    client = FakeAtomicJudge(HAPPY)
    result = _maybe_atomic_judge(
        synthesis_text="x", context={}, judge_dimensions=judge_dims,
        threshold=2.0, client=client,
    )
    assert result is None
    assert client.calls == 0
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/evals && uv run pytest teaching_knowledge/test_run_eval_atomic_gate.py -v
```
Expected: FAIL — `ImportError: cannot import name '_maybe_atomic_judge'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/evals/teaching_knowledge/run_eval.py`, add the helper near the other helpers:

```python
def _maybe_atomic_judge(
    *, synthesis_text: str, context: dict, judge_dimensions: list[dict],
    threshold: float, client,
) -> dict | None:
    """If mean judge score < threshold, run the atomic-matrix judge and return a serializable dict."""
    from shared.judge_atomic import judge_atomic_matrix

    scores = [
        float(d["score"]) for d in judge_dimensions
        if isinstance(d.get("score"), (int, float))
    ]
    if not scores:
        return None
    mean_score = sum(scores) / len(scores)
    if mean_score >= threshold:
        return None
    result = judge_atomic_matrix(synthesis_text=synthesis_text, context=context, client=client)
    return {
        "moves": [
            {"move_id": m.move_id, "attempted": m.attempted, "criteria": m.criteria}
            for m in result.moves
        ],
        "threshold": threshold,
    }
```

**Wiring step (same task, same commit):** in `run()`, after the row construction in the non-`dry_run` branch, before writing the row to the JSONL, add:

```python
                    atomic = None
                    if not dry_run and atomic_threshold is not None:
                        atomic_client = LLMClient(provider=judge_provider, model=judge_model)
                        atomic = _maybe_atomic_judge(
                            synthesis_text=synthesis_text,
                            context=judge_context,
                            judge_dimensions=result["judge_dimensions"],
                            threshold=atomic_threshold,
                            client=atomic_client,
                        )
                    if atomic is not None:
                        result["atomic_matrix"] = atomic
```

Add `atomic_threshold: float | None = 2.0` to `run()`'s signature, and to `main()`'s argparse:

```python
    parser.add_argument(
        "--atomic-threshold",
        type=float,
        default=2.0,
        help="Run atomic-matrix judge when mean judge score is below this. "
             "Set to 0.0 to never fire, 4.0 to always fire.",
    )
```

Pass through: `run(..., atomic_threshold=args.atomic_threshold)`.

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/evals && uv run pytest teaching_knowledge/test_run_eval_atomic_gate.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add apps/evals/teaching_knowledge/run_eval.py apps/evals/teaching_knowledge/test_run_eval_atomic_gate.py
git commit -m "feat(eval): atomic-matrix gate fires below 7-dim threshold"
```

---

## Task 25: `justfile` recipes — compile + sync check
**Group:** C (depends on Task 16)

**Behavior:** `just compile-playbook` writes `apps/api/src/lib/playbook.json`; `just check-playbook-sync` exits 0 when synced and nonzero when stale.

**Files:**
- Modify: `justfile`
- Create: `scripts/test_just_recipes.sh`

- [ ] **Step 1: Write the failing test**

```bash
#!/usr/bin/env bash
# scripts/test_just_recipes.sh
set -euo pipefail
cd "$(dirname "$0")/.."

rm -f apps/api/src/lib/playbook.json
just compile-playbook >/dev/null
test -f apps/api/src/lib/playbook.json || { echo "FAIL: compile did not produce playbook.json"; exit 1; }

just check-playbook-sync >/dev/null || { echo "FAIL: sync check failed after fresh compile"; exit 1; }

echo "{}" > apps/api/src/lib/playbook.json
if just check-playbook-sync >/dev/null 2>&1; then
  echo "FAIL: sync check should have failed on stale JSON"
  exit 1
fi

just compile-playbook >/dev/null
echo "PASS"
```

- [ ] **Step 2: Run test — verify it FAILS**

```
chmod +x scripts/test_just_recipes.sh && bash scripts/test_just_recipes.sh
```
Expected: FAIL — `error: Justfile does not contain recipe 'compile-playbook'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `justfile`:

```make
# Compile shared/teacher-style/playbook.yaml -> apps/api/src/lib/playbook.json
compile-playbook:
    uv run --with pyyaml python scripts/compile_playbook.py

# CI sync check: fail if compiled JSON is stale
check-playbook-sync:
    uv run --with pyyaml python scripts/compile_playbook.py --check
```

- [ ] **Step 4: Run test — verify it PASSES**

```
bash scripts/test_just_recipes.sh
```
Expected: `PASS`.

- [ ] **Step 5: Commit**

```
git add justfile scripts/test_just_recipes.sh
git commit -m "feat(build): add compile-playbook + check-playbook-sync just recipes"
```
