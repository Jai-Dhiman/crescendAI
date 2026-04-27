# Task 16 (Group B — compile script)

---

## Task 16: `compile_playbook.py` — YAML to JSON precompile
**Group:** B (depends on Task 1)

**Behavior:** `python scripts/compile_playbook.py` reads `shared/teacher-style/playbook.yaml`, writes `apps/api/src/lib/playbook.json`. `--check` mode exits nonzero on drift.

**Files:**
- Create: `scripts/compile_playbook.py`
- Create: `scripts/test_compile_playbook.py`
- Create: `apps/api/src/lib/playbook.json` (committed via test/script run)

- [ ] **Step 1: Write the failing test**

```python
# scripts/test_compile_playbook.py
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compile_playbook.py"
OUT = REPO_ROOT / "apps" / "api" / "src" / "lib" / "playbook.json"


def test_compile_writes_valid_json():
    res = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert OUT.exists()
    data = json.loads(OUT.read_text())
    assert "teaching_playbook" in data
    assert len(data["teaching_playbook"]["clusters"]) == 5


def test_check_mode_exits_zero_when_synced():
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    res = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_check_mode_exits_nonzero_on_drift():
    OUT.write_text(json.dumps({"stale": "value"}))
    res = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert res.returncode != 0
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd /Users/jdhiman/Documents/crescendai && uv run --with pyyaml --with pytest pytest scripts/test_compile_playbook.py -v
```
Expected: FAIL — `FileNotFoundError: scripts/compile_playbook.py`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# scripts/compile_playbook.py
"""Compile shared/teacher-style/playbook.yaml -> apps/api/src/lib/playbook.json."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "shared" / "teacher-style" / "playbook.yaml"
DST = REPO_ROOT / "apps" / "api" / "src" / "lib" / "playbook.json"


def _serialize() -> str:
    data = yaml.safe_load(SRC.read_text())
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = _serialize()
    if args.check:
        if not DST.exists() or DST.read_text() != text:
            print(f"DRIFT: {DST} stale. Run: python scripts/compile_playbook.py", file=sys.stderr)
            return 1
        return 0
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(text)
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd /Users/jdhiman/Documents/crescendai && uv run --with pyyaml --with pytest pytest scripts/test_compile_playbook.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add scripts/compile_playbook.py scripts/test_compile_playbook.py apps/api/src/lib/playbook.json
git commit -m "feat(teacher-style): YAML to JSON compile script"
```
