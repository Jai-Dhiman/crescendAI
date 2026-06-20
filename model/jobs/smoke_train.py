# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "trackio>=0.1.0",
#     "huggingface_hub>=0.25.0",
# ]
# ///
"""Minimal HF Jobs smoke training run (issue #74).

Self-contained -- no repo clone, no dataset -- so it validates the *cloud pipeline*
(`hf jobs uv run` -> tracked run -> persisted checkpoint + Trackio metrics) for a
few cents on cpu-basic. It deliberately mirrors the real training contract:

  - logs per-step metrics to Trackio, syncing to a persistent HF Space
    (TRACKIO_SPACE_ID) so the dashboard survives the ephemeral job, and
  - uploads a checkpoint to a HF Hub repo (CKPT_REPO) so weights survive too.

The real Aria fine-tune (jobs/train_aria.py, --flavor a100-large) follows the
same two contracts; this is the cheap proof that the launcher path works.

Env (HF_TOKEN is injected by `hf jobs uv run --secrets HF_TOKEN`):
  TRACKIO_SPACE_ID  HF Space id for the Trackio dashboard (default <user>/crescendai-trackio)
  CKPT_REPO         HF model repo for the checkpoint     (default <user>/crescendai-smoke-ckpt)
  SMOKE_STEPS       optimisation steps                   (default 50)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi


def main() -> None:
    api = HfApi()
    user = api.whoami()["name"]
    space_id = os.environ.get("TRACKIO_SPACE_ID", f"{user}/crescendai-trackio")
    ckpt_repo = os.environ.get("CKPT_REPO", f"{user}/crescendai-smoke-ckpt")
    steps = int(os.environ.get("SMOKE_STEPS", "50"))

    import trackio

    trackio.init(
        project="crescendai-model-v2",
        name="hf-jobs-smoke",
        space_id=space_id,
        config={"job": "smoke", "steps": steps, "flavor": os.environ.get("HF_FLAVOR", "cpu-basic")},
    )
    print(f"[smoke] Trackio dashboard -> https://huggingface.co/spaces/{space_id}")

    # Tiny, deterministic regression task: learn y = 3x + 2.
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 256).unsqueeze(1)
    y = 3 * x + 2 + 0.05 * torch.randn_like(x)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    final_loss = None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        final_loss = loss.item()
        trackio.log(
            {"loss": final_loss, "learning_rate": opt.param_groups[0]["lr"]}, step=step
        )
        if step % 10 == 0 or step == steps - 1:
            print(f"[smoke] step {step} loss {final_loss:.5f}")

    trackio.finish()

    # Persist the checkpoint to the Hub so it outlives the ephemeral job.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "smoke_model.pt"
        torch.save(
            {"state_dict": model.state_dict(), "final_loss": final_loss, "steps": steps},
            ckpt,
        )
        api.create_repo(ckpt_repo, repo_type="model", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo="smoke_model.pt",
            repo_id=ckpt_repo,
            repo_type="model",
        )
    print(f"[smoke] checkpoint -> https://huggingface.co/{ckpt_repo} (final_loss={final_loss:.5f})")
    print("[smoke] DONE: Trackio metrics + checkpoint both persisted.")


if __name__ == "__main__":
    main()
