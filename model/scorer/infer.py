import torch
from torch.utils.data import DataLoader

from scorer.dataset import SegmentDataset, collate_fn
from scorer.model import Evaluator


def predict_with_uncertainty(ckpt_path: str, manifest_path: str, passes: int = 8):
    ds = SegmentDataset(manifest_path)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Build the model skeleton and load weights
    model = Evaluator(num_dims=16)
    state = torch.load(ckpt_path, map_location="cpu")
    # If trained via Lightning, state may be nested; try common keys
    if "state_dict" in state:
        state = {k.replace("model.", ""): v for k, v in state["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)

    model.train()  # enable dropout for MC

    all_mean, all_var = [], []
    with torch.no_grad():
        for mel, y, m, ds, py, pm, pw, distill in loader:
            preds = []
            for _ in range(passes):
                p, _ = model(mel, ds)
                preds.append(p)
            P = torch.stack(preds)  # [K,B,D]
            all_mean.append(P.mean(0))
            all_var.append(P.var(0))
    return torch.cat(all_mean), torch.cat(all_var)