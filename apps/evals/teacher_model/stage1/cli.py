import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="teacher_model.stage1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_holdout = sub.add_parser("holdout", help="Generate held-out briefing manifest")
    p_holdout.add_argument("--frac", type=float, default=0.12)
    p_holdout.add_argument("--seed", type=int, default=42)
    p_holdout.add_argument("--cache-dir", type=str, required=False)
    p_holdout.add_argument("--out", type=str, required=False)

    p_distill = sub.add_parser("distill", help="Distill examples from Sonnet")
    p_distill.add_argument("--shape", choices=["synthesis", "chat"], required=True)
    p_distill.add_argument("--n", type=int, required=True)

    p_cov = sub.add_parser("coverage", help="Report argument-coverage matrix status")
    p_cov.add_argument("--include-negatives", action="store_true")

    p_render = sub.add_parser("render", help="Render examples to apply_chat_template output")
    p_render.add_argument("--out", type=str, required=True)

    p_harness = sub.add_parser("harness", help="Run eval harness against vLLM endpoint")
    p_harness.add_argument("--endpoint", type=str, required=True)
    p_harness.add_argument("--holdout", type=str, required=True)
    p_harness.add_argument("--tokenizer-pin", type=str, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(
        f"Subcommand '{args.cmd}' not yet implemented. "
        "Help text and arg validation are in place; integration to be filled in subsequent tasks."
    )
