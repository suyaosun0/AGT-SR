"""Build vocabulary from explicit training tokens; never inspect dev/test text."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(x) for x in Path(args.train_manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows or any("tokens" not in r for r in rows):
        raise ValueError("Training manifest must contain explicit tokens; choose tokenizer before building vocabulary")
    tokens = sorted({t for r in rows for t in r["tokens"]})
    if not tokens or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("Tokens must be nonempty strings")
    result = {"token_to_id": {t: i+1 for i, t in enumerate(tokens)}, "blank_pad_id": 0}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
