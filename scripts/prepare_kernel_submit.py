from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_username(repo_root: Path) -> str:
    config = json.loads((repo_root / "kaggle.json").read_text(encoding="utf-8"))
    username = str(config.get("username", "")).strip()
    if not username:
        raise ValueError("kaggle.json does not contain a username.")
    return username


def append_submission_override(notebook_path: Path, source_file: str) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source_literal = json.dumps(source_file)
    code = (
        "from pathlib import Path\n"
        "import shutil\n\n"
        f"_variant = Path('/kaggle/working') / {source_literal}\n"
        "_target = Path('/kaggle/working/submission.csv')\n"
        "if not _variant.exists():\n"
        "    raise FileNotFoundError(_variant)\n"
        "shutil.copy2(_variant, _target)\n"
        "print(f'Overrode submission.csv with {_variant.name}')\n"
    )
    notebook.setdefault("cells", []).append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True),
        }
    )
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a Kaggle notebook submission package.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-output-to-submission",
        help="Append a notebook cell that copies this /kaggle/working output file to submission.csv.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_dir = (repo_root / args.source_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    code_file = args.code_file

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not (source_dir / code_file).exists():
        raise FileNotFoundError(source_dir / code_file)

    username = read_username(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_notebook = out_dir / code_file
    shutil.copy2(source_dir / code_file, out_notebook)
    if args.copy_output_to_submission:
        append_submission_override(out_notebook, args.copy_output_to_submission)

    source_meta_path = source_dir / "kernel-metadata.json"
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8")) if source_meta_path.exists() else {}
    metadata = {
        "id": f"{username}/{args.slug}",
        "title": args.title,
        "code_file": code_file,
        "language": source_meta.get("language", "python"),
        "kernel_type": source_meta.get("kernel_type", "notebook"),
        "is_private": True,
        "enable_gpu": bool(source_meta.get("enable_gpu", False)),
        "enable_tpu": bool(source_meta.get("enable_tpu", False)),
        "enable_internet": bool(source_meta.get("enable_internet", False)),
        "dataset_sources": source_meta.get("dataset_sources", []),
        "competition_sources": source_meta.get("competition_sources", ["rogii-wellbore-geology-prediction"]),
        "model_sources": source_meta.get("model_sources", []),
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
