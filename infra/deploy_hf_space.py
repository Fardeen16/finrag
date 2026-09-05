"""Create or update the Hugging Face Docker Space and set runtime secrets.

Reads credentials from the gitignored .env. Does not print secret values.

Usage (from repo root, after `huggingface-cli login`):
    .venv/Scripts/python.exe infra/deploy_hf_space.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, whoami

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = os.environ.get("HF_SPACE_NAME", "finrag")

ALLOW_PATTERNS = [
    "Dockerfile",
    ".dockerignore",
    "backend/**",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/index.html",
    "frontend/vite.config.ts",
    "frontend/tsconfig.json",
    "frontend/tsconfig.app.json",
    "frontend/tsconfig.node.json",
    "frontend/src/**",
    "data/alphabet_financials_structured.csv",
]

VARIABLES = (
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "AGENT_MODE",
    "EMBEDDING_PROVIDER",
)
SECRETS = (
    "VLLM_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
)


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def main() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise SystemExit(f"Missing {env_path}. Copy .env.example and fill it in.")

    env = _load_dotenv(env_path)
    missing = [key for key in (*VARIABLES, *SECRETS) if not env.get(key)]
    if missing:
        raise SystemExit(f"Missing values in .env: {', '.join(missing)}")

    user = whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    api = HfApi()

    api.create_repo(
        repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    print(f"Space: https://huggingface.co/spaces/{repo_id}")

    for key in VARIABLES:
        api.add_space_variable(repo_id, key, env[key])
        print(f"Set variable {key}")
    for key in SECRETS:
        api.add_space_secret(repo_id, key, env[key])
        print(f"Set secret {key}")

    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=[
            "**/.env*",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/node_modules/**",
        ],
        commit_message="Deploy FinRAG Docker Space",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "infra" / "hf-space-readme.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Add Space card",
    )
    print(f"App URL: https://{user}-{SPACE_NAME}.hf.space")
    print("Build starts on Hugging Face. First image build takes 15–25 minutes.")


if __name__ == "__main__":
    main()
