"""Create or update the Render Docker web service and set runtime env vars.

Reads credentials from the gitignored .env. Does not print secret values.

Usage (from repo root):
    $env:RENDER_API_KEY = 'rnd_...'
    .venv/Scripts/python.exe infra/deploy_render.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.render.com/v1"
SERVICE_NAME = os.environ.get("RENDER_SERVICE_NAME", "finrag")
REPO = os.environ.get(
    "RENDER_REPO",
    "https://huggingface.co/spaces/fardeen16/finrag",
)

VARIABLES = {
    "AGENT_MODE": "live",
    "EMBEDDING_PROVIDER": "local",
    "WARMUP_ON_START": "0",
    "VLLM_MODEL": None,  # from .env
    "VLLM_BASE_URL": None,
}
SECRETS = ("VLLM_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _request(method: str, path: str, token: str, body: object | None = None) -> object:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Render API {exc.code} {method} {path}: {detail}") from exc


def main() -> None:
    token = os.environ.get("RENDER_API_KEY", "").strip()
    if not token:
        raise SystemExit("Set RENDER_API_KEY to a Render API key from Account Settings.")

    env_path = ROOT / ".env"
    if not env_path.exists():
        raise SystemExit(f"Missing {env_path}")
    env = _load_dotenv(env_path)
    missing = [key for key in ("VLLM_BASE_URL", "VLLM_MODEL", *SECRETS) if not env.get(key)]
    if missing:
        raise SystemExit(f"Missing values in .env: {', '.join(missing)}")

    owners = _request("GET", "/owners", token)
    owner_id = None
    if isinstance(owners, list) and owners:
        first = owners[0]
        owner_id = (first.get("owner") or first).get("id")
    if not owner_id:
        raise SystemExit(f"Could not find a Render workspace. Response: {owners!r}")
    print(f"Workspace: {owner_id}")

    services = _request("GET", f"/services?limit=50", token)
    existing = None
    if isinstance(services, list):
        for row in services:
            svc = row.get("service") or row
            if svc.get("name") == SERVICE_NAME:
                existing = svc
                break

    if existing:
        service_id = existing["id"]
        print(f"Using existing service {service_id}")
    else:
        created = _request(
            "POST",
            "/services",
            token,
            {
                "type": "web_service",
                "name": SERVICE_NAME,
                "ownerId": owner_id,
                "repo": REPO,
                "branch": "main",
                "plan": "free",
                "region": "oregon",
                "autoDeploy": "no",
                "serviceDetails": {
                    "runtime": "docker",
                    "healthCheckPath": "/api/health",
                    "dockerContext": ".",
                    "dockerfilePath": "./Dockerfile",
                },
            },
        )
        service = created.get("service") or created
        service_id = service["id"]
        print(f"Created service {service_id}")

    env_vars = [
        {"key": "AGENT_MODE", "value": "live"},
        {"key": "EMBEDDING_PROVIDER", "value": "local"},
        {"key": "WARMUP_ON_START", "value": "0"},
        {"key": "VLLM_BASE_URL", "value": env["VLLM_BASE_URL"]},
        {"key": "VLLM_MODEL", "value": env["VLLM_MODEL"]},
    ]
    for key in SECRETS:
        env_vars.append({"key": key, "value": env[key]})

    _request("PUT", f"/services/{service_id}/env-vars", token, env_vars)
    print("Set environment variables")

    details = _request("GET", f"/services/{service_id}", token)
    svc = details.get("service") or details
    url = (svc.get("serviceDetails") or {}).get("url") or f"https://{SERVICE_NAME}.onrender.com"
    print(f"App URL: {url}")
    print("Render is building the Docker image. First build is 15–25 minutes.")

    # Best-effort: trigger a deploy so env vars apply on a new service too.
    try:
        _request("POST", f"/services/{service_id}/deploys", token, {})
        print("Triggered deploy")
    except SystemExit as exc:
        print(exc, file=sys.stderr)

    for _ in range(6):
        time.sleep(2)


if __name__ == "__main__":
    main()
