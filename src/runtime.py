from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path


def run(
    cmd,
    *,
    cwd=None,
    env=None,
    check=True,
    capture_output=False,
):
    if isinstance(cmd, (list, tuple)):
        printable = shlex.join(str(part) for part in cmd)
    else:
        printable = cmd
    print(f"$ {printable}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def popen(cmd, *, cwd=None, env=None, log_path: Path):
    if isinstance(cmd, (list, tuple)):
        printable = shlex.join(str(part) for part in cmd)
    else:
        printable = cmd
    print(f"$ {printable}  > {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_http(url: str, timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        completed = run(
            ["curl", "-fsS", url],
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0:
            return True
        time.sleep(2)
    return False


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_uv(python_executable: str):
    run([python_executable, "-m", "pip", "install", "-q", "-U", "uv"])


def ensure_cloudflared(cloudflared_path: Path):
    if cloudflared_path.exists():
        return
    run(
        [
            "wget",
            "-q",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-O",
            str(cloudflared_path),
        ]
    )
    run(["chmod", "+x", str(cloudflared_path)])


def kill_old_processes(app_port: int, piper_backend_port: int):
    patterns = [
        "uvicorn app:app",
        "uvicorn openai_wrapper_app:app",
        "python -m piper.http_server",
        "cloudflared tunnel",
        "server_fastapi.py",
    ]
    for pattern in patterns:
        run(["pkill", "-f", pattern], check=False)
    run(["bash", "-lc", f"fuser -k {app_port}/tcp || true"], check=False)
    run(["bash", "-lc", f"fuser -k {piper_backend_port}/tcp || true"], check=False)


def ensure_git_clone(repo_url: str, target_dir: Path):
    if target_dir.exists():
        print(f"reuse: {target_dir}")
        return
    run(["git", "clone", repo_url, str(target_dir)])


def ensure_venv(engine_dir: Path) -> Path:
    venv_dir = engine_dir / ".venv"
    if not venv_dir.exists():
        run(["uv", "venv", str(venv_dir)])
    return venv_dir / "bin" / "python"


def uv_pip_install(python_bin: Path, packages, *, cwd=None):
    run(["uv", "pip", "install", "--python", str(python_bin), *packages], cwd=cwd)


def tail_log(log_path: Path, lines: int = 80):
    if not log_path.exists():
        print(f"log not found: {log_path}")
        return
    print(f"\n=== tail: {log_path.name} ===")
    completed = run(["tail", "-n", str(lines), str(log_path)], capture_output=True)
    print(completed.stdout)


def pretty_print_json_url(url: str, title: str):
    print(f"\n=== {title} ===")
    completed = run(["curl", "-fsS", url], capture_output=True, check=False)
    if completed.returncode != 0:
        print(f"failed to fetch: {url}")
        return
    try:
        print(json.dumps(json.loads(completed.stdout), ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(completed.stdout)
