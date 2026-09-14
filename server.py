"""Cross-platform supervisor for OceanMind's backend and frontend."""
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "apps/web"
STATE = ROOT / ".run"
STOP = STATE / "server.stop"
STOPPING = threading.Event()


class InstanceLock:
    """Released by the OS even after a crash."""
    def __enter__(self):
        STATE.mkdir(exist_ok=True)
        self.file = (STATE / "server.lock").open("a+b")
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("OceanMind already runs from this repository.")
        return self

    def __exit__(self, *args):
        self.file.close()


def logger(name):
    folder = ROOT / "logs/server"
    folder.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("oceanmind." + name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = RotatingFileHandler(folder / (name + ".log"), maxBytes=10_000_000,
                                backupCount=3, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(h)
    return log


def spawn(cmd, cwd, env, name):
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", start_new_session=(os.name != "nt"),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    def drain():
        with proc.stdout:
            for line in proc.stdout:
                logger(name).info(line.rstrip())
    threading.Thread(target=drain, daemon=True).start()
    return proc


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def stopping():
    return STOPPING.is_set() or STOP.exists()


def pause(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not stopping():
        STOPPING.wait(min(.5, max(0, deadline-time.monotonic())))


def wait_backend(proc, port, timeout):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None and not stopping():
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        pause(1)
    return False


def configuration(args):
    if not (1 <= args.api_port <= 65535 and 1 <= args.web_port <= 65535):
        raise RuntimeError("Ports must be in 1..65535.")
    if args.api_port == args.web_port or args.startup_timeout <= 0:
        raise RuntimeError("Use distinct ports and a positive startup timeout.")
    if not (ROOT / ".env").is_file():
        raise RuntimeError("Create the root .env first; see .env.example.")
    try:
        from dotenv import dotenv_values
        import uvicorn  # noqa: F401
    except ImportError:
        raise RuntimeError("Activate ocean and install backend dependencies first.")
    env = os.environ.copy()
    for key, val in dotenv_values(ROOT / ".env").items():
        if val is not None and key not in env:
            env[key] = val
    prefix = Path(sys.prefix)
    dirs = [prefix, prefix / "Scripts", prefix / "Library/bin",
            prefix / "Library/usr/bin", prefix / "Library/mingw-w64/bin"]
    env["PATH"] = os.pathsep.join(map(str, dirs)) + os.pathsep + env.get("PATH", "")
    env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", NEXT_TELEMETRY_DISABLED="1",
               BACKEND_API_BASE_URL=f"http://127.0.0.1:{args.api_port}")
    node = args.node or shutil.which("node", path=env["PATH"])
    if not node or not Path(node).is_file():
        raise RuntimeError("Node not found. Pass --node with the absolute executable path.")
    cli = WEB / "node_modules/next/dist/bin/next"
    if not cli.is_file():
        raise RuntimeError("Run npm ci inside apps/web first.")
    if not args.dev and not (WEB / ".next/BUILD_ID").is_file():
        raise RuntimeError("Run npm run build inside apps/web first.")
    if not args.dev:
        env["NODE_ENV"] = "production"
    return env, str(Path(node).resolve()), str(cli)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["start", "stop", "check"], nargs="?", default="start")
    p.add_argument("--node", help="Full path to node.exe/node")
    p.add_argument("--web-host", default="127.0.0.1")
    p.add_argument("--web-port", type=int, default=3000)
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=int, default=180)
    p.add_argument("--dev", action="store_true")
    args = p.parse_args(argv)
    if args.action == "stop":
        STATE.mkdir(exist_ok=True)
        STOP.touch()
        print("Stop requested; both services will be stopped.")
        return 0
    env, node, cli = configuration(args)
    if args.action == "check":
        print("Environment/build preflight passed (not a dataset or LLM API test).")
        return 0
    with InstanceLock():
        STOP.unlink(missing_ok=True)
        STOPPING.clear()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: STOPPING.set())
        log = logger("supervisor")
        print(f"OceanMind: http://{args.web_host}:{args.web_port}; Ctrl+C to stop.", flush=True)
        print(f"Logs: {ROOT / 'logs/server'}", flush=True)
        while not stopping():
            backend = frontend = None
            try:
                for port in (args.api_port, args.web_port):
                    with socket.socket() as s:
                        s.bind(("0.0.0.0", port))
                backend = spawn([sys.executable, "-m", "uvicorn", "apps.api.main:app",
                    "--host", "127.0.0.1", "--port", str(args.api_port), "--workers", "1"],
                    ROOT, env, "backend")
                if not wait_backend(backend, args.api_port, args.startup_timeout):
                    if not stopping():
                        raise RuntimeError("Backend failed/timed out; inspect backend.log.")
                else:
                    frontend = spawn([node, cli, "dev" if args.dev else "start",
                        "--hostname", args.web_host, "--port", str(args.web_port)],
                        WEB, env, "frontend")
                    log.info("Backend ready; frontend launched.")
                    while not stopping():
                        if backend.poll() is not None or frontend.poll() is not None:
                            raise RuntimeError("Service exited; restarting both in 10 seconds.")
                        pause(1)
            except (OSError, RuntimeError) as exc:
                log.error("%s", exc)
                print(str(exc), file=sys.stderr, flush=True)
            finally:
                stop_process(frontend)
                stop_process(backend)
            if not stopping():
                pause(10)
        STOP.unlink(missing_ok=True)
        log.info("Stopped.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f"OceanMind: {exc}", file=sys.stderr)
        sys.exit(1)
