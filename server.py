"""Cross-platform supervisor for OceanMind's backend and frontend."""
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import re
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import build_opener, ProxyHandler
from urllib.parse import urlsplit

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


def public_url(value):
    """Accept an origin, not a credential-bearing URL or an unsupported subpath."""
    if any(c.isspace() or c in '"\\' for c in value):
        raise argparse.ArgumentTypeError("Public URL must not contain whitespace, quotes or backslashes.")
    try:
        u = urlsplit(value)
        port = u.port
        valid = (u.scheme in ("http", "https") and u.hostname
                 and u.username is None and u.password is None
                 and u.path in ("", "/") and not u.query and not u.fragment
                 and (port is None or 1 <= port <= 65535))
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError("Use an http(s) origin, e.g. https://oceanmind.wavyocean.hkust.edu.hk/")
    return value.rstrip("/")


def nginx_settings(args):
    if not args.nginx:
        if args.tls_cert or args.tls_key:
            raise RuntimeError("--tls-cert/--tls-key require --nginx.")
        return None
    if not args.public_url or not args.tls_cert or not args.tls_key:
        raise RuntimeError("Nginx requires --public-url https://DOMAIN --tls-cert FILE --tls-key FILE.")
    url = urlsplit(args.public_url)
    if url.scheme != "https" or url.port not in (None, 443):
        raise RuntimeError("Managed Nginx requires an HTTPS URL on port 443.")
    if not re.fullmatch(r"[A-Za-z0-9.-]+", url.hostname or ""):
        raise RuntimeError("Use a DNS hostname for managed Nginx.")
    if args.web_host not in ("127.0.0.1", "0.0.0.0"):
        raise RuntimeError("Managed Nginx needs --web-host 127.0.0.1 (recommended) or 0.0.0.0.")
    if args.web_port in (80, 443) or args.api_port in (80, 443):
        raise RuntimeError("Ports 80/443 are reserved for Nginx, not frontend/backend.")
    paths = [Path(v).resolve() for v in (args.nginx, args.tls_cert, args.tls_key)]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Nginx executable/certificate/key file not found: {path}")
        if any(c in str(path) for c in ('"', '$', '\n', '\r')):
            raise RuntimeError("Unsupported characters in Nginx/certificate paths.")
    return paths


def prepare_nginx(args, paths):
    prefix = STATE / "nginx"
    (prefix / "logs").mkdir(parents=True, exist_ok=True)
    text = (ROOT / "nginx.conf.template").read_text(encoding="utf-8")
    for key, val in {"DOMAIN": urlsplit(args.public_url).hostname,
                     "CERT": paths[1].as_posix(), "KEY": paths[2].as_posix(),
                     "PORT": str(args.web_port)}.items():
        text = text.replace("@@" + key + "@@", val)
    config = prefix / "nginx.conf"
    config.write_text(text, encoding="utf-8")
    cmd = [str(paths[0]), "-p", prefix.as_posix() + "/", "-c", config.as_posix()]
    # A missing/bad certificate must fail before launching any service.
    result = subprocess.run(cmd + ["-t"], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise RuntimeError("Nginx configuration test failed:\n" + result.stderr)
    return cmd


def prepare_frontend(node, env, dev=False, check=False):
    cli = WEB / "node_modules/next/dist/bin/next"
    build = WEB / ".next/BUILD_ID"
    if cli.is_file() and (dev or build.is_file()):
        return str(cli)
    if check:
        raise RuntimeError("Frontend is not prepared yet. Run start-server.bat once; it installs and builds automatically.")
    # Never change dependencies underneath an already running instance.
    with InstanceLock():
        installed = False
        if not cli.is_file():
            npm = shutil.which("npm", path=env["PATH"])
            candidates = [Path(node).parent / "node_modules/npm/bin/npm-cli.js"]
            if npm:
                candidates.append(Path(npm).parent / "node_modules/npm/bin/npm-cli.js")
            npm_cli = next((p for p in candidates if p.is_file()), None)
            if npm_cli:
                command = [node, str(npm_cli)]
            elif npm and os.name != "nt":
                command = [npm]
            else:
                raise RuntimeError("npm not found alongside Node. Install Node.js including npm in the ocean environment.")
            print("OceanMind: installing frontend dependencies (first launch)...", flush=True)
            result = subprocess.run(command + ["ci", "--include=dev"], cwd=WEB, env=env)
            if result.returncode:
                raise RuntimeError("Frontend dependency installation failed; see npm output above.")
            installed = True
        if not dev and (installed or not build.is_file()):
            print("OceanMind: building frontend (first launch)...", flush=True)
            result = subprocess.run([node, str(cli), "build"], cwd=WEB, env=env)
            if result.returncode:
                raise RuntimeError("Frontend build failed; see output above.")
        if not cli.is_file() or (not dev and not build.is_file()):
            raise RuntimeError("Frontend setup did not produce the required files.")
    return str(cli)


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
    cli = prepare_frontend(str(Path(node).resolve()), env, args.dev,
                           check=args.action == "check")
    if not args.dev:
        env["NODE_ENV"] = "production"
    return env, str(Path(node).resolve()), str(cli)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["start", "stop", "check"], nargs="?", default="start")
    p.add_argument("--node", help="Full path to node.exe/node")
    p.add_argument("--nginx", help="Full path to nginx.exe/nginx; manage HTTPS proxy too")
    p.add_argument("--tls-cert", help="PEM full-chain certificate for the public domain")
    p.add_argument("--tls-key", help="PEM private key (must be readable without a password prompt)")
    p.add_argument("--public-url", type=public_url,
                   help="External website URL behind your reverse proxy; does not configure DNS/TLS")
    p.add_argument("--web-host", "--hostname", dest="web_host", default="127.0.0.1")
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
    nginx_paths = nginx_settings(args)
    if args.public_url:
        env["OCEANMIND_PUBLIC_URL"] = args.public_url
        print(f"Public address: {args.public_url}", flush=True)
        print(f"Proxy upstream: http://127.0.0.1:{args.web_port} (if proxy runs on this host).",
              flush=True)
    if args.action == "check":
        if nginx_paths:
            with InstanceLock():
                prepare_nginx(args, nginx_paths)
            print("Nginx configuration/certificate loading test passed.")
        print("Environment/build preflight passed (not a dataset or LLM API test).")
        if args.public_url:
            print("DNS, certificate hostname/expiry and external reachability are not tested.")
        return 0
    with InstanceLock():
        nginx_cmd = prepare_nginx(args, nginx_paths) if nginx_paths else None
        STOP.unlink(missing_ok=True)
        STOPPING.clear()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: STOPPING.set())
        log = logger("supervisor")
        if args.public_url:
            log.info("Configured public address: %s (external routing not verified)", args.public_url)
        print(f"OceanMind: http://{args.web_host}:{args.web_port}; Ctrl+C to stop.", flush=True)
        print(f"Logs: {ROOT / 'logs/server'}", flush=True)
        while not stopping():
            backend = frontend = proxy = None
            try:
                for port in (args.api_port, args.web_port) + ((80, 443) if nginx_cmd else ()):
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
                    if nginx_cmd:
                        proxy = spawn(nginx_cmd, ROOT, env, "nginx")
                        log.info("Nginx launched: %s -> frontend port %s",
                                 args.public_url, args.web_port)
                    while not stopping():
                        if any(p.poll() is not None for p in (backend, frontend, proxy) if p is not None):
                            raise RuntimeError("Service exited; restarting managed services in 10 seconds.")
                        pause(1)
            except (OSError, RuntimeError) as exc:
                log.error("%s", exc)
                print(str(exc), file=sys.stderr, flush=True)
            finally:
                stop_process(proxy)
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
