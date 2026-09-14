"""Run: python -m unittest test_server_launcher -v"""
import argparse
import importlib.util
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("launcher", Path(__file__).with_name("server.py"))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [
            patch.object(server, "ROOT", self.root),
            patch.object(server, "STATE", self.root / ".run"),
            patch.object(server, "STOP", self.root / ".run/server.stop"),
        ]
        for p in self.patches:
            p.start()
        server.STOPPING.clear()

    def tearDown(self):
        server.STOPPING.set()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_frontend_bootstrap_and_reuse(self):
        web = self.root / "web"
        web.mkdir()
        cli = web / "node_modules/next/dist/bin/next"
        build = web / ".next/BUILD_ID"
        npm = self.root / "node_modules/npm/bin/npm-cli.js"
        npm.parent.mkdir(parents=True)
        npm.touch()
        def execute(command, **kwargs):
            target = build if command[-1] == "build" else cli
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
            return subprocess.CompletedProcess(command, 0)
        with patch.object(server, "WEB", web), patch.object(server.subprocess, "run", side_effect=execute) as run:
            server.prepare_frontend(str(self.root / "node"), {"PATH": ""})
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0][-2:], ["ci", "--include=dev"])
            server.prepare_frontend(str(self.root / "node"), {"PATH": ""})
            self.assertEqual(run.call_count, 2)

    def test_frontend_check_does_not_install(self):
        with patch.object(server, "WEB", self.root / "web"), patch.object(server.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "not prepared"):
                server.prepare_frontend("node", {"PATH": ""}, check=True)
            run.assert_not_called()

    def test_frontend_failed_install_does_not_build(self):
        npm = self.root / "node_modules/npm/bin/npm-cli.js"
        npm.parent.mkdir(parents=True)
        npm.touch()
        with patch.object(server, "WEB", self.root / "web"), patch.object(server.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run:
            with self.assertRaisesRegex(RuntimeError, "installation failed"):
                server.prepare_frontend(str(self.root / "node"), {"PATH": ""})
            self.assertEqual(run.call_count, 1)

    def test_duplicate_lock(self):
        with server.InstanceLock():
            with self.assertRaisesRegex(RuntimeError, "already"):
                with server.InstanceLock():
                    pass
        with server.InstanceLock():
            pass

    def test_public_url(self):
        self.assertEqual(server.public_url("https://oceanmind.wavyocean.hkust.edu.hk/"),
                         "https://oceanmind.wavyocean.hkust.edu.hk")
        self.assertEqual(server.public_url("http://localhost:3000"), "http://localhost:3000")

    def test_nginx_requires_tls_files(self):
        args = argparse.Namespace(nginx="nginx", public_url="https://example.org",
                                  tls_cert=None, tls_key=None)
        with self.assertRaisesRegex(RuntimeError, "requires"):
            server.nginx_settings(args)

    def test_nginx_config_targets_frontend_and_streams(self):
        source = Path(server.__file__).with_name("nginx.conf.template")
        (self.root / "nginx.conf.template").write_text(source.read_text())
        paths = [self.root / name for name in ("nginx", "cert.pem", "key.pem")]
        for path in paths:
            path.touch()
        args = argparse.Namespace(nginx=str(paths[0]), public_url="https://example.org",
                                  tls_cert=str(paths[1]), tls_key=str(paths[2]),
                                  web_host="127.0.0.1", web_port=3100, api_port=8000)
        with patch.object(server.subprocess, "run") as run:
            run.return_value.returncode = 0
            cmd = server.prepare_nginx(args, server.nginx_settings(args))
        conf = (server.STATE / "nginx/nginx.conf").read_text()
        self.assertIn("proxy_pass http://127.0.0.1:3100;", conf)
        self.assertIn("proxy_buffering off;", conf)
        self.assertIn("listen 443 ssl;", conf)
        self.assertIn("server_name example.org;", conf)
        self.assertNotIn("@@", conf)
        self.assertEqual(run.call_args.args[0], cmd + ["-t"])

    def test_nginx_rejects_bad_cert_config(self):
        (self.root / "nginx.conf.template").write_text("ssl_certificate bad;")
        args = argparse.Namespace(public_url="https://example.org", web_port=3000)
        paths = [self.root / name for name in ("nginx", "cert.pem", "key.pem")]
        with patch.object(server.subprocess, "run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "invalid certificate"
            with self.assertRaisesRegex(RuntimeError, "invalid certificate"):
                server.prepare_nginx(args, paths)

    def test_invalid_public_url(self):
        for value in ["example.org", "https://user:secret@example.org", "ftp://example.org",
                      "https://example.org/app", "https://example.org/?token=x",
                      "https://example.org/#a", "https://example.org:99999",
                      "https://example.org:0", "https://example.org/\n"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                server.public_url(value)

    def test_stop_without_dependencies(self):
        with patch.object(server, "configuration", side_effect=AssertionError("must not load")):
            self.assertEqual(server.main(["stop"]), 0)
        self.assertTrue(server.STOP.exists())

    def test_invalid_ports(self):
        for api, web in [(0, 3000), (8000, 8000), (8000, 65536)]:
            args = argparse.Namespace(api_port=api, web_port=web, startup_timeout=10)
            with self.assertRaises(RuntimeError):
                server.configuration(args)

    def test_missing_env(self):
        args = argparse.Namespace(api_port=8000, web_port=3000, startup_timeout=10)
        with self.assertRaisesRegex(RuntimeError, ".env"):
            server.configuration(args)

    def test_stop_interrupts_delay(self):
        server.STOPPING.set()
        start = time.monotonic()
        server.pause(10)
        self.assertLess(time.monotonic() - start, 1)

    def test_dead_backend_never_ready(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.assertFalse(server.wait_backend(proc, 1, 2))

    def test_child_cleanup(self):
        proc = server.spawn([sys.executable, "-c", "import time; time.sleep(60)"],
                            self.root, dict(server.os.environ), "test-child")
        server.stop_process(proc)
        self.assertIsNotNone(proc.poll())

    def test_failed_backend_is_cleaned_then_retried(self):
        children = []
        attempts = []
        def fake_spawn(*args):
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                    start_new_session=(server.os.name != "nt"))
            children.append(proc)
            if len(children) == 3:
                server.STOP.touch()
            return proc
        def ready(*args):
            attempts.append(1)
            return len(attempts) > 1
        with socket.socket() as a, socket.socket() as b:
            a.bind(("127.0.0.1", 0)); b.bind(("127.0.0.1", 0))
            api, web = a.getsockname()[1], b.getsockname()[1]
        with patch.object(server, "configuration", return_value=({}, "node", "next")), \
             patch.object(server, "spawn", side_effect=fake_spawn), \
             patch.object(server, "wait_backend", side_effect=ready), \
             patch.object(server, "pause"), patch.object(server.signal, "signal"):
            server.main(["--api-port", str(api), "--web-port", str(web)])
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(p.poll() is not None for p in children))

    def test_supervisor_stops_both_and_clears_request(self):
        children = []
        def fake_spawn(*args):
            proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                    start_new_session=(server.os.name != "nt"))
            children.append(proc)
            if len(children) == 2:
                server.STOP.touch()
            return proc
        with socket.socket() as a, socket.socket() as b:
            a.bind(("127.0.0.1", 0)); b.bind(("127.0.0.1", 0))
            api, web = a.getsockname()[1], b.getsockname()[1]
        with patch.object(server, "configuration", return_value=({}, "node", "next")), \
             patch.object(server, "spawn", side_effect=fake_spawn), \
             patch.object(server, "wait_backend", return_value=True), \
             patch.object(server.signal, "signal"):
            self.assertEqual(server.main(["--api-port", str(api), "--web-port", str(web)]), 0)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(p.poll() is not None for p in children))
        self.assertFalse(server.STOP.exists())


if __name__ == "__main__":
    unittest.main()
