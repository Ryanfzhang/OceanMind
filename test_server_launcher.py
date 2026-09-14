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

    def test_duplicate_lock(self):
        with server.InstanceLock():
            with self.assertRaisesRegex(RuntimeError, "already"):
                with server.InstanceLock():
                    pass
        with server.InstanceLock():
            pass

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
