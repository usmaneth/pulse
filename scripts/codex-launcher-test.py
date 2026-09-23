#!/usr/bin/env python3
"""Test local launcher setup and child cleanup without a GPU."""
import json
import os
import runpy
from unittest.mock import patch, MagicMock
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parent.parent

def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]

class LauncherTest(unittest.TestCase):
    def managed_command(self, arguments):
        captured = []
        class Captured(Exception):
            pass
        def capture(command, **kwargs):
            captured.append(command)
            raise Captured()
        script = str(ROOT / 'scripts/codex-bonsai-server')
        with patch('sys.argv', [script, *arguments]), patch.object(Path, 'is_file', return_value=True), \
                patch('socket.socket', return_value=MagicMock()), patch('subprocess.Popen', side_effect=capture):
            with self.assertRaises(Captured):
                runpy.run_path(script, run_name='__main__')
        return captured[0]

    def test_corrected_default_and_legacy_control(self):
        default = self.managed_command([])
        self.assertTrue(default[default.index('-md') + 1].endswith('/bonsai2-v2-Q4_K_M-yarn32.gguf'))
        self.assertEqual(default[default.index('--spec-draft-n-max') + 1], '4')
        legacy = self.managed_command(['--legacy-v1', '--verified-spec-patch'])
        self.assertTrue(legacy[legacy.index('-md') + 1].endswith('/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf'))
        self.assertNotIn('-md', self.managed_command(['--no-draft']))

    def test_legacy_taper_and_draft_conflicts_fail_before_launch(self):
        script = str(ROOT / 'scripts/codex-bonsai-server')
        for arguments in [['--verified-spec-patch'], ['--draft', '/tmp/custom.gguf', '--verified-spec-patch'], ['--legacy-v1', '--no-draft']]:
            with patch('sys.argv', [script, *arguments]), patch('subprocess.Popen') as launch, patch('sys.stderr'):
                with self.assertRaises(SystemExit) as failure:
                    runpy.run_path(script, run_name='__main__')
                self.assertEqual(failure.exception.code, 2)
                launch.assert_not_called()

    def test_missing_corrected_default_has_actionable_error(self):
        script = str(ROOT / 'scripts/codex-bonsai-server')
        with patch('sys.argv', [script]), patch.object(Path, 'is_file', return_value=False), patch('sys.stderr') as error:
            with self.assertRaises(SystemExit):
                runpy.run_path(script, run_name='__main__')
            self.assertTrue(any('codex-bonsai-install-draft' in str(call) for call in error.write.call_args_list))
    def test_managed_lifecycle_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            backend = folder / 'backend'
            backend.write_text('''#!/usr/bin/env python3
import sys,json
from http.server import BaseHTTPRequestHandler,HTTPServer
port=int(sys.argv[sys.argv.index('--port')+1])
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  self.send_response(200); self.end_headers()
  self.wfile.write(json.dumps(dict(model_alias='test-bonsai',model_path='test.gguf',total_slots=1,build_info='test-build',default_generation_settings=dict(n_ctx=16384))).encode())
HTTPServer(('127.0.0.1',port),Handler).serve_forever()
''')
            backend.chmod(0o755)
            model = folder / 'test.gguf'
            model.touch()
            fake_codex = folder / 'codex'
            fake_codex.write_text('#!/usr/bin/env python3\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n')
            fake_codex.chmod(0o755)
            backend_port, port = free_port(), free_port()
            process = subprocess.Popen([str(ROOT / 'scripts/codex-bonsai-server'), '--backend-bin', str(backend),
                '--model', str(model), '--no-draft', '--context', '16384', '--backend-port', str(backend_port), '--port', str(port)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/ready', timeout=1) as response:
                            status = json.load(response)
                        break
                    except Exception:
                        if time.monotonic() > deadline:
                            self.fail('The test server did not start')
                        time.sleep(0.1)
                self.assertEqual(status['context_per_slot'], 16384)
                self.assertEqual(status['backend_build'], 'test-build')
                env = dict(os.environ, BONSAI_CODEX_URL=f'http://127.0.0.1:{port}/v1', CODEX_BIN=str(fake_codex), BONSAI_CODEX_MCP_STARTUP_GRACE_MS='0')
                env.pop('BONSAI_CODEX_CONTEXT', None)
                result = subprocess.run([str(ROOT / 'scripts/codex-bonsai'), 'exec', 'hello'], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('model_context_window=16384', json.loads(result.stdout))
                self.assertIn('mcp_optional_startup_grace_ms=0', json.loads(result.stdout))
                mismatch = subprocess.run([str(ROOT / 'scripts/codex-bonsai'), 'exec', 'hello'],
                    env=dict(env, BONSAI_CODEX_CONTEXT='65536'), capture_output=True, text=True)
                self.assertNotEqual(mismatch.returncode, 0)
                self.assertIn('differs from the backend', mismatch.stderr)
            finally:
                process.send_signal(signal.SIGTERM)
                process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0)
            for check in [port, backend_port]:
                with socket.socket() as sock:
                    self.assertNotEqual(sock.connect_ex(('127.0.0.1', check)), 0)

    def test_invalid_mcp_grace(self):
        for value in ('-1', 'not-a-number'):
            result = subprocess.run([str(ROOT / 'scripts/codex-bonsai'), '--version'], env=dict(os.environ, BONSAI_CODEX_MCP_STARTUP_GRACE_MS=value), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must be a nonnegative integer', result.stderr)

    def test_refuses_remote_provider(self):
        result = subprocess.run([str(ROOT / 'scripts/codex-bonsai'), '--version'],
            env=dict(os.environ, BONSAI_CODEX_URL='https://example.com/v1'), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('loopback', result.stderr)

if __name__ == '__main__':
    unittest.main()
