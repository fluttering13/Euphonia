"""Persistent, isolated inference processes for optional TTS backends."""
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {
    'faster': ('Qwen3-TTS 0.6B', '.venv-faster'),
    'qwen_streaming': ('Qwen3-TTS-streaming（標點預生成）', '.venv-faster'),
    'f5': ('F5-TTS v1（16 步 / GPU）', '.venv-f5'),
    'zipvoice': ('ZipVoice-Distill（4 步 / GPU）', '.venv-zipvoice'),
}
PROCESS_BACKENDS = dict(BACKENDS)


class BackendProcess:
    def __init__(self, backend):
        self.backend = backend
        self.process = None
        self.messages = queue.Queue()
        self.log = None
        self.last_metrics = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        if self.backend not in PROCESS_BACKENDS:
            raise ValueError(f'Unknown backend: {self.backend}')
        python = ROOT / PROCESS_BACKENDS[self.backend][1] / 'Scripts/python.exe'
        if not python.is_file():
            raise RuntimeError(
                f'{PROCESS_BACKENDS[self.backend][0]} 尚未安裝，請執行 setup_models.ps1 -Model {self.backend}。')
        log_path = ROOT / 'data' / 'logs' / f'{self.backend}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open('a', encoding='utf-8')
        self.messages = queue.Queue()
        try:
            self.process = subprocess.Popen(
                [str(python), '-u', str(ROOT / 'scripts/backend_worker.py'), self.backend],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding='utf-8', env={**os.environ, 'PYTHONUTF8': '1'},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
        except Exception:
            self.close()
            raise
        messages, stream = self.messages, self.process.stdout
        def read():
            try:
                for line in stream:
                    try:
                        messages.put(json.loads(line))
                    except ValueError:
                        pass
            finally:
                stream.close()
                messages.put({'error': f'模型程序已結束，請查看 {log_path}'})
        threading.Thread(target=read, daemon=True).start()

    def request(self, request, progress, cancelled=lambda: False):
        started = time.perf_counter()
        if cancelled():
            return None
        self.start()
        self.last_metrics = None
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        discarded = False
        while True:
            if cancelled():
                # Drain this reply before accepting another request. Keep model
                # weights and voice conditioning resident instead of killing it.
                discarded = True
            try:
                message = self.messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if 'progress' in message:
                if not discarded:
                    progress(message['progress'])
            elif 'error' in message:
                if self.process.poll() is not None:
                    self.close()
                if discarded:
                    return None
                raise RuntimeError(message['error'])
            elif 'result' in message:
                self.last_metrics = message.get('metrics')
                if self.last_metrics is not None:
                    self.last_metrics['request_seconds'] = time.perf_counter() - started
                if discarded or cancelled():
                    output = Path(message['result'])
                    if output.parent == ROOT / 'data/outputs':
                        output.unlink(missing_ok=True)
                    return None
                return message['result']

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                if os.name == 'nt':
                    # A venv launcher can spawn the real Python interpreter.
                    # Terminate the exact worker PID and all descendants so the
                    # CUDA process cannot survive after the main application.
                    try:
                        subprocess.run(
                            ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
                            check=False)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
            if process.stdin:
                process.stdin.close()
        if self.log:
            self.log.close()
            self.log = None
