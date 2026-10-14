"""One persistent thread serializes every VNA operation and live refresh."""
import queue
import threading
import time
from PySide6.QtCore import QThread, Signal
from .engine import Engine


class InstrumentWorker(QThread):
    event = Signal(str, object)

    def __init__(self):
        super().__init__()
        self.commands = queue.Queue()
        self.shutdown = threading.Event()
        self.engine = Engine(self.event.emit)

    def submit(self, command, **kwargs):
        if command == 'acquire_point':
            self.engine.cancel.clear()
        self.commands.put((command, kwargs))

    def stop_acquisition(self):
        # No VISA call from the GUI thread. The current driver call must finish
        # (or time out) before the cancellation is observed.
        self.engine.cancel.set()

    def run(self):
        next_refresh = 0.0
        try:
            while not self.shutdown.is_set():
                try:
                    name, kwargs = self.commands.get(timeout=0.05)
                except queue.Empty:
                    name = None
                if name:
                    try:
                        getattr(self.engine, name)(**kwargs)
                    except Exception as exc:
                        self.engine.live = False
                        if name in ('configure', 'acquire_point'):
                            self.engine.mode = 'unconfigured' if self.engine.device else 'disconnected'
                        self.event.emit('error', f'{name}: {type(exc).__name__}: {exc}')
                    finally:
                        self.event.emit('done', dict(command=name, status=self.engine.status()))
                    next_refresh = 0
                elif self.engine.live and time.monotonic() >= next_refresh:
                    try:
                        self.engine.poll()
                    except Exception as exc:
                        self.engine.live = False
                        self.event.emit('error', f'Live refresh stopped: {type(exc).__name__}: {exc}')
                        self.event.emit('status', self.engine.status())
                    interval = self.engine.config['interval'] if self.engine.config else 0.5
                    next_refresh = time.monotonic() + interval
        finally:
            try:
                self.engine.disconnect()
            except Exception as exc:
                self.event.emit('error', f'Disconnect: {exc}')

    def request_shutdown(self):
        self.stop_acquisition()
        self.shutdown.set()
