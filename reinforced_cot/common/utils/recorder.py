import signal
from tensorboardX import SummaryWriter


class Recorder:
    _active_writers = set()

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.__class__._active_writers.add(self.writer)
        self._register_signal_handler()

    def record(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer in self._active_writers:
            self.writer.flush()
            self.writer.close()
            self._active_writers.remove(self.writer)

    @classmethod
    def _handle_signal(cls, signum, frame) -> None:
        for writer in list(cls._active_writers):
            writer.flush()
            writer.close()
        cls._active_writers.clear()

    @classmethod
    def _register_signal_handler(cls) -> None:
        if not hasattr(cls, "_handler_registered"):
            signal.signal(signal.SIGINT, cls._handle_signal)
            signal.signal(signal.SIGTERM, cls._handle_signal)
            cls._handler_registered = True

    def __del__(self):
        self.close()
