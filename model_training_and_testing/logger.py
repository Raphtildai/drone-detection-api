import subprocess
import socket
import time
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config.TBOARD_DIR
        self.writer = None
        self.process = None

    def start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        print(f"📊 TensorBoard logs: {self.log_dir}")

        try:
            port = self._find_free_port()
            self.process = subprocess.Popen([
                "tensorboard",
                "--logdir", str(self.log_dir),
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload_multifile", "true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)
        except Exception as e:
            print(f"TensorBoard startup warning: {e}")
        return self.writer

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def log_metrics(self, metrics: dict, step: int):
        if self.writer:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, step)
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
        if self.process:
            self.process.terminate()
