# tensorboard_logger.py
import socket
import subprocess
import time


from torch.utils.tensorboard import SummaryWriter

# ==================== TENSORBOARD LOGGER ====================
class TensorBoardLogger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config.TBOARD_DIR
        self.writer = None
        self.process = None

    def start(self):
        """Start TensorBoard server"""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing logs
        for f in self.log_dir.glob("events.out.tfevents.*"):
            try:
                f.unlink()
            except:
                pass

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        print(f"📊 TensorBoard logs: {self.log_dir}")

        # Start TensorBoard process
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
            self._setup_public_url(port)

        except Exception as e:
            print(f"TensorBoard startup warning: {e}")

        return self.writer

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _setup_public_url(self, port):
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(port, bind_tls=True)
            print(f"🎯 TensorBoard URL: {public_url}")
        except ImportError:
            print(f"📊 TensorBoard on port {port}")
            print("💡 Install: !pip install pyngrok")

    def log_metrics(self, metrics: dict, step: int):
        """Log multiple metrics"""
        if self.writer:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, step)
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
        if self.process:
            print("TensorBoard process terminated")