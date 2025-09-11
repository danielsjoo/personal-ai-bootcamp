import threading
import time
import psutil
from pynvml import *
from torch.utils.tensorboard import SummaryWriter

class SystemMonitor:
    """
    A thread-safe class to monitor and log system resource usage to TensorBoard.
    
    This monitor is designed for DDP and will only activate on rank 0.
    """
    def __init__(self, writer: SummaryWriter, rank: int, interval: int = 15):
        # This monitor should only be active on the main process
        if rank != 0:
            self.thread = None
            return

        self.writer = writer
        self.rank = rank
        self.interval = interval
        self.stop_event = threading.Event()
        
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(self.rank)
            print(f"[Monitor Rank {rank}] NVML initialized successfully.")
        except NVMLError as e:
            print(f"[Monitor Rank {rank}] Failed to initialize NVML: {e}. GPU monitoring will be disabled.")
            self.gpu_handle = None

        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)

    def _monitor_loop(self):
        """The main monitoring loop that runs in the background."""
        print(f"[Monitor Rank {self.rank}] Starting system monitor thread.")
        while not self.stop_event.is_set():
            # Log CPU and RAM usage
            cpu_percent = psutil.cpu_percent()
            mem_percent = psutil.virtual_memory().percent
            self.writer.add_scalar("System/CPU_Usage_Percent", cpu_percent, time.time())
            self.writer.add_scalar("System/RAM_Usage_Percent", mem_percent, time.time())

            # Log GPU stats if NVML was initialized
            if self.gpu_handle:
                try:
                    gpu_util = nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                    gpu_mem_info = nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_mem_percent = 100 * gpu_mem_info.used / gpu_mem_info.total
                    self.writer.add_scalar(f"GPU_{self.rank}/GPU_Utilization_Percent", gpu_util, time.time())
                    self.writer.add_scalar(f"GPU_{self.rank}/VRAM_Usage_Percent", gpu_mem_percent, time.time())
                except NVMLError as e:
                    print(f"[Monitor Rank {self.rank}] Could not poll GPU stats: {e}")

            time.sleep(self.interval)
        
        if self.gpu_handle:
            nvmlShutdown()
        print(f"[Monitor Rank {self.rank}] System monitor thread stopped.")

    def start(self):
        """Starts the monitoring thread if it exists."""
        if self.thread:
            self.thread.start()

    def stop(self):
        """Signals the monitoring thread to stop."""
        if self.thread:
            self.stop_event.set()