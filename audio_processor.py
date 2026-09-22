import queue

import numpy as np
import sounddevice as sd


class AudioListener:
    """Queue-based microphone listener for a simple live monitoring loop."""

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = max(1, int(sample_rate * chunk_duration))
        self.queue = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"SoundDevice status: {status}")
        self.queue.put(np.asarray(indata, dtype=np.float32).copy())

    def start(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.callback,
                blocksize=self.chunk_samples,
            )
            self.stream.start()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Could not open microphone: {exc}") from exc

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None

    def get_latest_chunk(self):
        latest = None
        while not self.queue.empty():
            try:
                latest = self.queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            return np.asarray(latest, dtype=np.float32).reshape(-1)
        return None
