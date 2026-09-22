import queue
import sounddevice as sd

class AudioListener:
    def __init__(self, sample_rate=22050, chunk_duration=1.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.q = queue.Queue()
        self.stream = None
        
    def callback(self, indata, frames, time, status):
        """This is called for each audio block by sounddevice"""
        if status:
            print(f"SoundDevice Status: {status}")
        # Put a copy of the audio block into the queue
        self.q.put(indata.copy())

    def start(self):
        # We try to use the default device, mono input
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1,
            callback=self.callback, 
            blocksize=self.chunk_samples
        )
        self.stream.start()
        
    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
    def get_latest_chunk(self):
        # Flush the queue and get the most recent chunk
        latest = None
        while not self.q.empty():
            try:
                latest = self.q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            return latest.flatten()
        return None
