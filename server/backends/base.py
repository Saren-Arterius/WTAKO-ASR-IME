from abc import ABC, abstractmethod

class ASRBackend(ABC):
    def __init__(self, config=None):
        self.config = config or {}

    @abstractmethod
    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None, **kwargs):
        pass
