import os
import time
import torch
import numpy as np
from .base import ASRBackend


class QwenASRBackend(ASRBackend):
    def __init__(self, config=None):
        super().__init__(config)

        qwen_config = self.config.get("qwen_asr", {})
        model_id = qwen_config.get("model_id", "Qwen/Qwen3-ASR-1.7B")
        self.device = qwen_config.get(
            "device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if "cuda" in self.device else torch.float32

        print(f"Loading {model_id} model on {self.device}...")

        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=qwen_config.get(
                "max_inference_batch_size", 32),
            max_new_tokens=qwen_config.get("max_new_tokens", 256),
        )

        self.language = qwen_config.get("language", None)
        if self.language == "auto":
            self.language = None

    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None, **kwargs):
        start_time = time.time()

        # Qwen3-ASR accepts (np.ndarray, sr) tuple
        audio_input = (audio_data, sample_rate)

        # Use language from config if not provided in kwargs
        language = kwargs.get("language", self.language)

        results = self.model.transcribe(
            audio=audio_input,
            language=language,
        )

        text = results[0].text.strip()

        print(
            f"Qwen3-ASR took {time.time() - start_time:.2f}s, lang={language}")
        return text
