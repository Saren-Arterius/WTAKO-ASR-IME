import time
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoProcessor
from .base import ASRBackend

MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
TARGET_SAMPLE_RATE = 16000

class GLMBackend(ASRBackend):
    def __init__(self, config=None):
        super().__init__(config)
        print("Loading GLM-ASR model...")
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto", local_files_only=True)
        except Exception as e:
            print(f"Local model files not found or error loading locally: {e}. Attempting to download/load from internet...")
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
        self.device_model = self.model.device

    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None, **kwargs):
        start_time = time.time()
        audio_tensor = torch.from_numpy(audio_data).to(torch.float32)
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        
        if system_prompt or history:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            
            messages.append({"role": "user", "content": [{"type": "audio", "audio": audio_tensor.cpu().numpy()}]})
            
            for msg in messages:
                if isinstance(msg["content"], str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]
            
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, sampling_rate=TARGET_SAMPLE_RATE)
        else:
            inputs = self.processor.apply_transcription_request(audio_tensor, sampling_rate=TARGET_SAMPLE_RATE)
            
        inputs = {k: v.to(self.device_model) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if hasattr(self.model, "dtype"):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    inputs[k] = v.to(self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=500)
        
        decoded = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        
        duration = time.time() - start_time
        print(f"GLM-ASR took {duration:.2f}s")
        return text
