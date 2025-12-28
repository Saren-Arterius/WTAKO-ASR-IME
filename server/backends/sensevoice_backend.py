import time
import numpy as np
import sherpa_onnx
import os
import hashlib
import requests
import tarfile
from .base import ASRBackend

class SenseVoiceBackend(ASRBackend):
    def __init__(self, config=None):
        super().__init__(config)
        print("Loading SenseVoice model...")
        
        model_dir = self.config.get("model_dir", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
        num_threads = self.config.get("num_threads", 2)
        self.language = self.config.get("language", "auto")
        provider = self.config.get("provider", "cpu")
        
        self._ensure_model(model_dir)
        
        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        
        if not os.path.exists(model_path) or not os.path.exists(tokens_path):
            raise FileNotFoundError(f"SenseVoice model files not found in {model_dir}")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            use_itn=True,
            provider=provider,
            language=self.language,
        )

    def _ensure_model(self, model_dir):
        model_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
        expected_sha256 = "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"
        model_filename = os.path.basename(model_url)
        model_path = os.path.join(model_dir, "model.int8.onnx")

        if os.path.exists(model_path):
            # Verify existing model
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() == expected_sha256:
                return
            else:
                print(f"Model file {model_path} checksum mismatch. Re-downloading...")

        print(f"Downloading SenseVoice model to {model_dir}...")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Extracting {model_filename}...")
            with tarfile.open(model_filename, "r:bz2") as tar:
                tar.extractall(path=".")
            
            os.remove(model_filename)
            
            # Verify after download
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != expected_sha256:
                print(f"Warning: Downloaded model checksum mismatch! Expected {expected_sha256}, got {sha256_hash.hexdigest()}")
        else:
            raise Exception(f"Failed to download model from {model_url}")

    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None, **kwargs):
        start_time = time.time()
        
        language = kwargs.get("language", self.language)

        # SenseVoice expects 16kHz
        if sample_rate != 16000:
            # Simple resampling if needed, but usually handled by the caller or we can use sherpa-onnx's resampler if available
            # For now assume caller provides 16kHz or we handle it here
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_data)
        # If language is overridden in kwargs, we might need a different recognizer or 
        # just pass it if the recognizer supports it per-stream.
        # sherpa-onnx OfflineRecognizer.from_sense_voice doesn't seem to support per-stream language easily if it's fixed at init.
        # However, SenseVoice is often used with 'auto'.
        
        self.recognizer.decode_stream(stream)
        
        text = stream.result.text
        
        duration = time.time() - start_time
        print(f"SenseVoice took {duration:.2f}s")
        return text
