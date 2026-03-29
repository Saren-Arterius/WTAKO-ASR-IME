import os
import time
import torch
import numpy as np
import wave
import io
from .base import ASRBackend
import base64
import httpx


class QwenASRBackendVLLM(ASRBackend):
    def __init__(self, config=None):
        super().__init__(config)

        qwen_config = self.config.get("qwen_asr", {})

        # vLLM configuration
        self.base_url = qwen_config.get("base_url", "http://localhost:8003/v1")
        self.api_key = qwen_config.get("api_key", "EMPTY")
        model_id = qwen_config.get("model_id", "Qwen/Qwen3-ASR-1.7B")

        print(f"Initializing vLLM client for {model_id} at {self.base_url}...")

        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.model_id = model_id

        self.language = qwen_config.get("language", None)
        if self.language == "auto":
            self.language = None

    def _is_wav_file(self, data):
        """Check if data starts with WAV file header (RIFF)"""
        return isinstance(data, bytes) and len(data) > 12 and data[:4] == b'RIFF'

    def _array_to_wav(self, audio_array, sample_rate):
        """Convert numpy array to WAV format bytes"""
        # Convert to 16-bit PCM if needed
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            # Normalize to [-1, 1] range if needed
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            # Convert to 16-bit PCM
            audio_int = np.int16(audio_array * 32767)
        else:
            audio_int = audio_array.astype(np.int16)

        # Create WAV file in memory
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())

        return output.getvalue()

    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None, **kwargs):
        start_time = time.time()

        print(f"[ASR Prepare] Starting transcription...")

        # Convert audio data to WAV format with proper headers
        if isinstance(audio_data, np.ndarray):
            print(
                f"[ASR Prepare] Audio: dtype={audio_data.dtype}, shape={audio_data.shape}")
            # Convert numpy array to WAV bytes
            wav_start = time.time()
            audio_bytes = self._array_to_wav(audio_data, sample_rate)
            print(
                f"[ASR Prepare] WAV conversion took {time.time() - wav_start:.3f}s, WAV size={len(audio_bytes)} bytes")
        elif isinstance(audio_data, bytes):
            # Check if it's already a WAV file (has headers)
            if self._is_wav_file(audio_data):
                print("[ASR Prepare] Audio is already WAV format")
                audio_bytes = audio_data
            else:
                print("[ASR Prepare] Converting raw bytes to WAV")
                # Convert raw bytes to WAV
                wav_start = time.time()
                audio_bytes = self._array_to_wav(np.frombuffer(
                    audio_data, dtype=np.float32), sample_rate)
                print(
                    f"[ASR Prepare] WAV conversion took {time.time() - wav_start:.3f}s")
        else:
            print(f"[ASR Prepare] Reading from file: {audio_data}")
            file_start = time.time()
            with open(audio_data, 'rb') as f:
                audio_bytes = f.read()
            print(
                f"[ASR Prepare] File read took {time.time() - file_start:.3f}s, size={len(audio_bytes)} bytes")

        # Encode to base64
        b64_start = time.time()
        audio_data_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        print(
            f"[ASR Prepare] Base64 encoding took {time.time() - b64_start:.3f}s, b64_size={len(audio_data_b64)} chars")

        # Use language from config if not provided in kwargs
        language = kwargs.get("language", self.language)
        # If language is still None, check if it's in the config passed via kwargs (from server)
        if not language:
            language = self.config.get("qwen_asr", {}).get("language")
        if language == "auto":
            language = None

        print(f"[ASR Prepare] Language: {language}")

        # Build messages
        content = [{
            "type": "input_audio",
            "input_audio": {
                "data": audio_data_b64,
                "format": "wav"
            }
        }]

        if system_prompt:
            content.insert(0, {"type": "text", "text": system_prompt})
            print(f"[ASR Prepare] System prompt: {system_prompt}")

        print(f"[ASR Prepare] Sending request to vLLM at {self.base_url}...")
        prepare_end = time.time()
        print(
            f"[ASR Timing] Prepare phase completed in {prepare_end - start_time:.3f}s")
        print(f"[ASR Waiting] Waiting for LLM response...")

        # Create transcription request
        response_start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=kwargs.get("max_tokens", 500),
            extra_body={
                "chat_template_kwargs": {"language": language} if language else {}
            },
        )
        print(
            f"[ASR Timing] LLM response received in {time.time() - response_start:.3f}s")

        info = response.choices[0].message.content.split('<asr_text>')
        if len(info) == 1:
            text = info[0].strip()
            print(f"[ASR Result] Transcription: {text}")
        else:
            text = info[1].strip()
            print(f"[ASR Result] {info[0]}; Transcription: {text}")

        print(
            f"[ASR Summary] Total time: {time.time() - start_time:.3f}s, lang={language}")
        return text
