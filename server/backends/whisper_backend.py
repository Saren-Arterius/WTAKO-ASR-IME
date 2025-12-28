import os
import json
import time
import torch
import librosa
from transformers import pipeline, logging as transformers_logging
from .base import ASRBackend

# Suppress transformers logging
transformers_logging.set_verbosity_error()

MODEL_ID = "openai/whisper-large-v3"
TARGET_SAMPLE_RATE = 16000

class WhisperBackend(ASRBackend):
    def __init__(self, config=None):
        super().__init__(config)
        print(f"Loading {MODEL_ID} model...")
        
        whisper_config = self.config.get("whisper", {})
        self.device = whisper_config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        from transformers import GenerationConfig
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_kwargs={"low_cpu_mem_usage": True, "use_safetensors": True}
        )

        # Fix for "The generation config is outdated" error
        # Reload the generation config from the model ID to ensure it has all necessary attributes
        # like lang_to_id and task_to_id.
        self.pipe.model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)

        # Manually populate lang_to_id and task_to_id if they are missing
        # This is required for Whisper v3 with some transformers versions
        if not hasattr(self.pipe.model.generation_config, "lang_to_id"):
            from transformers.models.whisper.tokenization_whisper import LANGUAGES
            self.pipe.model.generation_config.lang_to_id = {
                lang: self.pipe.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
                for lang in LANGUAGES.keys()
            }
        
        if not hasattr(self.pipe.model.generation_config, "task_to_id"):
            self.pipe.model.generation_config.task_to_id = {
                "transcribe": self.pipe.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
                "translate": self.pipe.tokenizer.convert_tokens_to_ids("<|translate|>")
            }
        
        self.pipe.model.generation_config.is_multilingual = True

        language = whisper_config.get("language", "yue")
        
        self.generate_kwargs = {
            "task": whisper_config.get("task", "translate"),
        }
        
        if language and language != "auto":
            self.generate_kwargs["language"] = language

        print(self.generate_kwargs)

    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None, **kwargs):
        start_time = time.time()
        
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        
        # Merge default generate_kwargs with those passed in transcribe call
        merged_kwargs = self.generate_kwargs.copy()
        merged_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
        
        # Remove 'device' from kwargs as it's not a valid argument for generate()
        # and is already handled by the pipeline
        if "device" in merged_kwargs:
            del merged_kwargs["device"]
            
        kwargs = {k: v for k, v in merged_kwargs.items() if v is not None}
        # Pass generation_config explicitly to ensure it's used
        kwargs["generation_config"] = self.pipe.model.generation_config
        
        if system_prompt:
            kwargs["prompt_ids"] = self.pipe.tokenizer.get_prompt_ids(system_prompt, return_tensors="pt").to(self.device)

        result = self.pipe(audio_data, generate_kwargs=kwargs)
        text = result["text"].strip()
        
        print(f"Whisper-v3-large took {time.time() - start_time:.2f}s")
        return text
