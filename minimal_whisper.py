import torch
import logging
from transformers import pipeline, logging as transformers_logging, GenerationConfig

# Suppress transformers logging
transformers_logging.set_verbosity_error()

# Configuration
MODEL_ID = "openai/whisper-large-v3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
AUDIO_FILE = "/tmp/test.wav"

def main():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    
    # Suppress "Loading weights" progress bar
    from transformers.utils import logging as transformers_logging_utils
    transformers_logging_utils.disable_progress_bar()

    # Initialize the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        dtype=TORCH_DTYPE,
        device=DEVICE,
        model_kwargs={
            "low_cpu_mem_usage": True, 
            "use_safetensors": True,
        }
    )

    # Fix for "The generation config is outdated" error
    # Reload the generation config from the model ID to ensure it has all necessary attributes
    # like lang_to_id and task_to_id.
    pipe.model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)

    # Manually populate lang_to_id and task_to_id if they are missing
    # This is required for Whisper v3 with some transformers versions
    if not hasattr(pipe.model.generation_config, "lang_to_id"):
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        pipe.model.generation_config.lang_to_id = {
            lang: pipe.tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
            for lang in LANGUAGES.keys()
        }
    
    if not hasattr(pipe.model.generation_config, "task_to_id"):
        pipe.model.generation_config.task_to_id = {
            "transcribe": pipe.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            "translate": pipe.tokenizer.convert_tokens_to_ids("<|translate|>")
        }
    
    pipe.model.generation_config.is_multilingual = True

    print(f"Translating {AUDIO_FILE}...")
    # generate_kwargs: 
    # - language: "cantonese" or "yue" (Transformers handles the mapping to <|yue|>)
    # - task: "translate"
    # We pass the updated generation_config explicitly to ensure it's used
    result = pipe(
        AUDIO_FILE,
        generate_kwargs={
            "language": "cantonese", 
            "task": "translate",
            "generation_config": pipe.model.generation_config
        }
    )

    print("\nTranslation:")
    print(result["text"])

if __name__ == "__main__":
    main()
