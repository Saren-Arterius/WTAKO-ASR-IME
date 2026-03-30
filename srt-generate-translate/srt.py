import time
import os
import sys
import torch
import numpy as np
import subprocess
import json
import httpx
import opencc
import wave
import io
import base64
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from pyannote.audio import Pipeline

# Configuration Defaults
ASR_API_URL = "http://100.64.0.8:8003/v1"
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
LLM_API_URL = "http://100.64.0.8:8000/v1"
LLM_MODEL = "Intel/Qwen3-Coder-Next-int4-AutoRound"
ASR_BACKEND = "openai"  # "openai", "sensevoice", or "qwen_asr"
SOURCE_LANG = "ja"
DEST_LANG = "Traditional Chinese (Taiwan/Hong Kong style)"
MERGE_DURATION = 0.5
MERGE_DURATION_FORCE = 0.2
HF_TOKEN = ""
USE_DIARIZATION = True
MIN_SPEAKERS = 0
UNLOAD_MODELS_AFTER_USE = False
SAVE_DEBUG_SRT = False
SAVE_ORIGIN_SRT = True

_sensevoice_recognizer = None
_qwen_asr_model = None
_diarization_pipeline = None


def unload_models():
    global _sensevoice_recognizer, _qwen_asr_model, _diarization_pipeline
    if _qwen_asr_model is not None:
        print("Unloading Qwen-ASR model...")
        del _qwen_asr_model
        _qwen_asr_model = None

    if _sensevoice_recognizer is not None:
        print("Unloading SenseVoice recognizer...")
        del _sensevoice_recognizer
        _sensevoice_recognizer = None

    if _diarization_pipeline is not None:
        print("Unloading Diarization pipeline...")
        del _diarization_pipeline
        _diarization_pipeline = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def get_qwen_asr_model():
    global _qwen_asr_model
    if _qwen_asr_model is None:
        import torch
        from qwen_asr import Qwen3ASRModel
        model_id = globals().get('ASR_MODEL', "Qwen/Qwen3-ASR-1.7B")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        print(f"Loading {model_id} model on {device}...")
        _qwen_asr_model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
    return _qwen_asr_model


def get_sensevoice_recognizer():
    global _sensevoice_recognizer
    if _sensevoice_recognizer is None:
        import sherpa_onnx
        model_dir = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        _sensevoice_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=4,
            use_itn=True,
            provider="cpu",
        )
    return _sensevoice_recognizer


def extract_audio(video_path):
    print(f"Extracting audio from {video_path}...")
    # Extract to 16kHz mono wav for VAD and ASR
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        '-f', 'wav', 'pipe:1'
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data, _ = process.communicate()
    return audio_data


def get_vad_timestamps(audio_bytes):
    print("Running Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    temp_wav = "temp_vad_full.wav"
    with open(temp_wav, "wb") as f:
        f.write(audio_bytes)

    try:
        # Try normal file-path decode first (torchaudio/default path)
        wav = read_audio(temp_wav, sampling_rate=16000)
    except Exception as e:
        print(
            f"Warning: File-based VAD audio loading failed ({e}). Retrying with in-memory audio input fallback.")
        audio_input = wav_bytes_to_input_dict(audio_bytes)
        waveform = audio_input["waveform"]
        wav = waveform[0] if waveform.ndim > 1 else waveform
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    return speech_timestamps, wav


def wav_bytes_to_input_dict(audio_bytes):
    audio_stream = io.BytesIO(audio_bytes)
    with wave.open(audio_stream, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (channel, time)
    return {"waveform": waveform, "sample_rate": sample_rate}


def merge_short_segments(segments, min_duration=None, force_duration=None):
    if min_duration is None:
        # Try to get from config or default
        min_duration = globals().get('MERGE_DURATION', 0.5)
    if force_duration is None:
        force_duration = globals().get('MERGE_DURATION_FORCE', 0.2)

    if not segments:
        return []

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        duration = (current['end'] - current['start']) / 16000

        # Check for force merge (regardless of speaker)
        should_force_merge = duration < force_duration

        # Check for normal merge (same speaker)
        should_normal_merge = (duration < min_duration and
                               current.get('speaker') == next_seg.get('speaker'))

        # BUT ignore if either has do_not_merge flag
        can_merge = not current.get(
            'do_not_merge') and not next_seg.get('do_not_merge')

        if can_merge and (should_force_merge or should_normal_merge):
            current['end'] = next_seg['end']
        else:
            merged.append(current)
            current = next_seg

    merged.append(current)

    # Final pass to handle if the last segment is still too short
    if len(merged) > 1:
        last_duration = (merged[-1]['end'] - merged[-1]['start']) / 16000
        should_force_merge = last_duration < force_duration
        should_normal_merge = (last_duration < min_duration and
                               merged[-1].get('speaker') == merged[-2].get('speaker'))

        if (should_force_merge or should_normal_merge):
            last = merged.pop()
            merged[-1]['end'] = last['end']

    return merged


def run_diarization_community(audio_bytes, min_speakers=None, stats_callback=None):
    global _diarization_pipeline
    print("Running Speaker Diarization (pyannote-community-1)...")
    try:
        if _diarization_pipeline is None:
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=os.environ.get("HF_TOKEN", True)
            )
            if torch.cuda.is_available():
                _diarization_pipeline.to(torch.device("cuda"))
        pipeline = _diarization_pipeline
    except Exception as e:
        print(f"Warning: Could not load pyannote community pipeline: {e}")
        return None

    temp_wav = "temp_diarization_full.wav"
    with open(temp_wav, "wb") as f:
        f.write(audio_bytes)

    from pyannote.audio.pipelines.utils.hook import ProgressHook
    with ProgressHook() as hook:
        if stats_callback:
            # Hook into the progress hook to update our GUI
            original_call = hook.__call__

            def hooked_call(step_name, completed, total=None, **kwargs):
                res = original_call(step_name, completed, total, **kwargs)
                if total:
                    stats_callback(step_name, completed, total)
                return res
            hook.__call__ = hooked_call

        pipeline_kwargs = {}
        if min_speakers is not None and min_speakers > 0:
            pipeline_kwargs["min_speakers"] = min_speakers

        try:
            # Try normal file-path decode first (torchcodec/default path)
            output = pipeline(temp_wav, hook=hook, **pipeline_kwargs)
        except Exception as e:
            print(
                f"Warning: File-based diarization failed ({e}). Retrying with in-memory audio input fallback.")
            audio_input = wav_bytes_to_input_dict(audio_bytes)
            output = pipeline(audio_input, hook=hook, **pipeline_kwargs)
    os.remove(temp_wav)

    # Handle overlapping diarization results: prefer the result that has a later "start"
    # pyannote output.itertracks() provides segments which might overlap
    all_segments = []
    # For pyannote community pipeline, output might be a DiarizeOutput object
    # which contains the annotation in .annotation or similar.
    annotation = output
    if hasattr(output, 'annotation'):
        annotation = output.annotation
    elif hasattr(output, 'exclusive_speaker_diarization'):
        # If we can't get the raw annotation with overlaps, we might have to use this,
        # but the goal is to handle overlaps ourselves.
        # Let's try to find the most raw annotation.
        annotation = output.exclusive_speaker_diarization

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        all_segments.append({
            'start': int(turn.start * 16000),
            'end': int(turn.end * 16000),
            'speaker': speaker
        })

    # Sort by start time
    all_segments.sort(key=lambda x: x['start'])

    new_segments = []
    for seg in all_segments:
        if not new_segments:
            new_segments.append(seg)
            continue

        prev = new_segments[-1]
        # Check for overlap
        if seg['start'] < prev['end']:
            # Overlap detected.
            if seg['speaker'] == prev['speaker']:
                # SAME SPEAKER: Merge into the largest possible span
                # 20-30 and 10-40 => 10-40
                # 10-40 and 20-40 => 10-40
                prev['start'] = min(prev['start'], seg['start'])
                prev['end'] = max(prev['end'], seg['end'])
                continue
            else:
                # DIFFERENT SPEAKERS: Split into non-overlapping parts
                if seg['start'] > prev['start']:
                    prev['end'] = seg['start']
                    prev['do_not_merge'] = True
                    seg['do_not_merge'] = True
                    new_segments.append(seg)
                else:
                    if seg['end'] <= prev['end']:
                        continue
                    else:
                        new_segments[-1] = seg
        else:
            new_segments.append(seg)

    # For overlapping cases, we ignore MERGE_DURATION as requested.
    # However, merge_short_segments is still useful for general cleanup.
    # We'll pass a flag or handle it by ensuring the split points are preserved.
    return merge_short_segments(new_segments), new_segments


def format_timestamp(seconds):
    ms = int((seconds % 1) * 1000)
    full_seconds = int(seconds)
    hours = full_seconds // 3600
    minutes = (full_seconds % 3600) // 60
    secs = full_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def transcribe_one(client, i, ts, wav_tensor, total):
    start_sample = ts['start']
    end_sample = ts['end']
    # Add some padding
    start_sample = max(0, start_sample - 1600)
    end_sample = min(len(wav_tensor), end_sample + 1600)

    fragment = wav_tensor[start_sample:end_sample].numpy()

    if ASR_BACKEND == "sensevoice":
        recognizer = get_sensevoice_recognizer()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, fragment)
        recognizer.decode_stream(stream)
        text = stream.result.text
        # SenseVoice might include language tags like <|zh|>, remove them if present
        import re
        text = re.sub(r'<\|.*?\|>', '', text).strip()

        # Check for repeating characters (e.g., "aaaaaaaaaa..." or "うおおおお...")
        import re
        if len(text) > 10 and re.search(r'(.)\1{9,}', text):
            print(
                f"Warning: Segment {i+1} contains repeating characters. Cropping to 10 chars.")
            text = text[:10]
        elif len(text) > 100:
            print(
                f"Warning: Segment {i+1} text too long ({len(text)} chars). Cropping to 100 chars.")
            text = text[:100]
        return {
            'index': i,
            'start': ts['start'] / 16000,
            'end': ts['end'] / 16000,
            'text': text
        }

    if ASR_BACKEND == "qwen_asr":
        model = get_qwen_asr_model()
        # Qwen3-ASR accepts (np.ndarray, sr) tuple
        audio_input = (fragment, 16000)

        # Qwen3-ASR expects full language names
        qwen_lang_map = {
            "ja": "Japanese",
            "zh": "Chinese",
            "en": "English",
            "ko": "Korean",
            "yue": "Cantonese",
        }
        language = qwen_lang_map.get(
            SOURCE_LANG, SOURCE_LANG) if SOURCE_LANG != "auto" else None

        results = model.transcribe(
            audio=audio_input,
            language=language,
        )
        text = results[0].text.strip()

        # Check for repeating characters (e.g., "aaaaaaaaaa..." or "うおおおお...")
        import re
        if len(text) > 10 and re.search(r'(.)\1{9,}', text):
            print(
                f"Warning: Segment {i+1} contains repeating characters. Cropping to 10 chars.")
            text = text[:10]
        elif len(text) > 100:
            print(
                f"Warning: Segment {i+1} text too long ({len(text)} chars). Cropping to 100 chars.")
            text = text[:100]
        return {
            'index': i,
            'start': ts['start'] / 16000,
            'end': ts['end'] / 16000,
            'text': text
        }

    # Convert to WAV bytes
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes((fragment * 32767).astype(np.int16).tobytes())
    audio_b64 = base64.b64encode(output.getvalue()).decode("utf-8")

    print(f"Transcribing fragment {i+1}/{total} ({ts['start']/16000:.2f}s)...")
    try:
        # Use streaming to enforce a total timeout including response generation time
        response = client.chat.completions.create(
            model=ASR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {
                        "data": audio_b64, "format": "wav"}}
                ]
            }],
            extra_body={"chat_template_kwargs": {"language": SOURCE_LANG}},
            stream=True,
            timeout=15.0
        )

        text = ""
        start_time = time.time()
        for chunk in response:
            if time.time() - start_time > 15.0:
                print(
                    f"Warning: Segment {i+1} ASR request timed out during generation (>15s).")
                break
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        if '<asr_text>' in text:
            text = text.split('<asr_text>')[1].strip()
        text = text.strip()

        # Check for repeating characters (e.g., "aaaaaaaaaa..." or "うおおおお...")
        import re
        if len(text) > 10 and re.search(r'(.)\1{9,}', text):
            print(
                f"Warning: Segment {i+1} contains repeating characters. Cropping to 10 chars.")
            text = text[:10]
        elif len(text) > 100:
            print(
                f"Warning: Segment {i+1} text too long ({len(text)} chars). Cropping to 100 chars.")
            text = text[:100]
        return {
            'index': i,
            'start': ts['start'] / 16000,
            'end': ts['end'] / 16000,
            'text': text
        }
    except Exception as e:
        print(f"Error transcribing fragment {i}: {e}")
        # If timeout or other error, return empty string as requested
        return {
            'index': i,
            'start': ts['start'] / 16000,
            'end': ts['end'] / 16000,
            'text': ""
        }


def transcribe_fragments(wav_tensor, timestamps, stats_callback=None):
    # Cap each segment to be 15 seconds max.
    # If a segment is longer than 15 seconds, it's likely glitched; crop it to 15s.
    MAX_SEGMENT_DURATION_S = 15.0
    MAX_SEGMENT_SAMPLES = int(MAX_SEGMENT_DURATION_S * 16000)

    processed_timestamps = []
    for ts in timestamps:
        start = ts['start']
        end = ts['end']
        duration_samples = end - start
        if duration_samples > MAX_SEGMENT_SAMPLES:
            print(
                f"Warning: Segment at {start/16000:.2f}s is {duration_samples/16000:.2f}s long (exceeds 15s). Cropping to 15s.")
            ts['end'] = start + MAX_SEGMENT_SAMPLES
        processed_timestamps.append(ts)

    timestamps = processed_timestamps
    total = len(timestamps)
    completed = 0

    def update_progress():
        nonlocal completed
        completed += 1
        if stats_callback:
            stats_callback(completed, total)

    if ASR_BACKEND in ["sensevoice", "qwen_asr"]:
        # Local inference is usually better sequential
        results = []
        for i, ts in enumerate(timestamps):
            print(
                f"Transcribing fragment {i+1}/{total} ({ts['start']/16000:.2f}s) with {ASR_BACKEND}...")
            res = transcribe_one(None, i, ts, wav_tensor, total)
            if res:
                results.append(res)
            update_progress()
        return results

    client = OpenAI(base_url=ASR_API_URL, api_key="EMPTY")
    with ThreadPoolExecutor(max_workers=8) as executor:
        def wrapped_transcribe(i, ts):
            res = transcribe_one(client, i, ts, wav_tensor, total)
            update_progress()
            return res

        futures = [executor.submit(wrapped_transcribe, i, ts)
                   for i, ts in enumerate(timestamps)]
        results = [f.result() for f in futures]

    # Filter out None and sort by index
    results = sorted([r for r in results if r is not None],
                     key=lambda x: x['index'])
    return results


def save_srt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            f.write(f"{i+1}\n")
            f.write(
                f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


def get_translation_prompt(source_lang, dest_lang, lines, context=""):
    # Map language codes to names for the prompt
    lang_map = {
        "ja": "Japanese",
        "zh": "Chinese",
        "en": "English",
        "ko": "Korean",
        "yue": "Cantonese",
        "auto": "Source Language"
    }
    source_name = lang_map.get(source_lang, source_lang)

    # Format lines as a numbered list for the LLM
    formatted_lines = "\n".join(
        [f"{i+1}. {line}" for i, line in enumerate(lines)])

    prompt = f"""You are an expert translator specializing in {source_name} to {dest_lang} localization.

CRITICAL INSTRUCTIONS:
1. TRANSLATE ALL {source_name} text into natural, idiomatic {dest_lang}. 
2. DO NOT leave any {source_name} sentences or phrases untranslated in the output.
3. Maintain the exact same number of lines in the output.
4. Output ONLY the translated lines, one per line, without the original numbering or timestamps.
5. Use {dest_lang} characters and style.
{f'6. Additional Context/Instructions: {context}' if context else ''}

Input {source_name} lines:
{formatted_lines}

Output {dest_lang} lines:"""
    return prompt


def translate_chunk(client, lines, converter, context="", stats_callback=None, chunk_id=0):
    prompt = get_translation_prompt(
        SOURCE_LANG, DEST_LANG, lines, context)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=True
        )

        full_content = ""
        start_time = None
        token_count = 0

        for chunk in response:
            if chunk.choices[0].delta.content:
                if start_time is None:
                    start_time = time.time()

                content = chunk.choices[0].delta.content
                full_content += content
                token_count += 1  # Rough estimate per chunk
                if stats_callback:
                    # Pass current token count and elapsed time for real-time TPS
                    stats_callback(chunk_id, token_count,
                                   time.time() - start_time, full_content)

        translated = full_content.strip()
        return converter.convert(translated)
    except Exception as e:
        print(f"Chunk translation error: {e}")
        return chunk_content.strip()


def parse_srt(content):
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    parsed = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            index = lines[0]
            timestamp = lines[1]
            text = " ".join(lines[2:])
            parsed.append(
                {'index': index, 'timestamp': timestamp, 'text': text})
    return parsed


def translate_srt(input_path, output_path, context="", stats_callback=None):
    print(
        f"Translating to {DEST_LANG} (Parallel)... Context: {context if context else 'None'}")

    # Internal stats for CLI display
    _chunk_stats = {}

    def cli_stats_callback(chunk_id, tokens, duration, translated_blocks, total_blocks, chunk_total_blocks):
        _chunk_stats[chunk_id] = (tokens, duration, translated_blocks)
        total_tokens = sum(s[0] for s in _chunk_stats.values())
        total_translated = sum(s[2] for s in _chunk_stats.values())
        max_duration = max((s[1] for s in _chunk_stats.values()), default=0)

        if max_duration > 0:
            tps = total_tokens / max_duration
            progress = (total_translated / total_blocks) * \
                100 if total_blocks > 0 else 0
            sys.stdout.write(
                f"\rTranslation Progress: {progress:.1f}% | {tps:.1f} tokens/s")
            sys.stdout.flush()

    # Use provided callback or our CLI one
    effective_callback = stats_callback or cli_stats_callback

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    parsed_blocks = parse_srt(content)
    total_blocks = len(parsed_blocks)
    if total_blocks == 0:
        return

    # Split blocks into 4 chunks
    num_chunks = 4
    chunk_size = (total_blocks + num_chunks - 1) // num_chunks
    chunks_data = []
    for i in range(0, total_blocks, chunk_size):
        chunks_data.append(parsed_blocks[i:i + chunk_size])

    client = OpenAI(base_url=LLM_API_URL, api_key="EMPTY")
    converter = opencc.OpenCC('s2t')

    # Wrapper for stats_callback to include total_blocks
    def wrapped_callback(chunk_id, tokens, duration, chunk_content):
        # Count how many lines are in the current translated content
        translated_lines = len(
            [l for l in chunk_content.split('\n') if l.strip()])
        effective_callback(chunk_id, tokens, duration,
                           translated_lines, total_blocks, len(chunks_data[chunk_id]))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, chunk_blocks in enumerate(chunks_data):
            texts = [b['text'] for b in chunk_blocks]
            futures.append(executor.submit(translate_chunk, client,
                           texts, converter, context, wrapped_callback, i))

        translated_results = [f.result() for f in futures]

    if not stats_callback:
        print()  # New line after progress bar

    # Reconstruct SRT
    with open(output_path, 'w', encoding='utf-8') as f:
        global_idx = 1
        for chunk_idx, result_text in enumerate(translated_results):
            original_chunk_blocks = chunks_data[chunk_idx]
            translated_lines = [l.strip()
                                for l in result_text.split('\n') if l.strip()]

            # If LLM returned wrong number of lines, we might need to handle it,
            # but for now we'll try to match them.
            for i, block in enumerate(original_chunk_blocks):
                f.write(f"{global_idx}\n")
                f.write(f"{block['timestamp']}\n")
                # Use translated line if available, else original
                text = translated_lines[i] if i < len(
                    translated_lines) else block['text']
                # Remove any leading "1. " etc if LLM ignored instructions
                import re
                text = re.sub(r'^\d+\.\s*', '', text)
                f.write(f"{text}\n\n")
                global_idx += 1


def main():
    # Update global variables
    global ASR_API_URL, ASR_MODEL, LLM_API_URL, LLM_MODEL, ASR_BACKEND, SOURCE_LANG, DEST_LANG
    global MERGE_DURATION, MERGE_DURATION_FORCE, HF_TOKEN, USE_DIARIZATION, MIN_SPEAKERS
    global UNLOAD_MODELS_AFTER_USE, SAVE_DEBUG_SRT, SAVE_ORIGIN_SRT

    parser = argparse.ArgumentParser(
        description="WTAKO SRT Generator & Translator CLI")
    parser.add_argument("video_files", nargs="+",
                        help="Video files to process")
    parser.add_argument("--context", type=str, default="",
                        help="Translation context")
    parser.add_argument("--asr-url", type=str,
                        default=ASR_API_URL, help="ASR API URL")
    parser.add_argument("--asr-model", type=str,
                        default=ASR_MODEL, help="ASR Model name")
    parser.add_argument("--asr-backend", type=str, choices=[
                        "openai", "sensevoice", "qwen_asr"], default=ASR_BACKEND, help="ASR Backend")
    parser.add_argument("--source-lang", type=str, default=SOURCE_LANG,
                        help="Source language code (e.g., ja, zh, en)")
    parser.add_argument("--dest-lang", type=str, default=DEST_LANG,
                        help="Destination language description")
    parser.add_argument("--llm-url", type=str,
                        default=LLM_API_URL, help="LLM API URL")
    parser.add_argument("--llm-model", type=str,
                        default=LLM_MODEL, help="LLM Model name")
    parser.add_argument("--merge-duration", type=float,
                        default=MERGE_DURATION, help="Merge duration (s)")
    parser.add_argument("--merge-duration-force", type=float,
                        default=MERGE_DURATION_FORCE, help="Force merge duration (s)")
    parser.add_argument("--min-speakers", type=int,
                        default=MIN_SPEAKERS, help="Minimum speakers for diarization")
    parser.add_argument("--hf-token", type=str, default=HF_TOKEN,
                        help="Hugging Face Token for diarization")
    parser.add_argument("--no-diarization", action="store_false",
                        dest="use_diarization", help="Disable speaker diarization")
    parser.set_defaults(use_diarization=USE_DIARIZATION)
    parser.add_argument("--unload-models", action="store_true",
                        dest="unload_models", help="Unload models after use to save VRAM")
    parser.set_defaults(unload_models=UNLOAD_MODELS_AFTER_USE)
    parser.add_argument("--save-debug-srt", action="store_true",
                        dest="save_debug_srt", help="Save debug diarization SRT")
    parser.set_defaults(save_debug_srt=SAVE_DEBUG_SRT)
    parser.add_argument("--no-origin-srt", action="store_false",
                        dest="save_origin_srt", help="Do not save original language SRT")
    parser.set_defaults(save_origin_srt=SAVE_ORIGIN_SRT)

    args = parser.parse_args()

    ASR_API_URL = args.asr_url
    ASR_MODEL = args.asr_model
    LLM_API_URL = args.llm_url
    LLM_MODEL = args.llm_model
    ASR_BACKEND = args.asr_backend
    SOURCE_LANG = args.source_lang
    DEST_LANG = args.dest_lang
    MERGE_DURATION = args.merge_duration
    MERGE_DURATION_FORCE = args.merge_duration_force
    HF_TOKEN = args.hf_token
    USE_DIARIZATION = args.use_diarization
    MIN_SPEAKERS = args.min_speakers
    UNLOAD_MODELS_AFTER_USE = args.unload_models
    SAVE_DEBUG_SRT = args.save_debug_srt
    SAVE_ORIGIN_SRT = args.save_origin_srt
    context = args.context

    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

    for video_file in args.video_files:
        print(f"\nProcessing: {video_file}")
        base_path = os.path.splitext(video_file)[0]

        audio_bytes = extract_audio(video_file)

        # 2. Diarization / VAD
        timestamps = None
        if USE_DIARIZATION:
            if not HF_TOKEN:
                print("Warning: HF_TOKEN not set. Skipping diarization.")
            else:
                diarization_result = run_diarization_community(
                    audio_bytes,
                    min_speakers=MIN_SPEAKERS
                )
                if diarization_result:
                    timestamps, raw_timestamps = diarization_result
                    if SAVE_DEBUG_SRT:
                        diarization_srt = f"{base_path}.debug.diarization.srt"
                        with open(diarization_srt, 'w', encoding='utf-8') as f:
                            for idx, seg in enumerate(raw_timestamps):
                                f.write(f"{idx+1}\n")
                                f.write(
                                    f"{format_timestamp(seg['start'] / 16000)} --> {format_timestamp(seg['end'] / 16000)}\n")
                                speaker = seg.get('speaker', 'UNKNOWN')
                                f.write(
                                    f"{speaker} (start: {seg['start']}, end: {seg['end']})\n\n")
                        print(f"Saved debug diarization to {diarization_srt}")

        if timestamps is None:
            print("Running VAD...")
            timestamps, wav_tensor = get_vad_timestamps(audio_bytes)
        else:
            # Convert bytes to tensor
            audio_stream = io.BytesIO(audio_bytes)
            with wave.open(audio_stream, 'rb') as wav_file:
                params = wav_file.getparams()
                frames = wav_file.readframes(params.nframes)
                audio_np = np.frombuffer(
                    frames, dtype=np.int16).astype(np.float32) / 32768.0
                wav_tensor = torch.from_numpy(audio_np)

        if UNLOAD_MODELS_AFTER_USE:
            unload_models()

        # 3. Transcribe
        segments = transcribe_fragments(wav_tensor, timestamps)

        source_srt = f"{base_path}.{SOURCE_LANG}.srt"
        dest_srt = f"{base_path}.translated.srt"

        save_srt(segments, source_srt)
        print(f"Saved {SOURCE_LANG} subtitles to {source_srt}")

        if UNLOAD_MODELS_AFTER_USE:
            unload_models()

        # 4. Translate
        translate_srt(source_srt, dest_srt, context)
        print(f"Saved {DEST_LANG} subtitles to {dest_srt}")

        if not SAVE_ORIGIN_SRT:
            try:
                if os.path.exists(source_srt):
                    os.remove(source_srt)
            except Exception as e:
                print(f"Failed to remove original SRT: {e}")

    unload_models()


if __name__ == "__main__":
    main()
