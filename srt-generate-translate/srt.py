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
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from pyannote.audio import Pipeline

# Configuration
ASR_API_URL = "http://100.64.0.8:8003/v1"
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
LLM_API_URL = "http://100.64.0.8:8000/v1"
LLM_MODEL = "Intel/Qwen3-Coder-Next-int4-AutoRound"
ASR_BACKEND = "openai"  # "openai", "sensevoice", or "qwen_asr"
SOURCE_LANG = "ja"
DEST_LANG = "Traditional Chinese (Taiwan/Hong Kong style)"
MERGE_DURATION_FORCE = 0.2

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

    # Convert bytes to tensor
    audio_stream = io.BytesIO(audio_bytes)
    wav = read_audio(audio_stream, sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    return speech_timestamps, wav


def merge_short_segments(segments, min_duration=None, force_duration=None):
    print(segments)
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

        output = pipeline(temp_wav, hook=hook, **pipeline_kwargs)
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
        response = client.chat.completions.create(
            model=ASR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {
                        "data": audio_b64, "format": "wav"}}
                ]
            }],
            extra_body={"chat_template_kwargs": {"language": SOURCE_LANG}}
        )
        text = response.choices[0].message.content
        if '<asr_text>' in text:
            text = text.split('<asr_text>')[1].strip()
        return {
            'index': i,
            'start': ts['start'] / 16000,
            'end': ts['end'] / 16000,
            'text': text
        }
    except Exception as e:
        print(f"Error transcribing fragment {i}: {e}")
        return None


def transcribe_fragments(wav_tensor, timestamps, stats_callback=None):
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


def get_translation_prompt(source_lang, dest_lang, chunk_content, context=""):
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

    prompt = f"""You are an expert translator specializing in {source_name} to {dest_lang} localization.

CRITICAL INSTRUCTIONS:
1. TRANSLATE ALL {source_name} text into natural, idiomatic {dest_lang}. 
2. DO NOT leave any {source_name} sentences or phrases untranslated in the output.
3. Maintain the SRT format (index, timestamps) exactly.
4. Use {dest_lang} characters and style.
{f'5. Additional Context/Instructions: {context}' if context else ''}

Input {source_name} SRT:
{chunk_content}

Output {dest_lang} SRT:"""
    return prompt


def translate_chunk(client, chunk_content, converter, context="", stats_callback=None, chunk_id=0):
    prompt = get_translation_prompt(
        SOURCE_LANG, DEST_LANG, chunk_content, context)
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


def translate_srt(input_path, output_path, context="", stats_callback=None):
    print(
        f"Translating to {DEST_LANG} (Parallel)... Context: {context if context else 'None'}")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Group content into SRT blocks by double newlines
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    total_blocks = len(blocks)

    # Split blocks into 4 chunks
    num_chunks = 4
    chunk_size = (total_blocks + num_chunks - 1) // num_chunks
    chunks = []
    chunk_block_counts = []
    for i in range(0, total_blocks, chunk_size):
        chunk_blocks = blocks[i:i + chunk_size]
        chunks.append("\n\n".join(chunk_blocks))
        chunk_block_counts.append(len(chunk_blocks))

    client = OpenAI(base_url=LLM_API_URL, api_key="EMPTY")
    converter = opencc.OpenCC('s2t')

    # Wrapper for stats_callback to include total_blocks
    def wrapped_callback(chunk_id, tokens, duration, chunk_content):
        if stats_callback:
            # Count how many blocks are in the current translated content
            # This is a rough estimate of progress within the chunk
            translated_blocks = len(
                [b for b in chunk_content.split('\n\n') if b.strip()])
            stats_callback(chunk_id, tokens, duration,
                           translated_blocks, total_blocks, chunk_block_counts[chunk_id])

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Use enumerate to provide a unique chunk_id for each thread
        futures = [executor.submit(translate_chunk, client, chunk, converter, context, wrapped_callback, i)
                   for i, chunk in enumerate(chunks)]
        translated_chunks = [f.result() for f in futures]

    with open(output_path, 'w', encoding='utf-8') as f:
        # Join chunks with double newlines and ensure trailing newline
        f.write("\n\n".join(translated_chunks) + "\n\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python srt.py <video_file> [context]")
        return

    video_file = sys.argv[1]
    context = sys.argv[2] if len(sys.argv) > 2 else ""

    # Setup output directory
    video_name = os.path.basename(video_file)
    output_dir = os.path.join("output", video_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Working directory: {output_dir}")

    audio_bytes = extract_audio(video_file)

    # Try community diarization first (no VAD)
    diarization_result = run_diarization_community(audio_bytes)

    if diarization_result is not None:
        timestamps, raw_timestamps = diarization_result

        # Save raw debug diarization SRT before merging
        diarization_srt = os.path.join(output_dir, "debug.diarization.srt")
        with open(diarization_srt, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(raw_timestamps):
                f.write(f"{i+1}\n")
                f.write(
                    f"{format_timestamp(seg['start'] / 16000)} --> {format_timestamp(seg['end'] / 16000)}\n")
                speaker = seg.get('speaker', 'UNKNOWN')
                f.write(
                    f"{speaker} (start: {seg['start']}, end: {seg['end']})\n\n")
        print(f"Saved raw debug diarization to {diarization_srt}")
    else:
        timestamps = None

    # Fallback to VAD if community fails
    if timestamps is None:
        timestamps, wav_tensor = get_vad_timestamps(audio_bytes)
    else:
        # Need wav_tensor for transcription
        # Convert bytes to tensor without loading Silero VAD
        audio_stream = io.BytesIO(audio_bytes)
        with wave.open(audio_stream, 'rb') as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(params.nframes)
            audio_np = np.frombuffer(
                frames, dtype=np.int16).astype(np.float32) / 32768.0
            wav_tensor = torch.from_numpy(audio_np)

    segments = transcribe_fragments(wav_tensor, timestamps)

    source_srt = os.path.join(output_dir, f"{SOURCE_LANG}.srt")
    dest_srt = os.path.join(output_dir, "translated.srt")

    save_srt(segments, source_srt)
    print(f"Saved {SOURCE_LANG} subtitles to {source_srt}")

    translate_srt(source_srt, dest_srt, context)
    print(f"Saved {DEST_LANG} subtitles to {dest_srt}")


if __name__ == "__main__":
    main()
