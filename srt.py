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


def run_diarization_on_fragments(wav_tensor, speech_timestamps):
    print("Running Speaker Diarization (pyannote) on VAD fragments...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=os.environ.get("HF_TOKEN", True)
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        print(f"Warning: Could not load pyannote pipeline: {e}")
        return speech_timestamps

    new_segments = []
    for i, ts in enumerate(speech_timestamps):
        start_sample = ts['start']
        end_sample = ts['end']
        fragment = wav_tensor[start_sample:end_sample].numpy()

        # Save fragment to temporary WAV
        temp_fragment_wav = f"temp_fragment_{i}.wav"
        with wave.open(temp_fragment_wav, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes((fragment * 32767).astype(np.int16).tobytes())

        # Run diarization on the fragment
        diarization = pipeline(temp_fragment_wav)
        os.remove(temp_fragment_wav)

        # pyannote 3.1 returns an Annotation object which has itertracks
        # If it returns a dict-like object (DiarizeOutput), we access the annotation
        annotation = diarization
        if hasattr(diarization, "annotation"):
            annotation = diarization.annotation

        turns = list(annotation.itertracks(yield_label=True))

        if not turns:
            new_segments.append(ts)
            continue

        # If multiple speakers found in this fragment, split it
        for turn, _, speaker in turns:
            # turn.start/end are relative to the fragment start
            new_segments.append({
                'start': start_sample + int(turn.start * 16000),
                'end': start_sample + int(turn.end * 16000),
                'speaker': speaker
            })

    return merge_short_segments(new_segments)


def merge_short_segments(segments, min_duration=0.3):
    if not segments:
        return []

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        duration = (current['end'] - current['start']) / 16000
        # If current segment is too short, merge it with the next one
        if duration < min_duration:
            current['end'] = next_seg['end']
            # Optionally update speaker if needed, here we keep the first
        else:
            merged.append(current)
            current = next_seg

    merged.append(current)

    # Final pass to handle if the last segment is still too short
    if len(merged) > 1 and (merged[-1]['end'] - merged[-1]['start']) / 16000 < min_duration:
        last = merged.pop()
        merged[-1]['end'] = last['end']

    return merged


def run_diarization_community(audio_bytes):
    print("Running Speaker Diarization (pyannote-community-1)...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=os.environ.get("HF_TOKEN", True)
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        print(f"Warning: Could not load pyannote community pipeline: {e}")
        return None

    temp_wav = "temp_diarization_full.wav"
    with open(temp_wav, "wb") as f:
        f.write(audio_bytes)

    output = pipeline(temp_wav)
    os.remove(temp_wav)

    new_segments = []
    # iterate over speech turns without overlapping speech
    for turn, speaker in output.exclusive_speaker_diarization:
        new_segments.append({
            'start': int(turn.start * 16000),
            'end': int(turn.end * 16000),
            'speaker': speaker
        })

    return merge_short_segments(new_segments)


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
            extra_body={"chat_template_kwargs": {"language": "ja"}}
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


def transcribe_fragments(wav_tensor, timestamps):
    client = OpenAI(base_url=ASR_API_URL, api_key="EMPTY")
    total = len(timestamps)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(
            transcribe_one, client, i, ts, wav_tensor, total) for i, ts in enumerate(timestamps)]
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


def translate_chunk(client, chunk_content, converter, context=""):
    prompt = f"""You are an expert translator specializing in Japanese to Traditional Chinese (Taiwan/Hong Kong style) localization for anime.

CRITICAL INSTRUCTIONS:
1. TRANSLATE ALL Japanese text into natural, idiomatic Traditional Chinese. 
2. DO NOT leave any Japanese sentences or phrases untranslated in the output.
3. Maintain the SRT format (index, timestamps) exactly.
4. Use Traditional Chinese characters only.
{f'5. Additional Context/Instructions: {context}' if context else ''}

Input Japanese SRT:
{chunk_content}

Output Traditional Chinese SRT:"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        translated = response.choices[0].message.content.strip()
        return converter.convert(translated)
    except Exception as e:
        print(f"Chunk translation error: {e}")
        return chunk_content.strip()


def translate_srt(input_path, output_path, context=""):
    print(
        f"Translating to Traditional Chinese (Parallel)... Context: {context if context else 'None'}")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Group content into SRT blocks by double newlines
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]

    # Split blocks into 4 chunks
    num_chunks = 4
    chunk_size = (len(blocks) + num_chunks - 1) // num_chunks
    chunks = ["\n\n".join(blocks[i:i + chunk_size])
              for i in range(0, len(blocks), chunk_size)]

    client = OpenAI(base_url=LLM_API_URL, api_key="EMPTY")
    converter = opencc.OpenCC('s2t')

    with ThreadPoolExecutor(max_workers=4) as executor:
        translated_chunks = list(executor.map(
            lambda c: translate_chunk(client, c, converter, context), chunks))

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
    timestamps = run_diarization_community(audio_bytes)

    # Fallback to VAD + Fragmented Diarization if community fails
    if timestamps is None:
        vad_timestamps, wav_tensor = get_vad_timestamps(audio_bytes)
        timestamps = run_diarization_on_fragments(wav_tensor, vad_timestamps)
    else:
        # Need wav_tensor for transcription
        audio_stream = io.BytesIO(audio_bytes)
        # Using silero's read_audio utility for consistency
        _, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad')
        read_audio = utils[2]
        wav_tensor = read_audio(audio_stream, sampling_rate=16000)

    segments = transcribe_fragments(wav_tensor, timestamps)

    jp_srt = os.path.join(output_dir, "jp.srt")
    zh_srt = os.path.join(output_dir, "zh.srt")

    save_srt(segments, jp_srt)
    print(f"Saved Japanese subtitles to {jp_srt}")

    translate_srt(jp_srt, zh_srt, context)
    print(f"Saved Traditional Chinese subtitles to {zh_srt}")


if __name__ == "__main__":
    main()
