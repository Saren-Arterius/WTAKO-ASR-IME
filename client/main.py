import os
import sys
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from silero_vad import load_silero_vad
import pyperclip
import socket
import subprocess
import atexit
import io
import wave
import requests
import argparse
import json
import opencc

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "audio_devices": ["C615", "HD Webcam", "HD Camera", "Webcam", "USB Audio"],
        "sound_up": "deck_ui_slider_up.wav",
        "sound_down": "deck_ui_slider_down.wav",
        "socket_path": "/tmp/glm_asr_keyboard.sock",
        "default_asr_server": "http://localhost:8000",
        "hotkey": "f12",
        "gui_scale": 1.0
    }

CONFIG = load_config()

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def play_sound(path, wait=False):
    if path:
        # Resolve path relative to project root (one level up from client/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            if wait:
                subprocess.run(["pw-play", full_path], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            else:
                subprocess.Popen(["pw-play", full_path], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            print(f"Warning: Sound file not found: {full_path}")

def set_mute(mute=True):
    """Mute or unmute the default sink using pactl."""
    try:
        state = "1" if mute else "0"
        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", state], check=False)
    except Exception as e:
        print(f"Error setting mute: {e}")

class ASRClient:
    def __init__(self, asr_server_url, config=None):
        self.asr_server_url = asr_server_url
        self.config = config if config is not None else CONFIG
        print(f"Using ASR server: {self.asr_server_url}")
        
        print("Loading Silero VAD model...")
        self.vad_model = load_silero_vad()
        
        self.uinput_device = self.setup_uinput()
        self.is_recording_dict = {"active": False, "internal_active": False, "cancel": False}
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        
        self.input_device, self.input_sample_rate, self.input_channels = self.find_device(self.config.get("audio_devices", []))
        if self.input_device is None:
            print("Warning: Could not find suitable input device. Using default.")
            self.input_device = None
            self.input_sample_rate = 48000
            self.input_channels = 1
        else:
            print(f"Using device {self.input_device}: {sd.query_devices(self.input_device)['name']} at {self.input_sample_rate}Hz, {self.input_channels} channels")

    def setup_uinput(self):
        import uinput
        return uinput.Device([
            uinput.KEY_LEFTCTRL,
            uinput.KEY_V,
        ])

    def find_device(self, name_substrings):
        devices = sd.query_devices()
        for sub in name_substrings:
            for i, dev in enumerate(devices):
                if sub.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                    rate = int(dev['default_samplerate']) if dev['default_samplerate'] > 0 else 48000
                    return i, rate, dev['max_input_channels']
        return None, 48000, 1

    def wayland_type(self, text):
        if not text: return
        print(f"Typing: {text}")
        import uinput
        old_clipboard = ""
        try:
            old_clipboard = pyperclip.paste()
        except: pass
        pyperclip.copy(text)
        time.sleep(0.1)
        self.uinput_device.emit_combo([uinput.KEY_LEFTCTRL, uinput.KEY_V])
        time.sleep(0.2)
        if old_clipboard:
            try: pyperclip.copy(old_clipboard)
            except: pass

    def check_server_ready(self):
        try:
            # Simple GET request to check if server is up
            response = requests.get(self.asr_server_url, timeout=1)
            # Even if it returns 405 (Method Not Allowed) for GET, it means the server is listening
            return True
        except:
            return False

    def send_to_asr(self, audio_data, sample_rate):
        # Get system prompt from config
        backend = self.config.get("asr_backend", "glm")
        if backend == "sherpa-onnx/sense-voice":
            backend = "sensevoice"
        
        # Convert to WAV in memory
        with io.BytesIO() as bio:
            with wave.open(bio, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(sample_rate)
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_data = bio.getvalue()
        
        try:
            files = {'audio': ('audio.wav', wav_data, 'audio/wav')}
            
            # Prepare data with all settings from the current backend config
            data = {}
            backend_config = self.config.get(backend, {})
            for k, v in backend_config.items():
                if v is not None:
                    # If the value is a dict or list, send it as a JSON string
                    if isinstance(v, (dict, list)):
                        data[k] = json.dumps(v)
                    else:
                        data[k] = str(v)
                
            response = requests.post(self.asr_server_url, files=files, data=data, timeout=60)
            if response.status_code == 200:
                response.encoding = 'utf-8'
                text = response.text
                
                # Apply OpenCC immediately after receiving server response
                opencc_mode = self.config.get("opencc_convert")
                if opencc_mode:
                    try:
                        converter = opencc.OpenCC(opencc_mode)
                        text = converter.convert(text)
                        print(f"OpenCC converted ({opencc_mode}): {text}")
                    except Exception as e:
                        print(f"OpenCC conversion error: {e}")
                return text
            else:
                print(f"ASR Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"ASR Request failed: {e}")
        return ""

    def audio_callback(self, indata, frames, time_info, status):
        if status: print(f"Audio Status: {status}", file=sys.stderr)
        if self.is_recording_dict["active"] or self.is_recording_dict["internal_active"]:
            self.audio_queue.put(indata.copy())

    def recording_loop(self):
        VAD_SAMPLE_RATE = 16000
        FRAME_DURATION_MS = 32
        PADDING_DURATION_MS = 640
        max_silent_frames = int(PADDING_DURATION_MS / FRAME_DURATION_MS)

        while not self.stop_event.is_set():
            if not self.is_recording_dict["active"]:
                time.sleep(0.1)
                continue
            
            if not self.check_server_ready():
                print("ASR Server not ready. Waiting...")
                time.sleep(1)
                continue
                
            self.is_recording_dict["internal_active"] = True
            print("Triggered! Playing sound...")
            play_sound(self.config.get("sound_up"), wait=True)
            set_mute(True)
            # Wait a bit for the system to actually mute and for any residual audio to clear
            time.sleep(0.1)
            print("Muted. VAD Listening...")
            
            recorded_audio = []
            num_silent_frames = 0
            # Clear queue AFTER muting to ensure no pre-mute audio is processed
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            active = False
            speech_detected = False
            
            while not self.stop_event.is_set():
                if self.is_recording_dict.get("cancel"):
                    print("VAD: Cancelled by user")
                    recorded_audio = []
                    break

                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if self.input_channels > 1:
                    chunk_mono = np.mean(chunk, axis=1, keepdims=False).reshape(-1, 1)
                else:
                    chunk_mono = chunk.reshape(-1, 1)

                chunk_tensor = torch.from_numpy(chunk_mono.copy()).to(torch.float32).T
                if self.input_sample_rate != VAD_SAMPLE_RATE:
                    resampler_vad = torchaudio.transforms.Resample(self.input_sample_rate, VAD_SAMPLE_RATE)
                    chunk_vad_tensor = resampler_vad(chunk_tensor)
                else:
                    chunk_vad_tensor = chunk_tensor

                if chunk_vad_tensor.shape[1] < 512:
                    chunk_vad_tensor = torch.nn.functional.pad(chunk_vad_tensor, (0, 512 - chunk_vad_tensor.shape[1]))

                with torch.no_grad():
                    speech_prob = self.vad_model(chunk_vad_tensor.squeeze(0), VAD_SAMPLE_RATE).item()
                
                is_speech = speech_prob > 0.5
                
                if is_speech:
                    if not active:
                        print(f"VAD: Speech started (prob: {speech_prob:.2f})")
                        active = True
                        speech_detected = True
                    num_silent_frames = 0
                    recorded_audio.append(chunk_mono)
                elif active:
                    recorded_audio.append(chunk_mono)
                    num_silent_frames += 1
                    if num_silent_frames > max_silent_frames:
                        print("VAD: Silence timeout")
                        active = False
                        break
                elif not speech_detected:
                    pass

            if recorded_audio:
                print(f"Processing {len(recorded_audio)} chunks of audio...")
                full_audio = np.concatenate(recorded_audio).flatten()
                text = self.send_to_asr(full_audio, self.input_sample_rate)
                if text:
                    print(f"Result: {text}")
                    self.wayland_type(text)
            
            self.is_recording_dict["active"] = False
            self.is_recording_dict["internal_active"] = False
            self.is_recording_dict["cancel"] = False
            print("Recording cycle finished. Waiting for next trigger.")
            play_sound(self.config.get("sound_down"))
            set_mute(False)

    def socket_listener(self):
        socket_path = self.config.get("socket_path", "/tmp/glm_asr_keyboard.sock")
        if os.path.exists(socket_path):
            os.remove(socket_path)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.bind(socket_path)
            os.chmod(socket_path, 0o666)
            s.listen()
            print(f"Listening for keyboard events on {socket_path}...")
            while not self.stop_event.is_set():
                s.settimeout(1.0)
                try:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data == b"DOWN":
                            if self.is_recording_dict["active"]:
                                self.is_recording_dict["cancel"] = True
                            else:
                                self.is_recording_dict["active"] = True
                                self.is_recording_dict["cancel"] = False
                        elif data == b"UP":
                            pass
                except socket.timeout: continue
                except Exception as e:
                    if not self.stop_event.is_set(): print(f"Socket error: {e}")

    def start(self):
        # Start threads
        threading.Thread(target=self.socket_listener, daemon=True).start()
        threading.Thread(target=self.recording_loop, daemon=True).start()
        
        # Start keyboard listener
        self.start_keyboard_subprocess()
        
        # Start audio input stream
        CHUNK_SIZE = int(self.input_sample_rate * 32 / 1000)
        try:
            with sd.InputStream(device=self.input_device, samplerate=self.input_sample_rate, channels=self.input_channels, callback=self.audio_callback, blocksize=CHUNK_SIZE):
                while not self.stop_event.is_set():
                    time.sleep(1)
        except KeyboardInterrupt:
            self.stop_event.set()
            print("\nExiting...")

    def start_keyboard_subprocess(self):
        print("Starting keyboard listener with sudo...")
        hotkey = self.config.get("hotkey", "f12")
        cmd = ["sudo", sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "keyboard_listener.py"), "--hotkey", hotkey]
        # Suppress stderr to avoid "No such device" tracebacks on exit
        self.keyboard_proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
        return self.keyboard_proc

    def start_local_server(self):
        print("Starting local ASR server...")
        server_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "server.py")
        backend = self.config.get("asr_backend", "glm")
        
        # Pass the current in-memory CONFIG as a JSON string to the server
        # This avoids overwriting config.json while ensuring the server uses latest UI settings
        cmd = [sys.executable, server_script, "--backend", backend, "--config-json", json.dumps(self.config)]
        self.server_proc = subprocess.Popen(cmd)
        return self.server_proc

    def stop(self):
        print("Stopping ASRClient...")
        self.stop_event.set()
        
        # Ensure unmuted on stop
        set_mute(False)

        # Cleanup socket
        socket_path = self.config.get("socket_path", "/tmp/glm_asr_keyboard.sock")
        if os.path.exists(socket_path):
            try:
                os.remove(socket_path)
            except:
                pass

        # Close uinput device
        if hasattr(self, 'uinput_device') and self.uinput_device:
            try:
                # python-uinput devices don't have a close method, 
                # but they are closed when the object is deleted.
                del self.uinput_device
            except:
                pass

        if hasattr(self, 'keyboard_proc') and self.keyboard_proc:
            print("Cleaning up keyboard listener...")
            try:
                # Use sudo kill -9 to be more aggressive if needed, 
                # but first try regular kill to allow for some cleanup
                subprocess.run(["sudo", "kill", "-TERM", str(self.keyboard_proc.pid)], stderr=subprocess.DEVNULL)
                # Also kill by name just in case pid tracking failed or it spawned children
                subprocess.run(["sudo", "pkill", "-f", "keyboard_listener.py"], stderr=subprocess.DEVNULL)
                self.keyboard_proc.terminate()
                try:
                    self.keyboard_proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    subprocess.run(["sudo", "kill", "-9", str(self.keyboard_proc.pid)], stderr=subprocess.DEVNULL)
            except:
                pass
            self.keyboard_proc = None

        if hasattr(self, 'server_proc') and self.server_proc:
            print("Cleaning up local ASR server...")
            try:
                self.server_proc.terminate()
                try:
                    self.server_proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.server_proc.kill()
            except:
                pass
            self.server_proc = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-ASR Client")
    parser.add_argument("--asr-server", type=str, help="ASR server URL (if not provided, starts local server)")
    args = parser.parse_args()

    server_url = args.asr_server
    local_server_proc = None
    if not server_url:
        server_url = CONFIG.get("default_asr_server", "http://localhost:8000")
        client = ASRClient(server_url)
        local_server_proc = client.start_local_server()
        # Give the server a moment to start
        time.sleep(2)
    else:
        client = ASRClient(server_url)
        
    client.start()
