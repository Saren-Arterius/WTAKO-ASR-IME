import os
import sys
import time
import numpy as np
import io
import wave
import json
from email.parser import BytesParser
from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse

from backends.glm_backend import GLMBackend
from backends.sensevoice_backend import SenseVoiceBackend
from backends.whisper_backend import WhisperBackend

class ASRServer:
    def __init__(self, port, backend_type="glm", config=None):
        self.port = port
        self.config = config or {}
        if backend_type == "glm":
            self.backend = GLMBackend(config=self.config)
        elif backend_type == "sensevoice" or backend_type == "sherpa-onnx/sense-voice":
            self.backend = SenseVoiceBackend(config=self.config)
        elif backend_type == "whisper":
            self.backend = WhisperBackend(config=self.config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None, **kwargs):
        return self.backend.transcribe(audio_data, sample_rate, system_prompt, history, **kwargs)

    def run(self):
        server_instance = self
        class ASRRequestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_type = self.headers.get('Content-Type', '')
                system_prompt = None
                audio_np = None
                sample_rate = None

                if content_type.startswith('multipart/form-data'):
                    try:
                        # Parse multipart/form-data using email.parser
                        content_length = int(self.headers.get('Content-Length', 0))
                        body = self.rfile.read(content_length)
                        
                        # Construct a full message with headers and body for the parser
                        msg_headers = f"Content-Type: {content_type}\r\n\r\n".encode('ascii')
                        msg = BytesParser().parsebytes(msg_headers + body)
                        
                        if msg.is_multipart():
                            for part in msg.get_payload():
                                name = part.get_param('name', header='content-disposition')
                                if name == 'system_prompt':
                                    system_prompt = part.get_payload(decode=True).decode('utf-8')
                                elif name == 'audio':
                                    audio_data_bytes = part.get_payload(decode=True)
                                    audio_file = io.BytesIO(audio_data_bytes)
                                    try:
                                        with wave.open(audio_file, 'rb') as wav_file:
                                            params = wav_file.getparams()
                                            frames = wav_file.readframes(params.nframes)
                                            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                                            sample_rate = params.framerate
                                            if params.nchannels > 1:
                                                audio_np = audio_np.reshape(-1, params.nchannels).mean(axis=1)
                                    except Exception as e:
                                        print(f"Error parsing WAV from multipart: {e}")
                    except Exception as e:
                        print(f"Error parsing multipart data: {e}")
                else:
                    # Fallback to raw WAV in body
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length)
                    try:
                        with io.BytesIO(post_data) as bio:
                            with wave.open(bio, 'rb') as wav_file:
                                params = wav_file.getparams()
                                frames = wav_file.readframes(params.nframes)
                                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                                sample_rate = params.framerate
                                if params.nchannels > 1:
                                    audio_np = audio_np.reshape(-1, params.nchannels).mean(axis=1)
                    except Exception as e:
                        print(f"Error parsing raw WAV: {e}")

                if audio_np is None:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"No audio data found")
                    return

                print(f"System Prompt: {system_prompt}")
                
                # Extract other potential settings from headers or multipart
                # For now, let's check if there are any other form fields
                extra_kwargs = {}
                if content_type.startswith('multipart/form-data'):
                    try:
                        for part in msg.get_payload():
                            name = part.get_param('name', header='content-disposition')
                            if name not in ['system_prompt', 'audio'] and name is not None:
                                value = part.get_payload(decode=True).decode('utf-8')
                                # Handle nested dictionaries if sent as JSON strings or just pass as is
                                try:
                                    # If it looks like a dict/list, try to parse it
                                    if value.startswith('{') or value.startswith('['):
                                        extra_kwargs[name] = json.loads(value)
                                    else:
                                        extra_kwargs[name] = value
                                except:
                                    extra_kwargs[name] = value
                    except Exception:
                        pass

                text = server_instance.transcribe(audio_np, sample_rate, system_prompt=system_prompt, **extra_kwargs)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(text.encode('utf-8'))

        httpd = HTTPServer(('0.0.0.0', self.port), ASRRequestHandler)
        print(f"HTTP ASR Server listening on port {self.port}...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopping...")
            httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--backend", type=str, default="glm", choices=["glm", "sensevoice", "sherpa-onnx/sense-voice", "whisper"], help="ASR backend to use")
    parser.add_argument("--config", type=str, help="Path to config.json")
    parser.add_argument("--config-json", type=str, help="JSON string of config")
    args = parser.parse_args()

    config = {}
    if args.config_json:
        try:
            config = json.loads(args.config_json)
        except Exception as e:
            print(f"Error parsing --config-json: {e}")
    elif args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Try default location
        default_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client", "config.json")
        if os.path.exists(default_config):
            with open(default_config, 'r') as f:
                config = json.load(f)

    server = ASRServer(args.port, backend_type=args.backend, config=config)
    server.run()
