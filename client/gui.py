import os
import sys
import threading
import time
import json
import locale
import webbrowser
from enum import Enum, auto
import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import Image
import sounddevice as sd

# Import the existing ASRClient logic
try:
    from main import ASRClient, CONFIG, play_sound, save_config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from main import ASRClient, CONFIG, play_sound, save_config

# i18n helper
def get_i18n(config):
    lang = config.get("language", "auto")
    if lang == "auto":
        try:
            system_lang = locale.getdefaultlocale()[0]
        except:
            system_lang = None
            
        if system_lang and (system_lang.startswith("zh_TW") or system_lang.startswith("zh_HK")):
            lang = "zh_TW"
        elif system_lang and system_lang.startswith("zh"):
            lang = "zh_TW" # Default to zh_TW for now as requested
        else:
            lang = "en"
    
    # Try to load from i18n directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    # i18n is at the root, client/gui.py is in client/
    i18n_path = os.path.join(os.path.dirname(base_path), "i18n", f"{lang}.json")
    
    if not os.path.exists(i18n_path):
        # Fallback to en
        i18n_path = os.path.join(os.path.dirname(base_path), "i18n", "en.json")
        
    try:
        with open(i18n_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading i18n: {e}")
        return {}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AppState(Enum):
    DISCONNECTED = auto()
    STARTING = auto()
    WAITING_FOR_SERVER = auto()
    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()
    ERROR = auto()

class ASRGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.config = CONFIG.copy()
        self.i18n = get_i18n(self.config)

        # DPI Scaling from Config
        scale = self.config.get("gui_scale", 1.0)
        # ctk.set_widget_scaling(scale)
        # ctk.set_window_scaling(scale)
        
        base_width = 1000
        base_height = 600
        width = int(base_width * scale)
        height = int(base_height * scale)

        self.title(self.i18n.get("title", "WTAKO ASR IME"))
        self.geometry(f"{width}x{height}")

        self.client = None
        self.local_server_proc = None
        self.app_state = AppState.DISCONNECTED
        self.volume_stream = None

        self.setup_ui()
        self.transition_to(AppState.DISCONNECTED)

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="WTAKO ASR IME", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.start_button = ctk.CTkButton(self.sidebar_frame, text=self.i18n.get("start_asr", "Start ASR"), command=self.toggle_client)
        self.start_button.grid(row=1, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text=self.i18n.get("appearance_mode", "Appearance Mode:"), anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_optionemenu.set("Dark")

        self.language_label = ctk.CTkLabel(self.sidebar_frame, text=self.i18n.get("language", "Language:"), anchor="w")
        self.language_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.language_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["auto", "en", "zh_TW"],
                                                                command=self.change_language_event)
        self.language_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 10))
        self.language_optionemenu.set(self.config.get("language", "auto"))

        self.github_label = ctk.CTkLabel(self.sidebar_frame, text="GitHub", font=ctk.CTkFont(size=12, underline=True), cursor="hand2")
        self.github_label.grid(row=9, column=0, padx=20, pady=(10, 10))
        self.github_label.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/Saren-Arterius/WTAKO-ASR-IME"))

        # Main Content
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Status
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.grid(row=0, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.status_frame, text=self.i18n.get("status_disconnected", "Status: Disconnected"), font=ctk.CTkFont(size=16))
        self.status_label.grid(row=0, column=0, sticky="w")

        self.clear_button = ctk.CTkButton(self.status_frame, text=self.i18n.get("clear_log", "Clear Log"), width=80, command=self.clear_log)
        self.clear_button.grid(row=0, column=1, sticky="e")

        # Transcription Area
        self.textbox = ctk.CTkTextbox(self.main_frame, width=400, height=200)
        self.textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.textbox.insert("0.0", self.i18n.get("transcription_placeholder", "Transcriptions will appear here...\n"))
        self.textbox.configure(state="disabled")

        # Settings
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_frame.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.settings_frame.grid_columnconfigure(1, weight=1)

        self.use_custom_url = ctk.BooleanVar(value=not self.config.get("use_local_server", True))
        self.url_checkbox = ctk.CTkCheckBox(self.settings_frame, text=self.i18n.get("asr_server_url", "ASR Server URL"), variable=self.use_custom_url, command=self.toggle_url_entry)
        self.url_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.server_entry = ctk.CTkEntry(self.settings_frame)
        self.server_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        if not self.use_custom_url.get():
            self.server_entry.insert(0, self.i18n.get("local_server_placeholder", "Will start a local server automatically"))
            self.server_entry.configure(state="disabled", text_color="grey")
        else:
            self.server_entry.insert(0, self.config.get("default_asr_server", "http://localhost:8000"))
            self.server_entry.configure(state="normal")

        ctk.CTkLabel(self.settings_frame, text=self.i18n.get("input_device", "Input Device:")).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        devices = sd.query_devices()
        device_names = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        self.device_option = ctk.CTkOptionMenu(self.settings_frame, values=device_names, command=self.on_device_change)
        self.device_option.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.volume_label = ctk.CTkLabel(self.settings_frame, text=self.i18n.get("volume", "Volume:"))
        self.volume_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.volume_meter = ctk.CTkProgressBar(self.settings_frame)
        self.volume_meter.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.volume_meter.set(0)

        ctk.CTkLabel(self.settings_frame, text=self.i18n.get("hotkey", "Hotkey:")).grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        self.hotkey_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.hotkey_frame.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.hotkey_frame.grid_columnconfigure(0, weight=1)

        common_keys = ["f12", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "caps lock", "insert", "home", "page up", "page down", "end"]
        current_hotkey = self.config.get("hotkey", "f12")
        if current_hotkey not in common_keys:
            common_keys.append(current_hotkey)
        
        self.hotkey_option = ctk.CTkOptionMenu(self.hotkey_frame, values=common_keys)
        self.hotkey_option.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="ew")
        self.hotkey_option.set(current_hotkey)

        self.record_button = ctk.CTkButton(self.hotkey_frame, text=self.i18n.get("record", "Record"), width=60, command=self.start_recording_hotkey)
        self.record_button.grid(row=0, column=1, padx=0, pady=0)

        ctk.CTkLabel(self.settings_frame, text=self.i18n.get("asr_backend", "ASR Backend:")).grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.backend_option = ctk.CTkOptionMenu(self.settings_frame, values=["glm", "sherpa-onnx/sense-voice", "whisper"], command=self.on_backend_change)
        self.backend_option.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
        
        current_backend = self.config.get("asr_backend", "glm")
        if current_backend == "sensevoice":
            current_backend = "sherpa-onnx/sense-voice"
        self.backend_option.set(current_backend)

        self.system_prompt_label = ctk.CTkLabel(self.settings_frame, text=self.i18n.get("system_prompt", "System Prompt:"))
        self.system_prompt_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.system_prompt_entry = ctk.CTkEntry(self.settings_frame)
        self.system_prompt_entry.grid(row=5, column=1, padx=10, pady=5, sticky="ew")
        self.system_prompt_entry.insert(0, self.config.get("glm", {}).get("system_prompt", ""))

        # SenseVoice specific settings
        self.sensevoice_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.sensevoice_frame.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.sensevoice_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.sensevoice_frame, text=self.i18n.get("num_threads", "Threads:")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.threads_entry = ctk.CTkEntry(self.sensevoice_frame)
        self.threads_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.threads_entry.insert(0, str(self.config.get("sensevoice", {}).get("num_threads", 2)))

        ctk.CTkLabel(self.sensevoice_frame, text=self.i18n.get("sensevoice_language", "SenseVoice Language:")).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.sv_lang_option = ctk.CTkOptionMenu(self.sensevoice_frame, values=["auto", "zh", "en", "ja", "ko", "yue"])
        self.sv_lang_option.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.sv_lang_option.set(self.config.get("sensevoice", {}).get("language", "auto"))

        ctk.CTkLabel(self.sensevoice_frame, text=self.i18n.get("provider", "Provider:")).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.provider_option = ctk.CTkOptionMenu(self.sensevoice_frame, values=["cpu", "cuda", "coreml"])
        self.provider_option.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.provider_option.set(self.config.get("sensevoice", {}).get("provider", "cpu"))
        
        # Whisper specific settings
        self.whisper_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.whisper_frame.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.whisper_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.whisper_frame, text=self.i18n.get("whisper_device", "Device:")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.whisper_device_option = ctk.CTkOptionMenu(self.whisper_frame, values=["cuda", "cpu"])
        self.whisper_device_option.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.whisper_device_option.set(self.config.get("whisper", {}).get("device", "cuda"))

        ctk.CTkLabel(self.whisper_frame, text=self.i18n.get("whisper_language", "Language:")).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.whisper_lang_option = ctk.CTkOptionMenu(self.whisper_frame, values=["auto", "en", "zh", "ja", "ko", "yue"])
        self.whisper_lang_option.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.whisper_lang_option.set(self.config.get("whisper", {}).get("language", "auto"))

        ctk.CTkLabel(self.whisper_frame, text=self.i18n.get("whisper_task", "Task:")).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.whisper_task_option = ctk.CTkOptionMenu(self.whisper_frame, values=["transcribe", "translate"])
        self.whisper_task_option.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.whisper_task_option.set(self.config.get("whisper", {}).get("task", "transcribe"))

        if self.backend_option.get() == "glm":
            self.sensevoice_frame.grid_remove()
            self.whisper_frame.grid_remove()
        elif self.backend_option.get() == "whisper":
            self.system_prompt_label.grid_remove()
            self.system_prompt_entry.grid_remove()
            self.sensevoice_frame.grid_remove()
            self.whisper_frame.grid()
        else:
            self.system_prompt_label.grid_remove()
            self.system_prompt_entry.grid_remove()
            self.sensevoice_frame.grid()
            self.whisper_frame.grid_remove()

        self.disable_log_var = ctk.BooleanVar(value=self.config.get("disable_log", False))
        self.disable_log_checkbox = ctk.CTkCheckBox(self.settings_frame, text=self.i18n.get("disable_log", "Disable Transcribe Log"), variable=self.disable_log_var)
        self.disable_log_checkbox.grid(row=6, column=0, padx=10, pady=5, sticky="w")

        self.opencc_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.opencc_frame.grid(row=6, column=1, padx=10, pady=5, sticky="ew")
        self.opencc_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.opencc_frame, text=self.i18n.get("opencc_convert", "OpenCC Convert:")).grid(row=0, column=0, padx=(0, 10), pady=0, sticky="w")
        
        self.opencc_map = {
            self.i18n.get("opencc_none", "None"): None,
            self.i18n.get("opencc_s2t", "Simplified to Traditional"): "s2t",
            self.i18n.get("opencc_t2s", "Traditional to Simplified"): "t2s"
        }
        self.opencc_option = ctk.CTkOptionMenu(self.opencc_frame, values=list(self.opencc_map.keys()))
        self.opencc_option.grid(row=0, column=1, padx=0, pady=0, sticky="ew")
        
        current_opencc = self.config.get("opencc_convert")
        # Find the label for the current value
        current_label = self.i18n.get("opencc_none", "None")
        for label, val in self.opencc_map.items():
            if val == current_opencc:
                current_label = label
                break
        self.opencc_option.set(current_label)

        self.save_button = ctk.CTkButton(self.settings_frame, text=self.i18n.get("save_config", "Save Config"), command=self.save_config_event)
        self.save_button.grid(row=7, column=0, columnspan=2, padx=10, pady=10)
        
        # Try to select default from config
        default_device = self.find_default_device_index(self.config.get("audio_devices", []))
        if default_device is not None:
            for name in device_names:
                if name.startswith(f"{default_device}:"):
                    self.device_option.set(name)
                    break
        
        self.start_volume_monitor()

    def on_backend_change(self, backend):
        if self.use_custom_url.get():
            self.system_prompt_label.grid_remove()
            self.system_prompt_entry.grid_remove()
            self.sensevoice_frame.grid_remove()
            self.whisper_frame.grid_remove()
            return

        if backend == "glm":
            self.system_prompt_label.grid()
            self.system_prompt_entry.grid()
            self.sensevoice_frame.grid_remove()
            self.whisper_frame.grid_remove()
        elif backend == "whisper":
            self.system_prompt_label.grid_remove()
            self.system_prompt_entry.grid_remove()
            self.sensevoice_frame.grid_remove()
            self.whisper_frame.grid()
        else:
            self.system_prompt_label.grid_remove()
            self.system_prompt_entry.grid_remove()
            self.sensevoice_frame.grid()
            self.whisper_frame.grid_remove()

    def transition_to(self, new_state, error_msg=None):
        self.app_state = new_state
        
        def update_ui():
            # Disable/Enable settings based on running state
            is_running = new_state != AppState.DISCONNECTED and new_state != AppState.ERROR
            
            # Disallow changing settings when running
            state = "disabled" if is_running else "normal"
            
            self.url_checkbox.configure(state="disabled" if is_running else "normal")
            if not self.use_custom_url.get() or is_running:
                self.server_entry.configure(state="disabled")
            else:
                self.server_entry.configure(state="normal")
                
            self.device_option.configure(state="disabled" if is_running else "normal")
            self.hotkey_option.configure(state="disabled" if is_running else "normal")
            self.record_button.configure(state="disabled" if is_running else "normal")
            self.backend_option.configure(state="disabled" if is_running else "normal")
            
            # These settings can be changed while running
            self.system_prompt_entry.configure(state=state)
            self.threads_entry.configure(state=state)
            self.sv_lang_option.configure(state=state)
            self.provider_option.configure(state=state)
            self.whisper_device_option.configure(state=state)
            self.whisper_lang_option.configure(state=state)
            self.whisper_task_option.configure(state=state)
            self.opencc_option.configure(state=state)
            self.disable_log_checkbox.configure(state=state)
            self.save_button.configure(state=state)

            if new_state == AppState.DISCONNECTED:
                self.update_status(self.i18n.get("status_disconnected", "Disconnected").replace(self.i18n.get("status_prefix", "Status: "), ""))
                self.start_button.configure(text=self.i18n.get("start_asr", "Start ASR"), fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            elif new_state == AppState.STARTING:
                self.update_status(self.i18n.get("status_starting", "Starting..."), "orange")
                self.start_button.configure(text=self.i18n.get("stop_client", "Stop Client"), fg_color="red")
            elif new_state == AppState.WAITING_FOR_SERVER:
                self.update_status(self.i18n.get("status_waiting", "Waiting for ASR Server..."), "orange")
            elif new_state == AppState.LISTENING:
                self.update_status(self.i18n.get("status_listening", "Listening"), "green")
            elif new_state == AppState.RECORDING:
                self.update_status(self.i18n.get("status_recording", "Recording..."), "red")
            elif new_state == AppState.PROCESSING:
                self.update_status(self.i18n.get("status_processing", "Processing..."), "orange")
            elif new_state == AppState.ERROR:
                self.update_status(self.i18n.get("status_error", "Error: {error_msg}").format(error_msg=error_msg or 'Unknown error'), "red")
                self.start_button.configure(text=self.i18n.get("start_asr", "Start ASR"), fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

        self.after(0, update_ui)

    @property
    def is_running(self):
        return self.app_state != AppState.DISCONNECTED and self.app_state != AppState.ERROR

    def on_device_change(self, _):
        if not self.is_running:
            self.start_volume_monitor()

    def start_volume_monitor(self):
        # Use a lock to prevent multiple threads from starting/stopping the monitor simultaneously
        if not hasattr(self, "volume_monitor_lock"):
            self.volume_monitor_lock = threading.Lock()
            
        threading.Thread(target=self._start_volume_monitor_thread, daemon=True).start()

    def stop_volume_monitor(self):
        if not hasattr(self, "volume_monitor_lock"):
            self.volume_monitor_lock = threading.Lock()
            
        # Don't block the main thread waiting for the lock
        threading.Thread(target=self._stop_volume_monitor_thread, daemon=True).start()

    def _stop_volume_monitor_thread(self):
        with self.volume_monitor_lock:
            if self.volume_stream:
                try:
                    self.volume_stream.stop()
                    self.volume_stream.close()
                except:
                    pass
                self.volume_stream = None

    def _start_volume_monitor_thread(self):
        if self.is_running:
            return

        # Use a timeout or non-blocking acquire to avoid deadlocks
        acquired = self.volume_monitor_lock.acquire(timeout=2.0)
        if not acquired:
            print("Volume monitor lock acquisition timed out")
            return
            
        try:
            if self.volume_stream:
                try:
                    self.volume_stream.stop()
                    self.volume_stream.close()
                except:
                    pass
                self.volume_stream = None

            try:
                device_str = self.device_option.get()
                if not device_str or ":" not in device_str:
                    return
                device_id = int(device_str.split(":")[0])
                
                def audio_callback(indata, frames, time, status):
                    if not self.volume_stream:
                        return
                    self.update_volume_meter(indata)

                # Use a small blocksize to avoid blocking
                self.volume_stream = sd.InputStream(device=device_id, channels=1, callback=audio_callback, blocksize=1024)
                self.volume_stream.start()
            except Exception as e:
                print(f"Error starting volume monitor: {e}")
                self.volume_stream = None
        finally:
            self.volume_monitor_lock.release()

    def update_volume_meter(self, indata):
        volume_norm = np.linalg.norm(indata) * 10
        # Scale to 0-1 for progress bar
        level = min(1.0, volume_norm / 100)
        self.after(0, lambda: self.volume_meter.set(level))

    def find_default_device_index(self, name_substrings):
        devices = sd.query_devices()
        for sub in name_substrings:
            for i, dev in enumerate(devices):
                if sub.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                    return i
        return None

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_language_event(self, new_language: str):
        self.config["language"] = new_language
        # Reload UI with new language
        self.i18n = get_i18n(self.config)
        self.update_ui_texts()

    def update_ui_texts(self):
        self.title(self.i18n.get("title", "WTAKO ASR IME"))
        self.start_button.configure(text=self.i18n.get("start_asr", "Start ASR") if not self.is_running else self.i18n.get("stop_client", "Stop Client"))
        self.appearance_mode_label.configure(text=self.i18n.get("appearance_mode", "Appearance Mode:"))
        self.language_label.configure(text=self.i18n.get("language", "Language:"))
        self.clear_button.configure(text=self.i18n.get("clear_log", "Clear Log"))
        self.url_checkbox.configure(text=self.i18n.get("asr_server_url", "ASR Server URL"))
        
        # Update server entry placeholder if it's currently showing the placeholder
        if not self.use_custom_url.get():
            self.server_entry.configure(state="normal")
            self.server_entry.delete(0, "end")
            self.server_entry.insert(0, self.i18n.get("local_server_placeholder", "Will start a local server automatically"))
            self.server_entry.configure(state="disabled")
        else:
            # Update server entry with current config value if it was showing placeholder
            if self.server_entry.get() == self.i18n.get("local_server_placeholder", "Will start a local server automatically"):
                self.server_entry.configure(state="normal")
                self.server_entry.delete(0, "end")
                self.server_entry.insert(0, self.config.get("default_asr_server", "http://localhost:8000"))

        # Update labels that were missed
        for widget in self.settings_frame.winfo_children():
            if isinstance(widget, ctk.CTkLabel):
                text = widget.cget("text")
                if text == "Input Device:" or text == self.i18n.get("input_device", "Input Device:"): # Fallback check
                     widget.configure(text=self.i18n.get("input_device", "Input Device:"))
                elif text == "Hotkey:" or text == self.i18n.get("hotkey", "Hotkey:"):
                     widget.configure(text=self.i18n.get("hotkey", "Hotkey:"))
                elif text == "System Prompt:" or text == self.i18n.get("system_prompt", "System Prompt:"):
                     widget.configure(text=self.i18n.get("system_prompt", "System Prompt:"))

        self.volume_label.configure(text=self.i18n.get("volume", "Volume:"))
        self.record_button.configure(text=self.i18n.get("record", "Record"))
        self.disable_log_checkbox.configure(text=self.i18n.get("disable_log", "Disable Transcribe Log"))
        
        # Update OpenCC options and label
        self.opencc_map = {
            self.i18n.get("opencc_none", "None"): None,
            self.i18n.get("opencc_s2t", "Simplified to Traditional"): "s2t",
            self.i18n.get("opencc_t2s", "Traditional to Simplified"): "t2s"
        }
        current_val = self.config.get("opencc_convert")
        self.opencc_option.configure(values=list(self.opencc_map.keys()))
        
        current_label = self.i18n.get("opencc_none", "None")
        for label, val in self.opencc_map.items():
            if val == current_val:
                current_label = label
                break
        self.opencc_option.set(current_label)

        # Update OpenCC label in the frame
        for widget in self.opencc_frame.winfo_children():
            if isinstance(widget, ctk.CTkLabel):
                widget.configure(text=self.i18n.get("opencc_convert", "OpenCC Convert:"))

        self.save_button.configure(text=self.i18n.get("save_config", "Save Config"))
        
        # Update status label with current state
        self.transition_to(self.app_state)

    def toggle_url_entry(self):
        if self.use_custom_url.get():
            self.server_entry.configure(state="normal", text_color=ctk.ThemeManager.theme["CTkEntry"]["text_color"])
            # Restore previous value if it was the placeholder
            if self.server_entry.get() == self.i18n.get("local_server_placeholder", "Will start a local server automatically"):
                self.server_entry.delete(0, "end")
                self.server_entry.insert(0, self.config.get("default_asr_server", "http://localhost:8000"))
            
            # Disable backend settings
            self.backend_option.configure(state="disabled")
            self.system_prompt_label.grid_remove()
            self.system_prompt_entry.grid_remove()
            self.sensevoice_frame.grid_remove()
            self.whisper_frame.grid_remove()
        else:
            self.server_entry.delete(0, "end")
            self.server_entry.insert(0, self.i18n.get("local_server_placeholder", "Will start a local server automatically"))
            self.server_entry.configure(state="disabled", text_color="grey")
            
            # Enable backend settings
            self.backend_option.configure(state="normal")
            self.on_backend_change(self.backend_option.get())

    def update_status(self, status, color=None):
        prefix = self.i18n.get("status_prefix", "Status: ")
        if status.startswith(prefix):
            self.status_label.configure(text=status)
        else:
            self.status_label.configure(text=f"{prefix}{status}")
        if color:
            self.status_label.configure(text_color=color)
        else:
            self.status_label.configure(text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])

    def log_transcription(self, text):
        if self.disable_log_var.get() and "Config saved" not in text:
            return
        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"> {text}\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def clear_log(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", self.i18n.get("transcription_placeholder", "Transcriptions will appear here...\n"))
        self.textbox.configure(state="disabled")

    def start_recording_hotkey(self):
        self.record_button.configure(text="...", fg_color="orange")
        self.bind("<Key>", self.on_key_pressed)
        self.focus_set()

    def on_key_pressed(self, event):
        # Unbind immediately
        self.unbind("<Key>")
        
        # Map tkinter keysym to keyboard library names if necessary
        key = event.keysym.lower()
        
        # Common mappings
        mapping = {
            "caps_lock": "caps lock",
            "next": "page down",
            "prior": "page up",
            "return": "enter",
            "control_l": "ctrl",
            "control_r": "ctrl",
            "alt_l": "alt",
            "alt_r": "alt",
            "shift_l": "shift",
            "shift_r": "shift",
        }
        key = mapping.get(key, key)
        
        # Update option menu
        current_values = self.hotkey_option.cget("values")
        if key not in current_values:
            self.hotkey_option.configure(values=list(current_values) + [key])
        
        self.hotkey_option.set(key)
        self.record_button.configure(text=self.i18n.get("record", "Record"), fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        print(f"Recorded hotkey: {key}")

    def toggle_client(self):
        if not self.is_running:
            self.start_client()
        else:
            self.stop_client()

    def get_current_ui_settings(self):
        """Returns a dictionary of current settings from UI without modifying global CONFIG."""
        backend = self.backend_option.get()
        print(backend)
        settings = {
            "asr_backend": backend,
            "disable_log": self.disable_log_var.get(),
            "opencc_convert": self.opencc_map.get(self.opencc_option.get())
        }
        
        if backend == "glm":
            settings["glm"] = {"system_prompt": self.system_prompt_entry.get()}
        elif backend == "whisper":
            settings["whisper"] = {
                "device": self.whisper_device_option.get(),
                "language": self.whisper_lang_option.get(),
                "task": self.whisper_task_option.get()
            }
        else: # sensevoice
            try:
                threads = int(self.threads_entry.get())
            except:
                threads = 2
            settings["sensevoice"] = {
                "num_threads": threads,
                "language": self.sv_lang_option.get(),
                "provider": self.provider_option.get()
            }
        return settings

    def update_config_from_ui(self):
        ui_settings = self.get_current_ui_settings()
        
        # Deep update to avoid losing settings not in UI (e.g. model_dir)
        for k, v in ui_settings.items():
            if isinstance(v, dict) and k in self.config and isinstance(self.config[k], dict):
                self.config[k].update(v)
            else:
                self.config[k] = v
        
        # Handle non-backend specific settings
        use_custom = self.use_custom_url.get()
        self.config["use_local_server"] = not use_custom
        self.config["hotkey"] = self.hotkey_option.get()
        if use_custom:
            self.config["default_asr_server"] = self.server_entry.get()

    def save_config_event(self):
        self.update_config_from_ui()
        save_config(self.config)
        self.log_transcription(self.i18n.get("config_saved", "Config saved."))
        print("Config saved.")

    def start_client(self):
        if not self.use_custom_url.get():
            server_url = self.config.get("default_asr_server", "http://localhost:8000")
        else:
            server_url = self.server_entry.get()
            
        device_str = self.device_option.get()
        if not device_str or ":" not in device_str:
            self.transition_to(AppState.ERROR, self.i18n.get("no_device_selected", "No device selected"))
            return
        
        try:
            device_id = int(device_str.split(":")[0])
        except ValueError:
            self.transition_to(AppState.ERROR, self.i18n.get("invalid_device_id", "Invalid device ID"))
            return

        hotkey = self.hotkey_option.get()
        backend = self.backend_option.get()

        # Update config in memory
        self.config["hotkey"] = hotkey
        self.config["asr_backend"] = backend
        self.config["default_asr_server"] = server_url
        
        if "glm" not in self.config:
            self.config["glm"] = {}
        self.config["glm"]["system_prompt"] = self.system_prompt_entry.get()

        if "sensevoice" not in self.config:
            self.config["sensevoice"] = {}
        try:
            self.config["sensevoice"]["num_threads"] = int(self.threads_entry.get())
        except:
            pass
        self.config["sensevoice"]["language"] = self.sv_lang_option.get()
        self.config["sensevoice"]["provider"] = self.provider_option.get()

        if "whisper" not in self.config:
            self.config["whisper"] = {}
        self.config["whisper"]["device"] = self.whisper_device_option.get()
        self.config["whisper"]["language"] = self.whisper_lang_option.get()
        self.config["whisper"]["task"] = self.whisper_task_option.get()

        self.transition_to(AppState.STARTING)

        self.client_thread = threading.Thread(target=self.run_client, args=(server_url, device_id), daemon=True)
        self.client_thread.start()

    def run_client(self, server_url, device_id):
        try:
            self.client = ASRClient(server_url, config=self.config)
            # Override device from UI
            try:
                dev_info = sd.query_devices(device_id)
                self.client.input_device = device_id
                self.client.input_sample_rate = int(dev_info['default_samplerate'])
                self.client.input_channels = dev_info['max_input_channels']
            except Exception as e:
                self.transition_to(AppState.ERROR, self.i18n.get("invalid_device_error", "Invalid device - {e}").format(e=e))
                self.after(0, self.stop_client)
                return

            # Monkey patch the client to update UI
            original_send_to_asr = self.client.send_to_asr
            def patched_send_to_asr(audio_data, sample_rate):
                self.transition_to(AppState.PROCESSING)
                
                # Get current UI settings to "try" them without saving to file
                ui_settings = self.get_current_ui_settings()
                
                # Update self.config for the duration of this request
                # ASRClient uses self.config
                # Deep update to avoid losing settings not in UI (e.g. model_dir)
                for k, v in ui_settings.items():
                    if isinstance(v, dict) and k in self.config and isinstance(self.config[k], dict):
                        self.config[k].update(v)
                    else:
                        self.config[k] = v
                
                try:
                    res = original_send_to_asr(audio_data, sample_rate)
                finally:
                    pass

                if res:
                    self.after(0, lambda r=res: self.log_transcription(r))
                self.transition_to(AppState.LISTENING)
                return res
            
            self.client.send_to_asr = patched_send_to_asr

            # Patch audio_callback to update volume meter
            original_audio_callback = self.client.audio_callback
            def patched_audio_callback(indata, frames, time_info, status):
                self.update_volume_meter(indata)
                original_audio_callback(indata, frames, time_info, status)
            
            self.client.audio_callback = patched_audio_callback

            # Stop the standalone volume monitor before starting ASR stream
            self.stop_volume_monitor()

            # Start local server if needed
            if self.config.get("use_local_server", True):
                self.local_server_proc = self.client.start_local_server()
            
            # Start threads
            threading.Thread(target=self.client.socket_listener, daemon=True).start()
            threading.Thread(target=self.client.recording_loop, daemon=True).start()
            
            # Start keyboard listener
            self.client.start_keyboard_subprocess()
            
            # Wait for server to be ready
            self.transition_to(AppState.WAITING_FOR_SERVER)
            
            while self.is_running and self.client and not self.client.stop_event.is_set():
                if self.client.check_server_ready():
                    break
                time.sleep(1)
            
            if not self.is_running or not self.client or self.client.stop_event.is_set():
                return

            self.transition_to(AppState.LISTENING)

            # Start audio input stream
            CHUNK_SIZE = int(self.client.input_sample_rate * 32 / 1000)
            try:
                stream = sd.InputStream(device=self.client.input_device, 
                                    samplerate=self.client.input_sample_rate, 
                                    channels=self.client.input_channels, 
                                    callback=self.client.audio_callback, 
                                    blocksize=CHUNK_SIZE)
            except Exception as e:
                self.transition_to(AppState.ERROR, self.i18n.get("audio_error", "Audio Error: {e}").format(e=e))
                return

            with stream:
                while self.is_running and self.client and not self.client.stop_event.is_set():
                    # Check if recording is active to update UI status
                    if self.client.is_recording_dict.get("internal_active"):
                        if self.app_state != AppState.RECORDING:
                            self.transition_to(AppState.RECORDING)
                    elif self.is_running:
                        # Only set to listening if we weren't just processing
                        if self.app_state == AppState.RECORDING:
                             self.transition_to(AppState.LISTENING)
                    
                    time.sleep(0.1)
        except Exception as e:
            print(f"Client error: {e}")
            self.transition_to(AppState.ERROR, str(e))

    def stop_client(self):
        if self.client:
            self.client.stop()
            self.client = None
        self.transition_to(AppState.DISCONNECTED)
        self.start_volume_monitor()

    def on_closing(self):
        print("Exiting application...")
        # Set app_state to DISCONNECTED first to stop any loops
        self.app_state = AppState.DISCONNECTED
        
        # Force exit after a short timeout to prevent hanging
        def force_exit():
            time.sleep(1.0)
            print("Force exiting...")
            os._exit(0)
        
        threading.Thread(target=force_exit, daemon=True).start()
        
        # Try to stop client without starting volume monitor
        if self.client:
            try:
                self.client.stop()
            except Exception as e:
                print(f"Error stopping client: {e}")
        
        if self.volume_stream:
            try:
                self.volume_stream.stop()
                self.volume_stream.close()
            except Exception as e:
                print(f"Error stopping volume stream: {e}")
        
        # Destroy window
        try:
            self.destroy()
        except Exception as e:
            print(f"Error destroying window: {e}")
            
        # Force exit to ensure all threads and subprocesses are killed
        os._exit(0)

if __name__ == "__main__":
    app = ASRGui()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
