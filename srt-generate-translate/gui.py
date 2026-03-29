import sys
import os
import json
import threading
import time
import tkinter as tk
from tkinter import filedialog
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False
import customtkinter as ctk
from PIL import Image
import srt
import torch
import io

CONFIG_FILE = "srt_config.json"

DEFAULT_CONFIG = {
    "ASR_API_URL": "http://100.64.0.8:8003/v1",
    "ASR_MODEL": "Qwen/Qwen3-ASR-1.7B",
    "ASR_BACKEND": "openai",
    "SOURCE_LANG": "ja",
    "DEST_LANG": "Traditional Chinese (Taiwan/Hong Kong style)",
    "LLM_API_URL": "http://100.64.0.8:8000/v1",
    "LLM_MODEL": "Intel/Qwen3-Coder-Next-int4-AutoRound",
    "merge_duration": 0.5,
    "merge_duration_force": 0.2,
    "min_speakers": 0,
    "HF_TOKEN": "",
    "use_diarization": True,
    "appearance_mode": "Dark",
    "video_files": [],
    "context": "",
    "unload_models_after_use": False
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SRTGui(ctk.CTk, TkinterDnD.DnDWrapper if HAS_DND else object):
    def __init__(self):
        super().__init__()
        if HAS_DND:
            self.TkdndVersion = TkinterDnD._require(self)
        self.config = load_config()
        ctk.set_appearance_mode(self.config.get("appearance_mode", "Dark"))

        self.title("WTAKO SRT Generator & Translator")
        self.geometry("1000x800")

        self.video_files = self.config.get("video_files", [])
        self.is_processing = False
        self.stop_requested = False
        self.chunk_stats = {}  # Store stats per chunk for combined TPS

        self.setup_ui()
        self.load_ui_values()
        self.refresh_video_list()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, text="SRT Tool", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.add_btn = ctk.CTkButton(
            self.sidebar_frame, text="Add Videos", command=self.browse_videos)
        self.add_btn.grid(row=1, column=0, padx=20, pady=10)

        self.add_folder_btn = ctk.CTkButton(
            self.sidebar_frame, text="Add Folder (Rec)", command=self.browse_folder)
        self.add_folder_btn.grid(row=2, column=0, padx=20, pady=10)

        self.clear_list_btn = ctk.CTkButton(
            self.sidebar_frame, text="Clear List", fg_color="transparent", border_width=1, command=self.clear_videos)
        self.clear_list_btn.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(
            self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                             command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(
            row=6, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set(
            self.config.get("appearance_mode", "Dark"))

        # Main Content
        self.main_frame = ctk.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Video List
        self.list_label = ctk.CTkLabel(
            self.main_frame, text="Video Queue:", font=ctk.CTkFont(size=16, weight="bold"))
        self.list_label.grid(row=0, column=0, sticky="w", padx=10)

        self.video_listbox = ctk.CTkTextbox(
            self.main_frame, height=150)
        self.video_listbox.grid(
            row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.video_listbox.configure(state="disabled")

        if HAS_DND:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind('<<Drop>>', self.handle_drop)
            except Exception as e:
                print(f"Failed to register window for DnD: {e}")

        # Context
        self.context_label = ctk.CTkLabel(
            self.main_frame, text="Translation Context (Optional):")
        self.context_label.grid(row=2, column=0, sticky="w", padx=10)
        self.context_entry = ctk.CTkEntry(
            self.main_frame, placeholder_text="e.g. Anime localization, specific terminology...")
        self.context_entry.grid(row=3, column=0, padx=10,
                                pady=(0, 10), sticky="ew")

        # Config Section
        self.settings_tab = ctk.CTkTabview(self.main_frame, height=300)
        self.settings_tab.grid(row=4, column=0, padx=10,
                               pady=10, sticky="nsew")
        self.settings_tab.add("Settings")
        self.settings_tab.add("Advanced")

        # Settings Tab
        settings_tab = self.settings_tab.tab("Settings")
        settings_tab.grid_columnconfigure(1, weight=1)

        self.asr_url_label = ctk.CTkLabel(settings_tab, text="ASR URL:")
        self.asr_url_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.asr_url = ctk.CTkEntry(settings_tab)
        self.asr_url.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        self.asr_model_label = ctk.CTkLabel(settings_tab, text="ASR Model:")
        self.asr_model_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.asr_model = ctk.CTkEntry(settings_tab)
        self.asr_model.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(settings_tab, text="ASR Backend:").grid(
            row=2, column=0, padx=10, pady=5, sticky="w")
        self.asr_backend = ctk.CTkOptionMenu(
            settings_tab, values=["openai", "sensevoice", "qwen_asr"], command=self.update_asr_visibility)
        self.asr_backend.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(settings_tab, text="Source Lang:").grid(
            row=3, column=0, padx=10, pady=5, sticky="w")
        self.source_lang = ctk.CTkOptionMenu(
            settings_tab, values=["ja", "zh", "en", "ko", "yue", "auto"], command=self.on_source_lang_change)
        self.source_lang.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        self.custom_lang_label = ctk.CTkLabel(
            settings_tab, text="Custom Lang:")
        self.custom_lang = ctk.CTkEntry(settings_tab)

        ctk.CTkLabel(settings_tab, text="Dest Lang:").grid(
            row=4, column=0, padx=10, pady=5, sticky="w")
        self.dest_lang = ctk.CTkEntry(settings_tab)
        self.dest_lang.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(settings_tab, text="LLM URL:").grid(
            row=5, column=0, padx=10, pady=5, sticky="w")
        self.llm_url = ctk.CTkEntry(settings_tab)
        self.llm_url.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(settings_tab, text="LLM Model:").grid(
            row=6, column=0, padx=10, pady=5, sticky="w")
        self.llm_model = ctk.CTkEntry(settings_tab)
        self.llm_model.grid(row=6, column=1, padx=10, pady=5, sticky="ew")

        self.use_diarization_var = ctk.BooleanVar(
            value=self.config.get("use_diarization", True))
        self.diarization_checkbox = ctk.CTkCheckBox(settings_tab, text="Use Speaker Diarization (Recommended)",
                                                    variable=self.use_diarization_var)
        self.diarization_checkbox.grid(
            row=7, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        ctk.CTkLabel(settings_tab, text="HF Token:").grid(
            row=8, column=0, padx=10, pady=5, sticky="w")
        self.hf_token = ctk.CTkEntry(
            settings_tab, show="*")
        self.hf_token.grid(row=8, column=1, padx=10, pady=5, sticky="ew")

        self.unload_models_var = ctk.BooleanVar(
            value=self.config.get("unload_models_after_use", False))
        self.unload_checkbox = ctk.CTkCheckBox(settings_tab, text="Unload Models After Use (Save VRAM)",
                                               variable=self.unload_models_var)
        self.unload_checkbox.grid(
            row=9, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        # Advanced Tab
        adv_tab = self.settings_tab.tab("Advanced")
        adv_tab.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(adv_tab, text="Merge Duration (s):").grid(
            row=0, column=0, padx=10, pady=5, sticky="w")
        self.merge_duration = ctk.CTkEntry(adv_tab)
        self.merge_duration.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(adv_tab, text="Force Merge Duration (s):").grid(
            row=1, column=0, padx=10, pady=5, sticky="w")
        self.merge_duration_force = ctk.CTkEntry(adv_tab)
        self.merge_duration_force.grid(
            row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(adv_tab, text="Min Speakers:").grid(
            row=2, column=0, padx=10, pady=5, sticky="w")
        self.min_speakers = ctk.CTkEntry(adv_tab)
        self.min_speakers.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(adv_tab, text="Prompt Preview:").grid(
            row=3, column=0, padx=10, pady=5, sticky="nw")
        self.prompt_preview = ctk.CTkTextbox(adv_tab, height=150)
        self.prompt_preview.grid(
            row=3, column=1, padx=10, pady=5, sticky="nsew")

        self.update_preview_btn = ctk.CTkButton(
            adv_tab, text="Update Preview", command=self.update_prompt_preview)
        self.update_preview_btn.grid(
            row=4, column=1, padx=10, pady=5, sticky="e")

        # Status & Progress
        self.status_label = ctk.CTkLabel(
            self.main_frame, text="Status: Ready")
        self.status_label.grid(row=5, column=0, pady=(10, 0))

        self.tps_label = ctk.CTkLabel(
            self.main_frame, text="", font=ctk.CTkFont(size=12, slant="italic"))
        self.tps_label.grid(row=5, column=0, pady=(10, 0), sticky="e")

        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.grid(row=6, column=0, padx=20,
                               pady=(10, 5), sticky="ew")
        self.progress_bar.set(0)

        self.sub_progress_bar = ctk.CTkProgressBar(self.main_frame, height=8)
        self.sub_progress_bar.grid(
            row=7, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.sub_progress_bar.set(0)

        # Action Buttons
        self.action_frame = ctk.CTkFrame(
            self.main_frame, fg_color="transparent")
        self.action_frame.grid(row=8, column=0, pady=10)

        self.run_btn = ctk.CTkButton(self.action_frame, text="Start Processing", command=self.toggle_processing,
                                     width=200, height=40, font=ctk.CTkFont(size=15, weight="bold"))
        self.run_btn.pack(side=tk.LEFT, padx=10)

        self.save_btn = ctk.CTkButton(
            self.action_frame, text="Save Config", command=self.save_ui_config, width=100)
        self.save_btn.pack(side=tk.LEFT, padx=10)

    def load_ui_values(self):
        self.asr_url.insert(0, self.config["ASR_API_URL"])
        self.asr_model.insert(0, self.config["ASR_MODEL"])
        self.asr_backend.set(self.config.get("ASR_BACKEND", "openai"))
        self.source_lang.set(self.config.get("SOURCE_LANG", "ja"))
        self.dest_lang.insert(0, self.config.get(
            "DEST_LANG", "Traditional Chinese (Taiwan/Hong Kong style)"))
        self.llm_url.insert(0, self.config["LLM_API_URL"])
        self.llm_model.insert(0, self.config["LLM_MODEL"])
        self.merge_duration.insert(0, str(self.config["merge_duration"]))
        self.merge_duration_force.insert(
            0, str(self.config.get("merge_duration_force", 0.2)))
        self.min_speakers.insert(0, str(self.config.get("min_speakers", 0)))
        self.hf_token.insert(0, self.config["HF_TOKEN"])
        self.context_entry.insert(0, self.config.get("context", ""))
        self.update_asr_visibility(self.asr_backend.get())
        self.update_prompt_preview()

    def update_asr_visibility(self, backend):
        # Update source language options based on backend
        common_langs = ["ja", "zh", "en", "ko", "yue", "auto"]
        qwen_extra = [
            "Arabic", "German", "French", "Spanish", "Portuguese",
            "Indonesian", "Italian", "Russian", "Thai", "Vietnamese",
            "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish",
            "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
            "Romanian", "Hungarian", "Macedonian"
        ]

        if backend == "qwen_asr":
            self.source_lang.configure(values=common_langs + qwen_extra)
        elif backend == "openai":
            self.source_lang.configure(
                values=common_langs + qwen_extra + ["Custom"])
        else:
            self.source_lang.configure(values=common_langs)

        if backend == "openai":
            self.asr_url_label.grid()
            self.asr_url.grid()
            self.asr_model_label.grid()
            self.asr_model.grid()
        elif backend == "qwen_asr":
            self.asr_url_label.grid_remove()
            self.asr_url.grid_remove()
            self.asr_model_label.grid()
            self.asr_model.grid()
        else:
            self.asr_url_label.grid_remove()
            self.asr_url.grid_remove()
            self.asr_model_label.grid_remove()
            self.asr_model.grid_remove()

        self.on_source_lang_change(self.source_lang.get())

    def on_source_lang_change(self, lang):
        if lang == "Custom" and self.asr_backend.get() == "openai":
            self.custom_lang_label.grid(row=3, column=2, padx=10, pady=5)
            self.custom_lang.grid(row=3, column=3, padx=10, pady=5)
        else:
            self.custom_lang_label.grid_remove()
            self.custom_lang.grid_remove()
        self.update_prompt_preview()

    def update_prompt_preview(self):
        source = self.source_lang.get()
        dest = self.dest_lang.get()
        context = self.context_entry.get()
        sample_content = "1\n00:00:01,000 --> 00:00:04,000\nこんにちは、元気ですか？"
        prompt = srt.get_translation_prompt(
            source, dest, sample_content, context)
        self.prompt_preview.configure(state="normal")
        self.prompt_preview.delete("0.0", "end")
        self.prompt_preview.insert("0.0", prompt)
        self.prompt_preview.configure(state="disabled")

    def save_ui_config(self):
        try:
            self.config.update({
                "ASR_API_URL": self.asr_url.get(),
                "ASR_MODEL": self.asr_model.get(),
                "ASR_BACKEND": self.asr_backend.get(),
                "SOURCE_LANG": self.source_lang.get(),
                "DEST_LANG": self.dest_lang.get(),
                "LLM_API_URL": self.llm_url.get(),
                "LLM_MODEL": self.llm_model.get(),
                "merge_duration": float(self.merge_duration.get()),
                "merge_duration_force": float(self.merge_duration_force.get()),
                "min_speakers": int(self.min_speakers.get()),
                "HF_TOKEN": self.hf_token.get(),
                "use_diarization": self.use_diarization_var.get(),
                "appearance_mode": self.appearance_mode_optionemenu.get(),
                "video_files": self.video_files,
                "context": self.context_entry.get(),
                "unload_models_after_use": self.unload_models_var.get()
            })
            save_config(self.config)
            self.update_status("Configuration saved", "green")
            self.update_prompt_preview()
        except ValueError:
            self.update_status("Error: Invalid merge duration", "red")

    def change_appearance_mode_event(self, new_mode):
        ctk.set_appearance_mode(new_mode)
        self.config["appearance_mode"] = new_mode
        save_config(self.config)

    def browse_videos(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*")])
        if files:
            for f in files:
                if f not in self.video_files:
                    self.video_files.append(f)
            self.save_ui_config()
            self.refresh_video_list()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            video_extensions = ('.mp4', '.mkv', '.avi',
                                '.mov', '.wmv', '.flv', '.webm')
            added_count = 0
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        full_path = os.path.join(root, file)
                        if full_path not in self.video_files:
                            self.video_files.append(full_path)
                            added_count += 1
            if added_count > 0:
                self.save_ui_config()
                self.refresh_video_list()
                self.update_status(
                    f"Added {added_count} videos from folder", "green")

    def clear_videos(self):
        self.video_files = []
        self.save_ui_config()
        self.refresh_video_list()

    def handle_drop(self, event):
        if not event.data:
            return
        # Use tk's internal splitlist to reliably parse filenames with spaces
        files = self.tk.splitlist(event.data)
        video_extensions = ('.mp4', '.mkv', '.avi', '.mov',
                            '.wmv', '.flv', '.webm')
        added_count = 0
        for f in files:
            if f.lower().endswith(video_extensions):
                if f not in self.video_files:
                    self.video_files.append(f)
                    added_count += 1
        if added_count > 0:
            self.save_ui_config()
            self.refresh_video_list()
            self.update_status(f"Dropped {added_count} videos", "green")

    def refresh_video_list(self):
        self.video_listbox.configure(state="normal")
        self.video_listbox.delete("0.0", "end")
        if not self.video_files:
            self.video_listbox.insert("0.0", "No videos selected.")
        else:
            for i, f in enumerate(self.video_files):
                self.video_listbox.insert(
                    "end", f"{i+1}. {os.path.basename(f)}\n")
        self.video_listbox.configure(state="disabled")

    def update_status(self, text, color=None):
        self.status_label.configure(text=f"Status: {text}")
        if color:
            self.status_label.configure(text_color=color)
        else:
            self.status_label.configure(
                text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])

    def update_transcription_progress(self, completed, total):
        progress = (completed / total) if total > 0 else 0
        # ASR is 10% to 20% (10% total)
        sub_progress = 0.1 + (progress * 0.1)
        self.after(0, lambda: [
            self.tps_label.configure(text=f"Fragment {completed}/{total}"),
            self.sub_progress_bar.set(sub_progress)
        ])

    def update_diarization_progress(self, step_name, completed, total):
        progress = (completed / total) if total > 0 else 0
        # Diarization is 5% to 10% (5% total)
        sub_progress = 0.05 + (progress * 0.05)
        self.after(0, lambda: [
            self.tps_label.configure(text=f"{step_name}: {completed}/{total}"),
            self.sub_progress_bar.set(sub_progress)
        ])

    def update_tps(self, chunk_id, tokens, duration, translated_blocks, total_blocks, chunk_total_blocks):
        # Ensure we don't exceed the chunk's block count due to rough estimation
        translated_blocks = min(translated_blocks, chunk_total_blocks)
        self.chunk_stats[chunk_id] = (tokens, duration, translated_blocks)

        total_tokens = sum(s[0] for s in self.chunk_stats.values())
        total_translated_blocks = sum(s[2] for s in self.chunk_stats.values())

        max_duration = max((s[1]
                            for s in self.chunk_stats.values()), default=0)

        if max_duration > 0:
            combined_tps = total_tokens / max_duration
            progress = (total_translated_blocks /
                        total_blocks) if total_blocks > 0 else 0
            # Translation is 20% to 100% (80% total)
            sub_progress = 0.2 + (progress * 0.8)
            self.after(0, lambda: [
                self.tps_label.configure(
                    text=f"{progress*100:.1f}% | {combined_tps:.1f} tokens/s"),
                self.sub_progress_bar.set(sub_progress)
            ])

    def toggle_processing(self):
        if self.is_processing:
            self.request_stop()
        else:
            self.start_processing()

    def request_stop(self):
        if self.is_processing:
            self.stop_requested = True
            self.update_status("Stopping after current task...", "orange")
            self.run_btn.configure(state="disabled")

    def start_processing(self):
        if not self.video_files:
            self.update_status("Error: No videos selected", "red")
            return

        self.save_ui_config()

        if self.config["use_diarization"] and not self.config["HF_TOKEN"]:
            self.update_status(
                "Error: HF Token required for Diarization", "red")
            tk.messagebox.showerror(
                "Error", "Speaker Diarization requires a Hugging Face Token.\nPlease provide one in the Settings tab or uncheck Diarization.")
            return

        self.is_processing = True
        self.stop_requested = False
        self.run_btn.configure(text="Stop Processing",
                               fg_color="red", hover_color="#AA0000")
        self.add_btn.configure(state="disabled")
        self.add_folder_btn.configure(state="disabled")
        self.clear_list_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.sub_progress_bar.set(0)

        threading.Thread(target=self.process_queue, args=(
            self.context_entry.get(),), daemon=True).start()

    def process_queue(self, context):
        total = len(self.video_files)
        try:
            # Update srt module globals
            srt.ASR_API_URL = self.config["ASR_API_URL"]
            srt.ASR_MODEL = self.config["ASR_MODEL"]
            srt.ASR_BACKEND = self.config.get("ASR_BACKEND", "openai")

            source_lang = self.config.get("SOURCE_LANG", "ja")
            if source_lang == "Custom" and self.config.get("ASR_BACKEND") == "openai":
                source_lang = self.custom_lang.get()
            srt.SOURCE_LANG = source_lang

            srt.DEST_LANG = self.config.get(
                "DEST_LANG", "Traditional Chinese (Taiwan/Hong Kong style)")
            srt.LLM_API_URL = self.config["LLM_API_URL"]
            srt.LLM_MODEL = self.config["LLM_MODEL"]
            srt.MERGE_DURATION = self.config["merge_duration"]
            srt.MERGE_DURATION_FORCE = self.config.get(
                "merge_duration_force", 0.2)
            if self.config["HF_TOKEN"]:
                os.environ["HF_TOKEN"] = self.config["HF_TOKEN"]

            for i, video_file in enumerate(self.video_files):
                if self.stop_requested:
                    self.update_status("Stopped by user", "orange")
                    break

                base_name = os.path.basename(video_file)
                self.update_status(f"Processing {i+1}/{total}: {base_name}")

                base_path = os.path.splitext(video_file)[0]
                source_srt = f"{base_path}.{srt.SOURCE_LANG}.srt"
                dest_srt = f"{base_path}.translated.srt"

                # 1. Extract
                self.update_status(f"[{i+1}/{total}] Extracting audio...")
                self.after(0, lambda: self.sub_progress_bar.set(0.02))
                audio_bytes = srt.extract_audio(video_file)

                # 2. Diarization / VAD
                timestamps = None
                if self.config["use_diarization"]:
                    self.update_status(
                        f"[{i+1}/{total}] Running Diarization...")
                    self.after(0, lambda: self.sub_progress_bar.set(0.05))
                    # run_diarization_community now returns (merged, raw)
                    diarization_result = srt.run_diarization_community(
                        audio_bytes,
                        min_speakers=self.config.get("min_speakers"),
                        stats_callback=self.update_diarization_progress
                    )
                    if diarization_result:
                        timestamps, raw_timestamps = diarization_result
                        # Save raw debug diarization SRT before merging
                        diarization_srt = f"{base_path}.debug.diarization.srt"
                        with open(diarization_srt, 'w', encoding='utf-8') as f:
                            for idx, seg in enumerate(raw_timestamps):
                                f.write(f"{idx+1}\n")
                                f.write(
                                    f"{srt.format_timestamp(seg['start'] / 16000)} --> {srt.format_timestamp(seg['end'] / 16000)}\n")
                                speaker = seg.get('speaker', 'UNKNOWN')
                                f.write(
                                    f"{speaker} (start: {seg['start']}, end: {seg['end']})\n\n")
                    else:
                        timestamps = None

                if timestamps is None:
                    self.update_status(
                        f"[{i+1}/{total}] Running VAD...")
                    self.after(0, lambda: self.sub_progress_bar.set(0.05))
                    timestamps, wav_tensor = srt.get_vad_timestamps(
                        audio_bytes)
                else:
                    # Convert bytes to tensor without loading Silero VAD
                    audio_stream = io.BytesIO(audio_bytes)
                    import wave
                    import numpy as np
                    with wave.open(audio_stream, 'rb') as wav_file:
                        params = wav_file.getparams()
                        frames = wav_file.readframes(params.nframes)
                        audio_np = np.frombuffer(
                            frames, dtype=np.int16).astype(np.float32) / 32768.0
                        wav_tensor = torch.from_numpy(audio_np)

                self.after(0, lambda: self.sub_progress_bar.set(0.1))

                # Unload Diarization/VAD models before transcription if requested
                if self.unload_models_var.get():
                    # This will also clear torch cache which helps VAD/Diarization memory
                    srt.unload_models()

                # 3. Transcribe
                self.update_status(f"[{i+1}/{total}] Transcribing...")
                self.after(0, lambda: self.tps_label.configure(text=""))
                segments = srt.transcribe_fragments(
                    wav_tensor, timestamps, stats_callback=self.update_transcription_progress)
                srt.save_srt(segments, source_srt)

                # Unload ASR models before translation if requested
                if self.unload_models_var.get():
                    srt.unload_models()

                # 4. Translate
                self.update_status(f"[{i+1}/{total}] Translating...")
                self.chunk_stats = {}
                self.after(0, lambda: self.tps_label.configure(text=""))
                srt.translate_srt(source_srt, dest_srt, context,
                                  stats_callback=self.update_tps)

                self.progress_bar.set((i + 1) / total)

            # Unload models after processing all videos to free memory
            srt.unload_models()

            if not self.stop_requested:
                self.update_status("All tasks completed!", "green")
                self.after(0, lambda: [
                    self.tps_label.configure(text=""),
                    self.sub_progress_bar.set(1.0)
                ])
                tk.messagebox.showinfo(
                    "Success", f"Processed {total} videos successfully.")
                self.video_files = []
            else:
                # Keep remaining files in list if stopped
                self.video_files = self.video_files[i:]

            self.after(0, self.refresh_video_list)

        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
            tk.messagebox.showerror("Error", str(e))
        finally:
            self.is_processing = False
            self.stop_requested = False
            self.run_btn.configure(text="Start Processing", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"],
                                   hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"], state="normal")
            self.add_btn.configure(state="normal")
            self.add_folder_btn.configure(state="normal")
            self.clear_list_btn.configure(state="normal")
            self.progress_bar.set(0)
            self.sub_progress_bar.set(0)


if __name__ == "__main__":
    app = SRTGui()
    app.mainloop()
