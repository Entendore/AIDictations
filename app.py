#!/usr/bin/env python3
"""
WhisperDictate Pro — Complete Dictation & Transcription Suite
Features:
- Zen Mode (F11) for distraction-free live dictation
- Pause/Resume Mic (Ctrl+Alt+P) without model reloading
- Live Timestamp Injection for meeting minutes
- File Transcription with Hotwords & Basic Speaker Diarization
- CuPy + Numba + WebRTC VAD for ultra-low latency
"""

from __future__ import annotations
import sys
import os
import queue
import time
import json
import re
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime

# Qt
from PySide6.QtCore import (QObject, QThread, Qt, QTimer, Signal, Slot, 
                            QSettings, QPoint)
from PySide6.QtGui import (QColor, QFont, QIcon, QPainter, QPen, QPolygonF, 
                           QPalette, QCloseEvent, QDragEnterEvent, QDropEvent,
                           QGuiApplication, QPainterPath, QKeySequence, QAction)
from PySide6.QtWidgets import (QApplication, QComboBox, QPlainTextEdit, 
                               QPushButton, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QMessageBox, QCheckBox,
                               QSystemTrayIcon, QMenu, QLineEdit, QSlider,
                               QFileDialog, QDialog, QFormLayout,
                               QTableWidget, QTableWidgetItem, QHeaderView)

# Audio & ASR
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.tokenizers import Tokenizer
import scipy.signal
import ctranslate2
import webrtcvad
import numba
from numba import njit

# GPU Acceleration: CuPy with NumPy fallback
try:
    import cupy as cp
    import cupyx.scipy.signal as cpx_signal
    GPU_ENABLED = True
    def to_cpu(arr): return cp.asnumpy(arr)
except ImportError:
    import numpy as cp
    import scipy.signal as cpx_signal
    GPU_ENABLED = False
    def to_cpu(arr): return arr

# Typing & Hotkeys
try:
    from pynput.keyboard import Controller as KeyboardController, Key, Listener as KeyboardListener
    from pynput import keyboard as kb_module
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# =====================================================================
# Constants
# =====================================================================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_MS = 30
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)

VAD_FRAME_MS = 16
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)

SILENCE_FLUSH_MS = 450
MAX_BUFFER_S = 5
MIN_SPEECH_MS = 350
PRE_ROLL_FRAMES = 6

FILLER_WORDS_REGEX = re.compile(r'\b(um|uh|er|ah|umm|uhh|ermm)\b[\s,]*', re.IGNORECASE)

DEFAULT_MACROS = {
    "scratch that": "<<BACKSPACE_WORD>>",
    "delete word": "<<DELETE_WORD>>",
    "undo that": "<<UNDO>>",
    "correct that": "<<CORRECT>>",
    "move left": "<<MOVE_LEFT>>",
    "move right": "<<MOVE_RIGHT>>",
    "select word": "<<SELECT_WORD>>",
    "caps lock on": "<<CAPS_ON>>",
    "caps lock off": "<<CAPS_OFF>>",
    "new paragraph": "<<NEW_PARAGRAPH>>",
    "new line": "<<NEW_LINE>>",
    "bold that": "<<BOLD>>",
    "italicize that": "<<ITALIC>>",
    "underline that": "<<UNDERLINE>>",
    "copy that": "<<COPY>>",
    "comma": ",",
    "period": ".",
    "full stop": ".",
    "question mark": "?",
    "exclamation mark": "!",
    "semicolon": ";",
    "colon": ":",
}

LANGUAGES = {
    "Auto-Detect": None, "English": "en", "Spanish": "es", "French": "fr", 
    "German": "de", "Italian": "it", "Portuguese": "pt", "Dutch": "nl", 
    "Japanese": "ja", "Chinese": "zh", "Korean": "ko", "Russian": "ru", 
    "Arabic": "ar", "Hindi": "hi"
}

MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

@dataclass(slots=True)
class Utterance:
    text: str
    start_time: float
    end_time: float

# =====================================================================
# Utility: Dynamic Compute Type
# =====================================================================
def get_optimal_compute_type(device: str) -> str:
    if device == "cuda": return "float16"
    supported = ctranslate2.get_supported_compute_types("cpu")
    if "int8_float16" in supported: return "int8_float16"
    elif "int8" in supported: return "int8"
    elif "float16" in supported: return "float16"
    return "float32"

# =====================================================================
# PERFORMANCE: Numba JIT Fused Audio DSP (CPU Path)
# =====================================================================
@njit(fastmath=True, cache=True)
def numba_process_audio_block(frame, b, a, zi, env_zi, gate_threshold, target_rms, agc_smoothing, agc_state):
    n = len(frame)
    out = np.empty(n, dtype=np.float32)
    env_gain = agc_state[0]
    agc_gain = agc_state[1]
    
    for i in range(n):
        x = frame[i]
        if x > 0.99: x = 0.99
        elif x < -0.99: x = -0.99
        
        y = b[0] * x + zi[0]
        for j in range(1, len(a)):
            zi[j-1] = b[j] * x - a[j] * y + zi[j]
        zi[len(a)-2] = b[len(a)-1] * x - a[len(a)-1] * y
        
        abs_y = abs(y)
        if abs_y > gate_threshold:
            env_gain = 0.1 + 0.9 * env_gain
        else:
            env_gain = 0.0 + 0.9 * env_gain
        y *= env_gain
        
        if y > 1e-5:
            desired = target_rms / abs(y)
            agc_gain = (agc_gain * agc_smoothing) + (desired * (1.0 - agc_smoothing))
            if agc_gain > 10.0: agc_gain = 10.0
            
        y *= agc_gain
        if y > 0.95: y = 0.95
        elif y < -0.95: y = -0.95
        
        out[i] = y
        
    agc_state[0] = env_gain
    agc_state[1] = agc_gain
    return out

@njit(cache=True)
def numba_get_contiguous(buffer, idx, full, out):
    n = len(buffer)
    if not full:
        for i in range(idx):
            out[i] = buffer[i]
        return idx
    else:
        j = 0
        for i in range(idx, n):
            out[j] = buffer[i]
            j += 1
        for i in range(idx):
            out[j] = buffer[i]
            j += 1
        return n

@njit(cache=True)
def numba_get_agreed_len(curr_tokens, prev_tokens):
    min_len = min(len(curr_tokens), len(prev_tokens))
    for i in range(min_len):
        if curr_tokens[i] != prev_tokens[i]:
            return i
    return min_len

# =====================================================================
# PERFORMANCE: Circular Audio Buffer (Zero-Fragmentation)
# =====================================================================
class CircularAudioBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = np.zeros(maxlen, dtype=np.float32)
        self.out_array = np.zeros(maxlen, dtype=np.float32)
        self.idx = 0
        self.full = False
        if GPU_ENABLED:
            self.buffer = cp.asarray(self.buffer)

    def push(self, frame):
        n = len(frame)
        if self.idx + n > self.maxlen:
            end = self.maxlen - self.idx
            self.buffer[self.idx:] = frame[:end]
            self.buffer[:n-end] = frame[end:]
            self.idx = n - end
            self.full = True
        else:
            self.buffer[self.idx:self.idx+n] = frame
            self.idx += n
            if self.idx == self.maxlen:
                self.idx = 0
                self.full = True

    def get_data(self) -> np.ndarray:
        if GPU_ENABLED:
            if not self.full:
                gpu_data = self.buffer[:self.idx]
            else:
                gpu_data = cp.concatenate((self.buffer[self.idx:], self.buffer[:self.idx]))
            return to_cpu(gpu_data)
        else:
            valid_len = numba_get_contiguous(self.buffer, self.idx, self.full, self.out_array)
            return self.out_array[:valid_len].copy()

    def clear(self):
        self.idx = 0
        self.full = False
        self.buffer.fill(0.0)

# =====================================================================
# Audio Preprocessing Wrapper
# =====================================================================
class AudioPreprocessor:
    def __init__(self, fs=SAMPLE_RATE, cutoff_hz=80, noise_gate_db=-40.0, target_rms=0.08):
        self.use_gpu = GPU_ENABLED
        nyq = 0.5 * fs
        normal_cutoff = cutoff_hz / nyq
        self.b, self.a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1, dtype=np.float64)
        self.env_zi = np.zeros(1, dtype=np.float64)
        self.gate_threshold = 10 ** (noise_gate_db / 20.0)
        self.target_rms = target_rms
        self.agc_smoothing = 0.95
        self.agc_state = np.array([0.0, 1.0], dtype=np.float32)
        
        if self.use_gpu:
            self.b_gpu = cp.asarray(self.b)
            self.a_gpu = cp.asarray(self.a)
            self.zi_gpu = cp.zeros(max(len(self.a), len(self.b)) - 1, dtype=cp.float32)
            self.env_b = cp.array([0.1])
            self.env_a = cp.array([1, -0.9])
            self.env_zi_gpu = cp.zeros(1, dtype=cp.float32)

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            frame_gpu = cp.asarray(frame)
            limited = cp.clip(frame_gpu, -0.99, 0.99)
            filtered, self.zi_gpu = cpx_signal.lfilter(self.b_gpu, self.a_gpu, limited, zi=self.zi_gpu)
            env = cp.abs(filtered)
            target = cp.where(env > self.gate_threshold, 1.0, 0.0).astype(cp.float32)
            gain, self.env_zi_gpu = cpx_signal.lfilter(self.env_b, self.env_a, target, zi=self.env_zi_gpu)
            filtered *= gain
            rms = cp.sqrt(cp.mean(filtered**2))
            if rms > 1e-5:
                desired_gain = self.target_rms / rms
                self.agc_state[1] = (self.agc_state[1] * self.agc_smoothing) + (desired_gain * (1.0 - self.agc_smoothing))
                if self.agc_state[1] > 10.0: self.agc_state[1] = 10.0
                filtered *= self.agc_state[1]
            return cp.clip(filtered, -0.95, 0.95).astype(cp.float32)
        else:
            return numba_process_audio_block(
                frame, self.b, self.a, self.zi, self.env_zi, 
                self.gate_threshold, self.target_rms, self.agc_smoothing, self.agc_state
            )

# =====================================================================
# WebRTC VAD (Zero-Latency C implementation)
# =====================================================================
class WebRTC_VAD:
    def __init__(self, aggressiveness: int = 3):
        self.vad = webrtcvad.Vad(aggressiveness)

    def process_frame(self, frame: np.ndarray) -> bool:
        frame_int16 = (frame * 32767).astype(np.int16).tobytes()
        try:
            is_speech = False
            for i in range(0, len(frame_int16), VAD_FRAME_SAMPLES * 2):
                chunk = frame_int16[i : i + VAD_FRAME_SAMPLES * 2]
                if len(chunk) == VAD_FRAME_SAMPLES * 2:
                    if self.vad.is_speech(chunk, SAMPLE_RATE):
                        is_speech = True
                        break
            return is_speech
        except Exception:
            return False

# =====================================================================
# Thread 1: Audio Ingest Worker (VAD & Preprocessing)
# =====================================================================
class AudioIngestWorker(QObject):
    vad_state_changed = Signal(bool)
    level = Signal(float)
    error = Signal(str)
    started_evt = Signal()
    stopped_evt = Signal()

    def __init__(self, vad_aggressiveness: int = 3, ptt_mode: bool = False, inference_interval: float = 0.3):
        super().__init__()
        self.vad_aggressiveness = vad_aggressiveness
        self.ptt_mode = ptt_mode
        self.inference_interval = inference_interval
        self._ptt_active = False
        
        self._stop_flag = False
        self._paused = False
        self._audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=100)
        
        self.vad: Optional[WebRTC_VAD] = None
        self.preprocessor = AudioPreprocessor()
        
        self.audio_buffer = CircularAudioBuffer(MAX_BUFFER_S * SAMPLE_RATE)
        self.pre_roll_buffer = deque(maxlen=PRE_ROLL_FRAMES)
        self.silence_ms = 0
        self.is_speaking = False
        self._prev_vad_state = False
        self.utter_start_time = 0.0
        self.last_inference_time = 0.0
        
        self.inference_worker: Optional['InferenceWorker'] = None 
        self._last_level_emit = 0.0

    def enqueue(self, frame: np.ndarray):
        if self._paused: return  # Drop frames if paused
        try:
            self._audio_q.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        self._stop_flag = True
        try:
            self._audio_q.put_nowait(np.zeros(BLOCK_SAMPLES, dtype=np.float32))
        except queue.Full:
            pass

    def set_paused(self, paused: bool):
        self._paused = paused
        if paused:
            self.vad_state_changed.emit(False)
            self.is_speaking = False
            self.audio_buffer.clear()
            self.pre_roll_buffer.clear()
        else:
            self._prev_vad_state = False

    def set_ptt_active(self, active: bool):
        self._ptt_active = active
        if active:
            self.utter_start_time = time.time()
            self.is_speaking = True
            self.vad_state_changed.emit(True)
            if self.inference_worker: self.inference_worker.utter_start_time = self.utter_start_time
        else:
            self.vad_state_changed.emit(False)
            if self.inference_worker: self.inference_worker.flush_utterance()

    def set_vad_threshold(self, val: float):
        agg = 3 if val > 0.6 else 2 if val > 0.3 else 1
        self.vad = WebRTC_VAD(agg)

    def set_inference_interval(self, val: float):
        self.inference_interval = val

    def abort_current(self):
        self.audio_buffer.clear()
        self.pre_roll_buffer.clear()
        if self.inference_worker: self.inference_worker.abort_current()

    def run(self):
        try:
            self.vad = WebRTC_VAD(self.vad_aggressiveness)
            self.started_evt.emit()
        except Exception as e:
            self.error.emit(f"VAD init failed: {e}")
            return

        while not self._stop_flag:
            frame = self._audio_q.get()
            if self._stop_flag: break
            
            clean_frame = self.preprocessor.process(frame)
            
            current_time = time.time()
            if current_time - self._last_level_emit > 0.033:
                rms = float(np.sqrt(np.mean(to_cpu(clean_frame) ** 2)))
                self.level.emit(rms)
                self._last_level_emit = current_time
            
            clean_frame_cpu = to_cpu(clean_frame)
            
            if self.ptt_mode:
                is_speech = self._ptt_active
            else:
                is_speech = self.vad.process_frame(clean_frame_cpu)
            
            if is_speech != self._prev_vad_state and not self.ptt_mode:
                self.vad_state_changed.emit(is_speech)
                if is_speech:
                    self.audio_buffer.clear()
                    if self.pre_roll_buffer:
                        self.audio_buffer.push(np.concatenate(self.pre_roll_buffer))
                    self.pre_roll_buffer.clear()
                    self.utter_start_time = time.time()
                    if self.inference_worker: self.inference_worker.utter_start_time = self.utter_start_time
                self._prev_vad_state = is_speech
            
            if is_speech:
                self.is_speaking = True
                self.silence_ms = 0
                self.audio_buffer.push(clean_frame)
            elif self.is_speaking:
                self.audio_buffer.push(clean_frame)
                self.silence_ms += BLOCK_MS
                if self.silence_ms >= SILENCE_FLUSH_MS:
                    if self.inference_worker: self.inference_worker.flush_utterance()
                    self.is_speaking = False
                    self.audio_buffer.clear()
            else:
                self.pre_roll_buffer.append(clean_frame_cpu)
            
            if self.is_speaking and (current_time - self.last_inference_time) >= self.inference_interval:
                if self.audio_buffer.idx > SAMPLE_RATE * 0.2 or self.audio_buffer.full:
                    if self.inference_worker:
                        audio_np = self.audio_buffer.get_data()
                        self.inference_worker.push_audio(audio_np)
                    self.last_inference_time = current_time

        self.stopped_evt.emit()

# =====================================================================
# Thread 2: AI Inference Worker (Whisper & LA2)
# =====================================================================
class InferenceWorker(QObject):
    text_committed = Signal(str)
    utterance_logged = Signal(object)
    partial_text   = Signal(str)
    status_msg     = Signal(str)
    telemetry_update = Signal(float, float)
    processing_state_changed = Signal(bool)
    error          = Signal(str)
    started_evt    = Signal()
    stopped_evt    = Signal()

    def __init__(self, model_size: str = "large-v3-turbo", language: Optional[str] = None, 
                 hotwords: str = "", translate: bool = False, beam_size: int = 1):
        super().__init__()
        self.model_size = model_size
        self.language = language
        self.locked_language: Optional[str] = None
        self.hotwords = hotwords
        self.translate = translate
        self.beam_size = beam_size
        
        self._stop_flag = False
        self._asr_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        
        self._abort_flag = False
        self._flush_flag = False
        self._clear_ctx_flag = False
        self._swap_model_flag = False
        self._swap_model_name = ""
        
        self.model: Optional[WhisperModel] = None
        self.tokenizer: Optional[Tokenizer] = None
        
        self.utter_start_time = 0.0
        self.committed_tokens: List[int] = []
        self.prev_tokens: List[int] = []
        self.context_prompt = self.hotwords

    def push_audio(self, audio_np: np.ndarray):
        self._asr_q.put(audio_np)

    def stop(self):
        self._stop_flag = True
        self._asr_q.put(None)

    def request_model_swap(self, new_model: str):
        self._swap_model_name = new_model
        self._swap_model_flag = True
        self._asr_q.put(None)

    def abort_current(self):
        self._abort_flag = True
        self._asr_q.put(None)

    def clear_context(self):
        self._clear_ctx_flag = True
        self._asr_q.put(None)

    def flush_utterance(self):
        self._flush_flag = True
        self._asr_q.put(None)

    def _load_models(self):
        self.status_msg.emit(f"Loading Whisper '{self.model_size}'...")
        device = "cuda" if GPU_ENABLED else "cpu"
        compute = get_optimal_compute_type(device)
        
        self.model = WhisperModel(
            self.model_size, device=device, compute_type=compute,
            cpu_threads=os.cpu_count() or 4, num_workers=1
        )
        self.tokenizer = self.model.hf_tokenizer
        self.status_msg.emit("Ready. Listening...")
        self.started_evt.emit()

    def _check_model_swap(self):
        if self._swap_model_flag:
            self._swap_model_flag = False
            self.status_msg.emit(f"Swapping to {self._swap_model_name}...")
            del self.model
            del self.tokenizer
            if GPU_ENABLED: cp.get_default_memory_pool().free_all_blocks()
                
            self.model_size = self._swap_model_name
            device = "cuda" if GPU_ENABLED else "cpu"
            compute = get_optimal_compute_type(device)
            self.model = WhisperModel(self.model_size, device=device, compute_type=compute, cpu_threads=os.cpu_count(), num_workers=1)
            self.tokenizer = self.model.hf_tokenizer
            self._reset_context()
            self.status_msg.emit("Ready. Listening...")

    def _reset_context(self):
        self.context_prompt = self.hotwords
        self.committed_tokens.clear()
        self.prev_tokens.clear()

    def run(self):
        try: self._load_models()
        except Exception as e:
            self.error.emit(f"Model load failed: {e}")
            return

        while not self._stop_flag:
            audio_np = self._asr_q.get()
            
            if audio_np is None:
                if self._stop_flag: break
                if self._abort_flag:
                    self._abort_flag = False
                    self._asr_q.queue.clear()
                    self._reset_context()
                    self.partial_text.emit("")
                    continue
                if self._clear_ctx_flag:
                    self._clear_ctx_flag = False
                    self._reset_context()
                    self.status_msg.emit("Context cleared")
                    continue
                if self._swap_model_flag:
                    self._check_model_swap()
                    continue
                if self._flush_flag:
                    self._flush_flag = False
                    self._do_flush()
                    continue
                continue
                
            while self._asr_q.qsize() > 0:
                next_item = self._asr_q.get()
                if next_item is None:
                    self._asr_q.put(None)
                    audio_np = None
                    break
                audio_np = next_item
            
            if audio_np is None: continue
                
            self._run_la2_inference(audio_np)

        self._do_flush()
        self.stopped_evt.emit()

    def _run_la2_inference(self, audio_np: np.ndarray):
        self.processing_state_changed.emit(True)
        prompt = self.context_prompt if self.context_prompt else None
        current_lang = self.locked_language if self.locked_language else self.language
        
        inf_start = time.time()
        segments, info = self.model.transcribe(
            audio_np, language=current_lang,
            task="translate" if self.translate else "transcribe",
            beam_size=self.beam_size, without_timestamps=True,
            vad_filter=False, initial_prompt=prompt,
            condition_on_previous_text=bool(prompt),
            hotwords=self.hotwords if self.hotwords else None,
            temperature=0.0, max_initial_timestamp_index=1
        )
        
        if self.language is None and not self.locked_language and info.language:
            self.locked_language = info.language
            self.status_msg.emit(f"Locked Language: {info.language}")
        
        text = "".join([seg.text for seg in segments]).strip()
        inf_time = time.time() - inf_start
        audio_dur = len(audio_np) / SAMPLE_RATE
        
        if audio_dur > 0:
            self.telemetry_update.emit(inf_time / audio_dur, inf_time * 1000)
        self.processing_state_changed.emit(False)
        
        if not text: return
            
        current_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        curr_arr = np.array(current_tokens, dtype=np.int32)
        prev_arr = np.array(self.prev_tokens, dtype=np.int32)
        
        agreed_len = numba_get_agreed_len(curr_arr, prev_arr)
                
        if agreed_len > len(self.committed_tokens):
            new_tokens = current_tokens[len(self.committed_tokens):agreed_len]
            new_text = self.tokenizer.decode(new_tokens)
            if not new_text.startswith((" ", ",", ".", "?", "!", ";", ":")) and self.committed_tokens:
                new_text = " " + new_text
            self.text_committed.emit(new_text.strip())
            self.committed_tokens = current_tokens[:agreed_len]
        
        pending_tokens = current_tokens[len(self.committed_tokens):]
        self.partial_text.emit(self.tokenizer.decode(pending_tokens).strip())
        self.prev_tokens = current_tokens

    def _do_flush(self):
        utter_duration_ms = (time.time() - self.utter_start_time) * 1000
        if utter_duration_ms < MIN_SPEECH_MS and not self._ptt_active:
            self._reset_context()
            self.partial_text.emit("")
            return

        if len(self.prev_tokens) > len(self.committed_tokens):
            remaining_text = self.tokenizer.decode(self.prev_tokens[len(self.committed_tokens):])
            if remaining_text.strip():
                if not remaining_text.startswith((" ", ",", ".", "?", "!", ";", ":")) and self.committed_tokens:
                    remaining_text = " " + remaining_text
                self.text_committed.emit(remaining_text.strip())
        
        if self.prev_tokens:
            final_text = self.tokenizer.decode(self.prev_tokens).strip()
            if final_text:
                utt = Utterance(text=final_text, start_time=self.utter_start_time, end_time=time.time())
                self.utterance_logged.emit(utt)
            base_prompt = self.hotwords + " " if self.hotwords else ""
            self.context_prompt = base_prompt + self.tokenizer.decode(self.prev_tokens[-32:])
        
        self._reset_context()
        self.partial_text.emit("")

# =====================================================================
# Batch File Transcriber (Drag & Drop) - Now with Hotwords & Speaker Diarization
# =====================================================================
class FileTranscriber(QObject):
    status_msg = Signal(str)
    error = Signal(str)
    segment_ready = Signal(str)
    finished = Signal()

    def __init__(self, filepath: str, model_size: str, hotwords: str = ""):
        super().__init__()
        self.filepath = filepath
        self.model_size = model_size
        self.hotwords = hotwords
        self._stop_flag = False

    def stop(self): self._stop_flag = True

    def run(self):
        try:
            self.status_msg.emit("Loading model for file...")
            device = "cuda" if GPU_ENABLED else "cpu"
            compute = get_optimal_compute_type(device)
            model = WhisperModel(self.model_size, device=device, compute_type=compute, cpu_threads=os.cpu_count(), num_workers=1)
            self.status_msg.emit(f"Transcribing {Path(self.filepath).name}...")
            
            # Use VAD filter to split segments by 2s silence for basic diarization
            segments, info = model.transcribe(
                self.filepath, beam_size=5, word_timestamps=True, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=2000),
                hotwords=self.hotwords if self.hotwords else None
            )
            
            speaker = 1
            for seg in segments:
                if self._stop_flag: break
                text = seg.text.strip()
                if text:
                    self.segment_ready.emit(f"Speaker {speaker}: {text} ")
                    speaker = 2 if speaker == 1 else 1
                    
            self.status_msg.emit("File transcription complete.")
        except Exception as e:
            self.error.emit(f"File transcribe failed: {e}")
        finally:
            self.finished.emit()

# =====================================================================
# Audio Capture (C-Thread)
# =====================================================================
class AudioCapture(QObject):
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stream = None
        self._worker: Optional[AudioIngestWorker] = None

    def attach(self, worker: AudioIngestWorker): self._worker = worker

    def start(self, device=None):
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                blocksize=BLOCK_SAMPLES, device=device, callback=self._callback
            )
            self._stream.start()
        except Exception as e:
            self.error.emit(f"Audio init failed: {e}\nCheck microphone permissions.")

    def stop(self):
        if self._stream is not None:
            self._stream.stop(); self._stream.close(); self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if self._worker is not None:
            if self._worker._audio_q.full(): return
            self._worker.enqueue(indata[:, 0].copy())

# =====================================================================
# UI: Teleprompter Overlay
# =====================================================================
class CaptionOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 10, 15, 10)
        
        self.lbl_committed = QLabel("")
        self.lbl_committed.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_committed.setStyleSheet("color: white; background: transparent;")
        self.lbl_committed.setAlignment(Qt.AlignCenter)
        self.lbl_committed.setWordWrap(True)
        self.layout.addWidget(self.lbl_committed)
        
        self.lbl_partial = QLabel("")
        self.lbl_partial.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_partial.setStyleSheet("color: #888888; background: transparent;")
        self.lbl_partial.setAlignment(Qt.AlignCenter)
        self.lbl_partial.setWordWrap(True)
        self.layout.addWidget(self.lbl_partial)
        
        self._drag_pos = None
        self._timer = QTimer(self, interval=2000, timeout=self._hide_self)
        self._timer.start()

    @Slot(str)
    def update_committed(self, text: str):
        if text:
            if not self.isVisible(): self.show()
            self.lbl_committed.setText(text)
            self._timer.stop(); self._timer.start()

    @Slot(str)
    def update_partial(self, text: str):
        if text:
            if not self.isVisible(): self.show()
            self.lbl_partial.setText(text)
            self._timer.stop(); self._timer.start()
        else: self.lbl_partial.setText("")

    def _hide_self(self): self.hide()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton and self._drag_pos is not None:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e): self._drag_pos = None; e.accept()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(20, 20, 20, 200))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 15, 15)

# =====================================================================
# UI: Custom Voice Commands Dialog
# =====================================================================
class CommandsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Voice Commands")
        self.resize(500, 300)
        self.commands: Dict[str, str] = {}
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Voice Phrase (Spoken)", "Text / Action"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        v.addWidget(self.table)
        btn_add = QPushButton("Add New Command")
        btn_add.clicked.connect(self._add_row)
        v.addWidget(btn_add)
        btns = QHBoxLayout()
        ok = QPushButton("Save"); ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(cancel); btns.addWidget(ok)
        v.addLayout(btns)
        self._load_settings()

    def _add_row(self, phrase="", action=""):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(phrase))
        self.table.setItem(row, 1, QTableWidgetItem(action))

    def _load_settings(self):
        s = QSettings("WhisperDictate", "Pro")
        cmds = s.value("custom_commands", {}, type=dict)
        for p, a in cmds.items(): self._add_row(p, a)

    def accept(self):
        s = QSettings("WhisperDictate", "Pro")
        cmds = {}
        for r in range(self.table.rowCount()):
            p_item = self.table.item(r, 0); a_item = self.table.item(r, 1)
            if p_item and a_item and p_item.text().strip():
                cmds[p_item.text().strip().lower()] = a_item.text()
        s.setValue("custom_commands", cmds)
        self.commands = cmds
        super().accept()

# =====================================================================
# Visualizer with Pre-Allocated Memory
# =====================================================================
class WaveformHistory(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self._maxlen = 100
        self._history = deque(maxlen=self._maxlen)
        self._timer = QTimer(self, interval=33, timeout=self.update)
        self._timer.start()
        self._color = QColor("#3a4a5a")
        self._vad_threshold = 0.5
        
        self._points = [QPoint(0, 0) for _ in range(self._maxlen + 2)]
        self._poly = QPolygonF(self._points)
        
    @Slot(bool)
    def set_speaking(self, is_speaking: bool):
        self._color = QColor("#3ad95a") if is_speaking else QColor("#3a4a5a")
        
    @Slot(bool)
    def set_processing(self, is_processing: bool):
        if is_processing: self._color = QColor("#d9c14a")

    @Slot(float)
    def set_vad_threshold(self, val: float):
        self._vad_threshold = val
        self.update()

    @Slot(float)
    def push(self, rms: float):
        import math
        db = 20 * math.log10(max(rms, 1e-8))
        norm = max(0.0, min(1.0, (db + 60) / 60))
        self._history.append(norm)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#101418"))

        if len(self._history) < 2: return
        step = w / self._maxlen
        
        self._points[0].setX(0); self._points[0].setY(h)
        for i, v in enumerate(self._history):
            x = int(i * step)
            y = h - int(v * (h - 6)) - 3
            self._points[i+1].setX(x); self._points[i+1].setY(y)
        
        last_idx = len(self._history)
        self._points[last_idx+1].setX(int(last_idx * step)); self._points[last_idx+1].setY(h)

        self._poly.clear()
        for pt in self._points[:last_idx+2]:
            self._poly.append(pt)

        grad = QColor(self._color); grad.setAlpha(140)
        p.setBrush(grad); p.setPen(Qt.NoPen)
        p.drawPolygon(self._poly)
        
        import math
        threshold_db = 20 * math.log10(max(1e-5, self._vad_threshold))
        threshold_norm = max(0.0, min(1.0, (threshold_db + 60) / 60))
        thresh_y = h - int(threshold_norm * (h - 6)) - 3
        
        pen = QPen(QColor("#d9534f"), 1, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(0, thresh_y, w, thresh_y)

# =====================================================================
# Main Window
# =====================================================================
class DictationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperDictate Pro")
        self.resize(950, 650)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_tray()
        
        self.overlay = CaptionOverlay()
        self.overlay.move(100, 100)
        
        self._ingest_thread: Optional[QThread] = None
        self._infer_thread: Optional[QThread] = None
        self._ingest_worker: Optional[AudioIngestWorker] = None
        self._inference_worker: Optional[InferenceWorker] = None
        
        self._file_thread: Optional[QThread] = None
        self._file_worker: Optional[FileTranscriber] = None
        
        self._capture = AudioCapture()
        self._capture.error.connect(self._show_error)
        
        self.keyboard = KeyboardController() if KEYBOARD_AVAILABLE else None
        self._last_char_was_punct = False
        self._utterances: List[Utterance] = []
        self._custom_macros = DEFAULT_MACROS.copy()
        self._typing_muted = False
        self._zen_mode = False
        
        self._restore_settings()
        self._setup_global_hotkeys()
        self._update_word_count()
        
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(15000)
        self._autosave_timer.timeout.connect(self._save_text_to_settings)
        self._autosave_timer.start()

    def _build_ui(self):
        self.v_layout = QVBoxLayout(self)
        
        # Top Control Bar
        self.bar_top = QHBoxLayout()
        self.btn_record = QPushButton("⏺  Start  (Ctrl+Alt+Space)")
        self.btn_record.setCheckable(True)
        self.btn_record.setStyleSheet("QPushButton:checked { background:#d9534f; color:white; }")
        
        self.btn_pause = QPushButton("⏸  Pause (Ctrl+Alt+P)")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("QPushButton:checked { background:#d9c14a; color:black; }")
        
        self.cmb_mic = QComboBox()
        self.cmb_mic.addItem("Default Mic", None)
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0: self.cmb_mic.addItem(dev['name'], i)
                
        self.cmb_model = QComboBox()
        for m in MODELS: self.cmb_model.addItem(m)
        self.cmb_model.setCurrentText("large-v3-turbo")
        
        self.cmb_lang = QComboBox()
        for ln, code in LANGUAGES.items(): self.cmb_lang.addItem(ln, code)
        self.cmb_lang.setCurrentText("Auto-Detect")
        
        self.btn_export = QPushButton("💾 Export")
        self.btn_cmds = QPushButton("🗣️ Commands")
        self.btn_clear_ctx = QPushButton("🧹 Clear Ctx")
        
        self.bar_top.addWidget(self.btn_record)
        self.bar_top.addWidget(self.btn_pause)
        self.bar_top.addWidget(self.cmb_mic)
        self.bar_top.addWidget(QLabel("Model:")); self.bar_top.addWidget(self.cmb_model)
        self.bar_top.addWidget(QLabel("Lang:")); self.bar_top.addWidget(self.cmb_lang)
        self.bar_top.addWidget(self.btn_cmds); self.bar_top.addWidget(self.btn_clear_ctx); self.bar_top.addWidget(self.btn_export)
        self.bar_top.addStretch(1)
        
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color:#9aa0a6; font-weight: bold;")
        self.bar_top.addWidget(self.lbl_status)
        self.v_layout.addLayout(self.bar_top)

        # Options Bar
        self.bar_opts = QHBoxLayout()
        self.chk_systemwide = QCheckBox("System-Wide")
        self.chk_fast_type = QCheckBox("Fast Type (Ctrl+V)")
        self.chk_overlay = QCheckBox("Overlay"); self.chk_overlay.setChecked(True)
        self.chk_translate = QCheckBox("Translate")
        self.chk_beam = QCheckBox("High Acc")
        self.chk_ptt = QCheckBox("PTT (Hold F8 / R-Shift)")
        self.chk_clean = QCheckBox("Clean Text"); self.chk_clean.setChecked(True)
        self.chk_timestamps = QCheckBox("Add Timestamps")
        
        self.lbl_vad = QLabel("VAD:")
        self.sld_vad = QSlider(Qt.Horizontal); self.sld_vad.setRange(10, 90)
        self.sld_vad.setValue(60); self.sld_vad.setFixedWidth(80)
        
        self.lbl_resp = QLabel("Speed:")
        self.sld_resp = QSlider(Qt.Horizontal); self.sld_resp.setRange(10, 100)
        self.sld_resp.setValue(30); self.sld_resp.setFixedWidth(80)
        
        self.bar_opts.addWidget(self.chk_systemwide); self.bar_opts.addWidget(self.chk_fast_type); self.bar_opts.addWidget(self.chk_clean)
        self.bar_opts.addWidget(self.chk_timestamps)
        self.bar_opts.addWidget(self.chk_overlay); self.bar_opts.addWidget(self.chk_translate); self.bar_opts.addWidget(self.chk_beam); self.bar_opts.addWidget(self.chk_ptt)
        self.bar_opts.addWidget(self.lbl_vad); self.bar_opts.addWidget(self.sld_vad)
        self.bar_opts.addWidget(self.lbl_resp); self.bar_opts.addWidget(self.sld_resp)
        self.bar_opts.addStretch(1)
        
        engine_name = "CuPy" if GPU_ENABLED else "Numba"
        self.lbl_telemetry = QLabel(f"RTF: 0.00x | Latency: 0ms | Engine: {engine_name}")
        self.lbl_telemetry.setStyleSheet("color: #4a9aDA; font-weight: bold;")
        self.bar_opts.addWidget(self.lbl_telemetry)
        self.v_layout.addLayout(self.bar_opts)

        # Hotwords Bar
        self.hw_layout = QHBoxLayout()
        self.hw_layout.addWidget(QLabel("Hotwords:"))
        self.edt_hotwords = QLineEdit()
        self.edt_hotwords.setPlaceholderText("CRISPR, PySide6, QWERTY, names...")
        self.hw_layout.addWidget(self.edt_hotwords)
        self.v_layout.addLayout(self.hw_layout)

        self.waveform = WaveformHistory()
        self.sld_vad.valueChanged.connect(self.waveform.set_vad_threshold)
        self.v_layout.addWidget(self.waveform)

        self.lbl_partial = QLabel("")
        f = QFont("Arial", 12); f.setItalic(True)
        self.lbl_partial.setFont(f)
        self.lbl_partial.setStyleSheet("color:#888888; padding: 4px;")
        self.lbl_partial.setMinimumHeight(20)
        self.v_layout.addWidget(self.lbl_partial)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Start speaking or drag & drop an audio file here...\nPress F11 for Zen Mode.")
        self.editor.setFont(QFont("Arial", 12))
        self.editor.textChanged.connect(self._update_word_count)
        self.v_layout.addWidget(self.editor, 1)

        self.lbl_word_count = QLabel("Words: 0")
        self.lbl_word_count.setStyleSheet("color:#9aa0a6;")
        self.v_layout.addWidget(self.lbl_word_count)

        # Connections
        self.btn_record.toggled.connect(self._on_record_toggled)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_cmds.clicked.connect(self._on_open_cmds)
        self.btn_clear_ctx.clicked.connect(self._on_clear_context)
        self.sld_vad.valueChanged.connect(self._on_vad_changed)
        self.sld_resp.valueChanged.connect(self._on_resp_changed)
        self.chk_ptt.toggled.connect(self._on_ptt_toggled)
        self.cmb_model.currentTextChanged.connect(self._on_model_changed_live)

    def _build_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable(): return
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(self.style().standardIcon(self.style().SP_MediaVolume))
        self.tray.setToolTip("WhisperDictate")
        menu = QMenu()
        menu.addAction("Show", self.showNormal)
        menu.addAction("Toggle Mic", self._toggle_record_from_hotkey)
        menu.addAction("Pause/Resume", self._toggle_pause_from_hotkey)
        menu.addAction("Mute Typing", self._toggle_mute_from_hotkey)
        menu.addAction("Abort Utterance", self._abort_from_hotkey)
        menu.addAction("Zen Mode", self._toggle_zen_mode)
        menu.addAction("Quit", QApplication.quit)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _setup_global_hotkeys(self):
        if KEYBOARD_AVAILABLE:
            self._hotkeys = {
                '<ctrl>+<alt>+<space>': self._toggle_record_from_hotkey,
                '<ctrl>+<alt>+<p>': self._toggle_pause_from_hotkey,
                '<ctrl>+<alt>+<backspace>': self._abort_from_hotkey,
                '<ctrl>+<alt>+<m>': self._toggle_mute_from_hotkey
            }
            self._listener = kb_module.GlobalHotKeys(self._hotkeys)
            self._listener.start()
            
            def on_press(key):
                if key in (kb_module.Key.f8, kb_module.Key.shift_r) and self.chk_ptt.isChecked() and self._ingest_worker:
                    self._ingest_worker.set_ptt_active(True)
                if key == kb_module.Key.f11:
                    self._toggle_zen_mode()
            def on_release(key):
                if key in (kb_module.Key.f8, kb_module.Key.shift_r) and self.chk_ptt.isChecked() and self._ingest_worker:
                    self._ingest_worker.set_ptt_active(False)
                    
            self._ptt_listener = KeyboardListener(on_press=on_press, on_release=on_release)
            self._ptt_listener.start()

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            if filepath.lower().endswith(('.mp3', '.wav', '.mp4', '.m4a', '.flac')):
                self._transcribe_file(filepath)

    def _transcribe_file(self, filepath: str):
        if self._file_thread is not None:
            QMessageBox.warning(self, "Busy", "Already processing a file. Please wait.")
            return
        hotwords = self.edt_hotwords.text().strip()
        self._file_worker = FileTranscriber(filepath, self.cmb_model.currentText(), hotwords)
        self._file_thread = QThread()
        self._file_worker.moveToThread(self._file_thread)
        self._file_worker.status_msg.connect(self.lbl_status.setText)
        self._file_worker.error.connect(self._show_error)
        self._file_worker.segment_ready.connect(self._on_file_segment)
        self._file_worker.finished.connect(self._on_file_finished)
        self._file_thread.started.connect(self._file_worker.run)
        self._file_thread.start()

    @Slot(str)
    def _on_file_segment(self, text: str):
        cursor = self.editor.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text + "\n")
        self.editor.ensureCursorVisible()

    @Slot()
    def _on_file_finished(self):
        if self._file_thread:
            self._file_thread.quit(); self._file_thread.wait(2000)
            self._file_thread = None; self._file_worker = None

    def _toggle_record_from_hotkey(self): self.btn_record.toggle()
    def _abort_from_hotkey(self):
        if self._ingest_worker: self._ingest_worker.abort_current()
            
    def _toggle_pause_from_hotkey(self):
        if self.btn_pause.isEnabled():
            self.btn_pause.toggle()

    def _toggle_mute_from_hotkey(self):
        self._typing_muted = not self._typing_muted
        if self._typing_muted:
            self.lbl_status.setText("Typing Muted")
            self.lbl_status.setStyleSheet("color: #d9534f; font-weight: bold;")
        else: self._on_status_msg("Listening...")

    def _toggle_zen_mode(self):
        self._zen_mode = not self._zen_mode
        self.bar_top.setVisible(not self._zen_mode)
        self.bar_opts.setVisible(not self._zen_mode)
        self.hw_layout.itemAt(0).widget().setVisible(not self._zen_mode)
        self.edt_hotwords.setVisible(not self._zen_mode)
        self.waveform.setVisible(not self._zen_mode)
        self.lbl_partial.setVisible(not self._zen_mode)
        self.lbl_word_count.setVisible(not self._zen_mode)
        if self._zen_mode:
            self.setWindowTitle("Zen Mode - WhisperDictate")
            self.editor.setStyleSheet("background-color: #1a1a1a; color: #e8e8e8;")
        else:
            self.setWindowTitle("WhisperDictate Pro")
            self.editor.setStyleSheet("")

    def _on_model_changed_live(self, new_model: str):
        if self._inference_worker: self._inference_worker.request_model_swap(new_model)

    def _on_vad_changed(self, val: int):
        if self._ingest_worker: self._ingest_worker.set_vad_threshold(val / 100.0)

    def _on_resp_changed(self, val: int):
        if self._ingest_worker: self._ingest_worker.set_inference_interval(val / 100.0)

    def _on_record_toggled(self, on: bool):
        if on:
            self._start_recording()
            self.btn_record.setText("⏹  Stop (Ctrl+Alt+Space)")
            self.btn_pause.setEnabled(True)
        else:
            self._stop_recording()
            self.btn_record.setText("⏺  Start (Ctrl+Alt+Space)")
            self.btn_pause.setEnabled(False)
            self.btn_pause.setChecked(False)

    def _on_pause_toggled(self, on: bool):
        if self._ingest_worker:
            self._ingest_worker.set_paused(on)
            if on:
                self._on_status_msg("Paused")
            else:
                self._on_status_msg("Listening...")

    def _start_recording(self):
        hotwords = self.edt_hotwords.text().strip()
        translate = self.chk_translate.isChecked()
        beam_size = 5 if self.chk_beam.isChecked() else 1
        vad_threshold = self.sld_vad.value() / 100.0
        ptt_mode = self.chk_ptt.isChecked()
        interval = self.sld_resp.value() / 100.0
        lang = self.cmb_lang.currentData()
        
        self._ingest_worker = AudioIngestWorker(vad_threshold, ptt_mode, interval)
        self._inference_worker = InferenceWorker(self.cmb_model.currentText(), lang, hotwords, translate, beam_size)
        self._ingest_worker.inference_worker = self._inference_worker
        
        self._ingest_thread = QThread()
        self._infer_thread = QThread()
        self._ingest_worker.moveToThread(self._ingest_thread)
        self._inference_worker.moveToThread(self._infer_thread)
        
        self._ingest_worker.vad_state_changed.connect(self.waveform.set_speaking)
        self._ingest_worker.level.connect(self.waveform.push)
        self._ingest_worker.error.connect(self._show_error)
        self._ingest_worker.started_evt.connect(lambda: self._on_status_msg("Listening..."))
        self._ingest_worker.stopped_evt.connect(self._on_asr_stopped)
        
        self._inference_worker.text_committed.connect(self._on_text_committed)
        self._inference_worker.partial_text.connect(self._on_partial_text)
        self._inference_worker.utterance_logged.connect(self._on_utterance_logged)
        self._inference_worker.telemetry_update.connect(self._on_telemetry)
        self._inference_worker.status_msg.connect(self._on_status_msg)
        self._inference_worker.processing_state_changed.connect(self.waveform.set_processing)
        self._inference_worker.error.connect(self._show_error)
        
        self._capture.attach(self._ingest_worker)

        self._ingest_thread.started.connect(self._ingest_worker.run)
        self._infer_thread.started.connect(self._inference_worker.run)
        self._ingest_thread.start()
        self._infer_thread.start()
        
        mic_idx = self.cmb_mic.currentData()
        self._capture.start(device=mic_idx)
        self._on_status_msg("Loading models...")

    def _stop_recording(self):
        self._capture.stop()
        if self._ingest_worker: self._ingest_worker.stop()
        if self._inference_worker: self._inference_worker.stop()
        self._on_status_msg("Stopping...")

    @Slot(float, float)
    def _on_telemetry(self, rtf, latency):
        engine_name = "CuPy" if GPU_ENABLED else "Numba"
        self.lbl_telemetry.setText(f"RTF: {rtf:.2f}x | Latency: {latency:.0f}ms | Engine: {engine_name}")

    @Slot(object)
    def _on_utterance_logged(self, utt: Utterance): self._utterances.append(utt)

    @Slot(str)
    def _on_status_msg(self, msg: str):
        if self._typing_muted: return
        self.lbl_status.setText(msg)
        if "Listen" in msg: self.lbl_status.setStyleSheet("color: #3ad95a; font-weight: bold;")
        elif "Stop" in msg or "Idle" in msg or "Ready" in msg or "Pause" in msg:
            self.lbl_status.setStyleSheet("color: #9aa0a6; font-weight: bold;")
            self.waveform.set_speaking(False)
        elif "Load" in msg or "Transcrib" in msg or "Context" in msg or "Abort" in msg or "Swap" in msg or "Locked" in msg:
            self.lbl_status.setStyleSheet("color: #d9c14a; font-weight: bold;")

    @Slot(str)
    def _on_partial_text(self, text: str):
        self.lbl_partial.setText(text)
        if self.chk_overlay.isChecked(): self.overlay.update_partial(text)

    @Slot(str)
    def _on_text_committed(self, text: str):
        text_lower = text.lower()
        hard_flush = ["<<NEW_PARAGRAPH>>", "<<NEW_LINE>>", ".", "?", "!"]
        text = self._apply_macros(text)
        if text.startswith("<<"):
            self._execute_macro(text)
            if any(m in text for m in hard_flush):
                if self._inference_worker: self._inference_worker.flush_utterance()
            return
            
        text = self._auto_capitalize(text)
        if self.chk_clean.isChecked(): text = self._clean_text(text)
        
        # Timestamp Injection
        if self.chk_timestamps.isChecked() and (self._last_char_was_punct or self.editor.toPlainText() == ""):
            timestamp = datetime.now().strftime("[%H:%M:%S] ")
            text = timestamp + text
            
        if self._typing_muted:
            cursor = self.editor.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText(text)
            self.editor.setTextCursor(cursor)
            sb = self.editor.verticalScrollBar()
            if sb.value() >= sb.maximum() - 10: self.editor.ensureCursorVisible()
            if self.chk_overlay.isChecked(): self.overlay.update_committed(text)
            self._last_char_was_punct = text.endswith(('.', '!', '?'))
            return
            
        if self.chk_systemwide.isChecked() and self.keyboard:
            try:
                if self.chk_fast_type.isChecked():
                    cb = QGuiApplication.clipboard()
                    cb.setText(text)
                    self.keyboard.press(Key.ctrl); self.keyboard.press('v')
                    self.keyboard.release('v'); self.keyboard.release(Key.ctrl)
                else: self.keyboard.type(text)
            except Exception as e: print(f"Typing error: {e}")
        else:
            cursor = self.editor.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText(text)
            self.editor.setTextCursor(cursor)
            sb = self.editor.verticalScrollBar()
            if sb.value() >= sb.maximum() - 10: self.editor.ensureCursorVisible()
            if self.chk_overlay.isChecked(): self.overlay.update_committed(text)
            
        self._last_char_was_punct = text.endswith(('.', '!', '?'))
        if any(m in text_lower for m in ["period", "full stop", "question mark", "exclamation mark"]):
            self._last_char_was_punct = True
            if self._inference_worker: self._inference_worker.flush_utterance()
            
    def _clean_text(self, text: str) -> str:
        text = FILLER_WORDS_REGEX.sub('', text)
        text = re.sub(r'\s+([,.;:?!])', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _apply_macros(self, text: str) -> str:
        out = text.lower()
        for phrase, action in self._custom_macros.items():
            if phrase in out: return action
        for phrase, repl in DEFAULT_MACROS.items():
            if phrase in out: return repl
        return text

    def _execute_macro(self, macro: str):
        if self.keyboard:
            if macro == "<<BACKSPACE_WORD>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press(Key.backspace)
                self.keyboard.release(Key.backspace); self.keyboard.release(Key.ctrl)
            elif macro == "<<DELETE_WORD>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press(Key.delete)
                self.keyboard.release(Key.delete); self.keyboard.release(Key.ctrl)
            elif macro == "<<UNDO>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press('z')
                self.keyboard.release('z'); self.keyboard.release(Key.ctrl)
            elif macro == "<<CORRECT>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press(Key.shift); self.keyboard.press(Key.left)
                self.keyboard.release(Key.left); self.keyboard.release(Key.shift); self.keyboard.release(Key.ctrl)
            elif macro == "<<SELECT_WORD>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press(Key.shift); self.keyboard.press(Key.left)
                self.keyboard.release(Key.left); self.keyboard.release(Key.shift); self.keyboard.release(Key.ctrl)
            elif macro == "<<MOVE_LEFT>>":
                self.keyboard.press(Key.left); self.keyboard.release(Key.left)
            elif macro == "<<MOVE_RIGHT>>":
                self.keyboard.press(Key.right); self.keyboard.release(Key.right)
            elif macro == "<<CAPS_ON>>" or macro == "<<CAPS_OFF>>":
                self.keyboard.press(Key.caps_lock); self.keyboard.release(Key.caps_lock)
            elif macro == "<<NEW_LINE>>":
                self.keyboard.press(Key.enter); self.keyboard.release(Key.enter)
            elif macro == "<<NEW_PARAGRAPH>>":
                self.keyboard.press(Key.enter); self.keyboard.release(Key.enter)
                self.keyboard.press(Key.enter); self.keyboard.release(Key.enter)
            elif macro == "<<BOLD>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press('b')
                self.keyboard.release('b'); self.keyboard.release(Key.ctrl)
            elif macro == "<<ITALIC>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press('i')
                self.keyboard.release('i'); self.keyboard.release(Key.ctrl)
            elif macro == "<<UNDERLINE>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press('u')
                self.keyboard.release('u'); self.keyboard.release(Key.ctrl)
            elif macro == "<<COPY>>":
                self.keyboard.press(Key.ctrl); self.keyboard.press('c')
                self.keyboard.release('c'); self.keyboard.release(Key.ctrl)
        else:
            if macro == "<<BACKSPACE_WORD>>": self.editor.textCursor().deletePreviousChar()
            elif macro == "<<NEW_LINE>>": self.editor.appendPlainText("")
            elif macro == "<<NEW_PARAGRAPH>>":
                self.editor.appendPlainText(""); self.editor.appendPlainText("")

    def _auto_capitalize(self, text: str) -> str:
        if not text: return text
        if self._last_char_was_punct or self.editor.toPlainText() == "":
            text = text[0].upper() + text[1:]
        return text

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Transcription", str(Path.home() / "dictation.txt"),
            "Plain Text (*.txt);;SubRip Subtitle (*.srt);;JSON (*.json)")
        if not path: return
        text = self.editor.toPlainText()
        if path.endswith('.json'):
            data = {"text": text, "model": self.cmb_model.currentText(), "utterances": [u.__dict__ for u in self._utterances]}
            Path(path).write_text(json.dumps(data, indent=2))
        elif path.endswith('.srt'):
            srt_content = ""
            for i, utt in enumerate(self._utterances):
                srt_content += f"{i+1}\n"
                srt_content += f"{self._format_srt_time(utt.start_time)} --> {self._format_srt_time(utt.end_time)}\n"
                srt_content += f"{utt.text}\n\n"
            Path(path).write_text(srt_content)
        else: Path(path).write_text(text)

    def _format_srt_time(self, t: float) -> str:
        ms = int(t * 1000) % 1000; s = int(t) % 60
        m = int(t) // 60 % 60; h = int(t) // 3600
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _on_open_cmds(self):
        dlg = CommandsDialog(self)
        if dlg.exec(): self._custom_macros = dlg.commands

    def _on_clear_context(self):
        if self._inference_worker: self._inference_worker.clear_context()

    def _update_word_count(self):
        words = len(self.editor.toPlainText().split())
        self.lbl_word_count.setText(f"Words: {words}")

    @Slot(str)
    def _show_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.btn_record.setChecked(False)

    @Slot()
    def _on_asr_stopped(self):
        if self._ingest_thread:
            self._ingest_thread.quit(); self._ingest_thread.wait(2000)
            self._ingest_thread = None; self._ingest_worker = None
        if self._infer_thread:
            self._infer_thread.quit(); self._infer_thread.wait(2000)
            self._infer_thread = None; self._inference_worker = None
            
        self._on_status_msg("Idle")
        self.lbl_partial.setText("")
        self.overlay.hide()
        self.waveform.set_speaking(False)

    def _save_text_to_settings(self):
        QSettings("WhisperDictate", "Pro").setValue("doc", self.editor.toPlainText())
        # Briefly flash "Saved" indicator
        orig_text = self.lbl_word_count.text()
        if "Saved" not in orig_text:
            self.lbl_word_count.setText(f"{orig_text} | Saved!")
            QTimer.singleShot(2000, lambda: self.lbl_word_count.setText(orig_text))

    def _save_settings(self):
        s = QSettings("WhisperDictate", "Pro")
        s.setValue("mic_idx", self.cmb_mic.currentData())
        s.setValue("model", self.cmb_model.currentText())
        s.setValue("lang", self.cmb_lang.currentData())
        s.setValue("systemwide", self.chk_systemwide.isChecked())
        s.setValue("fast_type", self.chk_fast_type.isChecked())
        s.setValue("clean", self.chk_clean.isChecked())
        s.setValue("timestamps", self.chk_timestamps.isChecked())
        s.setValue("overlay", self.chk_overlay.isChecked())
        s.setValue("translate", self.chk_translate.isChecked())
        s.setValue("beam", self.chk_beam.isChecked())
        s.setValue("ptt", self.chk_ptt.isChecked())
        s.setValue("vad", self.sld_vad.value())
        s.setValue("resp", self.sld_resp.value())
        s.setValue("hotwords", self.edt_hotwords.text())
        s.setValue("doc", self.editor.toPlainText())

    def _restore_settings(self):
        s = QSettings("WhisperDictate", "Pro")
        if s.contains("mic_idx"):
            i = self.cmb_mic.findData(s.value("mic_idx"))
            if i >= 0: self.cmb_mic.setCurrentIndex(i)
        if s.contains("model"): self.cmb_model.setCurrentText(s.value("model"))
        if s.contains("lang"):
            i = self.cmb_lang.findData(s.value("lang"))
            if i >= 0: self.cmb_lang.setCurrentIndex(i)
        self.chk_systemwide.setChecked(s.value("systemwide", False, type=bool))
        self.chk_fast_type.setChecked(s.value("fast_type", True, type=bool))
        self.chk_clean.setChecked(s.value("clean", True, type=bool))
        self.chk_timestamps.setChecked(s.value("timestamps", False, type=bool))
        self.chk_overlay.setChecked(s.value("overlay", True, type=bool))
        self.chk_translate.setChecked(s.value("translate", False, type=bool))
        self.chk_beam.setChecked(s.value("beam", False, type=bool))
        self.chk_ptt.setChecked(s.value("ptt", False, type=bool))
        self.sld_vad.setValue(s.value("vad", 60, type=int))
        self.sld_resp.setValue(s.value("resp", 30, type=int))
        self.edt_hotwords.setText(s.value("hotwords", "", type=str))
        if s.contains("doc"): self.editor.setPlainText(s.value("doc"))
        cmds = s.value("custom_commands", {}, type=dict)
        self._custom_macros.update(cmds)

    def closeEvent(self, e: QCloseEvent):
        self._stop_recording()
        self._save_settings()
        if hasattr(self, '_listener') and self._listener: self._listener.stop()
        if hasattr(self, '_ptt_listener') and self._ptt_listener: self._ptt_listener.stop()
        super().closeEvent(e)

# =====================================================================
# Entry Point
# =====================================================================
def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#1e1e1e"))
    pal.setColor(QPalette.WindowText, QColor("#e8e8e8"))
    pal.setColor(QPalette.Base, QColor("#101418"))
    pal.setColor(QPalette.Text, QColor("#e8e8e8"))
    pal.setColor(QPalette.Button, QColor("#262a30"))
    pal.setColor(QPalette.ButtonText, QColor("#e8e8e8"))
    app.setPalette(pal)

    win = DictationWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()