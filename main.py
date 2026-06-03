import os
import sys
import json
import time
import queue
import wave
import shutil
import logging
import tempfile
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, QUrl
from PySide6.QtGui import QAction, QKeySequence, QTextCursor, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QSpinBox, QGroupBox, QGridLayout, QLineEdit, QListWidget, QDialog,
    QProgressBar
)
from docx import Document
from docx.shared import Pt

APP_NAME = "VoiceDictationSTT"
DEVELOPER_NAME = "UnicornisIT"
TELEGRAM_CHANNEL = "https://t.me/unicornis_pulse"
DEVELOPER_EMAIL = "earov18@gmail.com"
APP_ICON_FILE = "app_icon.ico"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
MAX_PENDING_TRANSCRIPTION_CHUNKS = 1

try:
    APP_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / APP_NAME
    APP_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    APP_DIR = Path(tempfile.gettempdir()) / APP_NAME
    APP_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = APP_DIR / "app.log"
SETTINGS_FILE = APP_DIR / "settings.json"
AUTOSAVE_FILE = APP_DIR / "autosave_transcript.txt"
HISTORY_FILE = APP_DIR / "save_history.json"
TMP_AUDIO_DIR = APP_DIR / "tmp_audio"
TMP_AUDIO_DIR.mkdir(exist_ok=True)
LOCAL_MODELS_DIR = APP_DIR / "models"
LOCAL_MODELS_DIR.mkdir(exist_ok=True)


def resource_path(filename: str) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_dir / filename


def app_icon() -> QIcon:
    icon_path = resource_path(APP_ICON_FILE)
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


def set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{DEVELOPER_NAME}.{APP_NAME}")
    except Exception:
        logging.exception("Failed to set Windows AppUserModelID")

try:
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
        encoding="utf-8",
    )
except OSError:
    LOG_FILE = Path(tempfile.gettempdir()) / f"{APP_NAME}.log"
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
        encoding="utf-8",
    )

FILLER_WORDS_RU = [
    "ну", "как бы", "типа", "это самое", "значит", "короче", "в общем", "собственно",
    "э-э", "ээ", "эм", "мм"
]
FILLER_WORDS_EN = ["um", "uh", "like", "you know", "sort of", "kind of"]


@dataclass
class AppSettings:
    language: str = "ru"  # ru, en, auto
    accuracy_mode: str = "balanced"  # fast, balanced, max
    autopunctuation: bool = True
    spell_correction: bool = False
    remove_fillers: bool = False
    autosave_enabled: bool = True
    autosave_interval_sec: int = 10
    save_folder: str = str(Path.home() / "Documents")
    dark_theme: bool = False
    timestamps: bool = False
    microphone_index: Optional[int] = None


def load_settings() -> AppSettings:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return AppSettings(**{**asdict(AppSettings()), **data})
        except Exception:
            logging.exception("Failed to load settings")
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    SETTINGS_FILE.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(path: str) -> None:
    history = []
    try:
        if HISTORY_FILE.exists():
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        logging.exception("Failed to read history")
    if path in history:
        history.remove(path)
    history.insert(0, path)
    history = history[:10]
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history() -> List[str]:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))[:10]
    except Exception:
        logging.exception("Failed to load history")
    return []


def model_size_for_mode(mode: str) -> str:
    # Для работы на CPU задержка сильно зависит от размера модели.
    # Поэтому режимы настроены так, чтобы диктовка была практичной на обычном ПК:
    # fast -> минимальная задержка, balanced -> нормальная точность, max -> медленнее, но точнее.
    return {
        "fast": "tiny",
        "balanced": "small",
        "max": "medium",
    }.get(mode, "small")

def model_repo_for_size(model_size: str) -> str:
    return f"Systran/faster-whisper-{model_size}"


def model_local_dir(model_size: str) -> Path:
    return LOCAL_MODELS_DIR / f"faster-whisper-{model_size}"


def model_complete_marker(model_size: str) -> Path:
    return model_local_dir(model_size) / ".complete"


def is_model_downloaded(model_size: str) -> bool:
    """Проверяет наличие модели в локальной папке приложения без выхода в интернет."""
    model_dir = model_local_dir(model_size)
    return (
        model_complete_marker(model_size).exists()
        and (model_dir / "config.json").exists()
        and any((model_dir / name).exists() for name in ("model.bin", "model.bin.index.json"))
    )


def cuda_is_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        logging.exception("Failed to check CUDA availability")
        return False


def human_kb(value: int) -> str:
    return f"{value / 1024:.0f} КБ"



def clean_text(text: str, remove_fillers: bool, language: str) -> str:
    cleaned = " ".join(text.split())
    if remove_fillers:
        words = FILLER_WORDS_RU if language in ("ru", "auto") else FILLER_WORDS_EN
        for w in sorted(words, key=len, reverse=True):
            cleaned = cleaned.replace(f" {w} ", " ")
            cleaned = cleaned.replace(f" {w},", "")
            cleaned = cleaned.replace(f" {w}.", ".")
    return cleaned.strip()


def maybe_capitalize(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -100.0
    rms = np.sqrt(np.mean(np.square(audio.astype(np.float64))))
    return 20 * np.log10(max(rms, 1e-9))


def normalize_audio(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    peak = np.max(np.abs(audio)) if audio.size else 0
    if peak <= 1e-6:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


class RecorderWorker(QObject):
    status = Signal(str)
    audio_ready = Signal(str, float)
    warning = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, microphone_index: Optional[int], chunk_sec: int = 12):
        super().__init__()
        self.microphone_index = microphone_index
        self.chunk_sec = chunk_sec
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.clear()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def request_stop(self):
        self._stop.set()

    def request_pause(self, paused: bool):
        if paused:
            self._pause.set()
        else:
            self._pause.clear()

    def _callback(self, indata, frames, time_info, status):
        if status:
            logging.warning("Audio callback status: %s", status)
        if not self._pause.is_set():
            self._audio_queue.put(indata.copy().reshape(-1))

    def run(self):
        try:
            self.status.emit("Идёт запись")
            blocksize = int(SAMPLE_RATE * 0.5)
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=blocksize,
                device=self.microphone_index,
                callback=self._callback,
            )
            with self._stream:
                buffer: List[np.ndarray] = []
                frames_target = SAMPLE_RATE * self.chunk_sec
                last_emit = time.time()
                while not self._stop.is_set():
                    try:
                        data = self._audio_queue.get(timeout=0.2)
                        buffer.append(data)
                    except queue.Empty:
                        continue
                    frames = sum(x.shape[0] for x in buffer)
                    if frames >= frames_target:
                        audio = normalize_audio(np.concatenate(buffer))
                        buffer = []
                        db = rms_db(audio)
                        if db < -50:
                            logging.info("Silent chunk skipped: level %.1f dB", db)
                            last_emit = time.time()
                            continue
                        if db < -38:
                            self.warning.emit("Низкий уровень сигнала микрофона. Распознавание может быть неточным.")
                        path = TMP_AUDIO_DIR / f"chunk_{int(time.time()*1000)}.wav"
                        write_wav(path, audio)
                        logging.info("Audio chunk ready: %s, level %.1f dB", path, db)
                        self.audio_ready.emit(str(path), db)
                        last_emit = time.time()
                if buffer:
                    audio = normalize_audio(np.concatenate(buffer))
                    if audio.size > SAMPLE_RATE * 0.5:
                        db = rms_db(audio)
                        if db >= -50:
                            path = TMP_AUDIO_DIR / f"final_{int(time.time()*1000)}.wav"
                            write_wav(path, audio)
                            logging.info("Final audio chunk ready: %s, level %.1f dB", path, db)
                            self.audio_ready.emit(str(path), db)
                        else:
                            logging.info("Silent final chunk skipped: level %.1f dB", db)
        except Exception as exc:
            logging.exception("Recorder error")
            self.error.emit(f"Ошибка микрофона: {exc}")
        finally:
            self.finished.emit()


class ModelDownloadWorker(QObject):
    stage = Signal(str)
    progress = Signal(int, object, object, str)  # percent, downloaded_bytes, total_bytes, current_file
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, model_size: str):
        super().__init__()
        self.model_size = model_size

    def run(self):
        repo_id = model_repo_for_size(self.model_size)
        target_dir = model_local_dir(self.model_size)
        tmp_dir = target_dir.with_name(target_dir.name + ".partial")
        try:
            import requests
            from huggingface_hub import HfApi, hf_hub_url

            self.stage.emit(f"Получение списка файлов модели: {self.model_size}")
            api = HfApi()
            info = api.model_info(repo_id, files_metadata=True)
            siblings = [s for s in info.siblings if getattr(s, "rfilename", None)]
            files = []
            for item in siblings:
                filename = item.rfilename
                if filename.endswith(".msgpack") or filename.endswith(".h5"):
                    continue
                size = getattr(item, "size", None) or 0
                files.append((filename, int(size)))

            if not files:
                raise RuntimeError("Hugging Face не вернул список файлов модели.")

            total_bytes = sum(size for _, size in files)
            if total_bytes <= 0:
                total_bytes = 1

            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            downloaded_total = 0
            self.stage.emit(
                f"Скачивание модели {self.model_size}. Общий размер: {human_kb(total_bytes)}."
            )
            self.progress.emit(0, 0, total_bytes, "Подготовка")

            for filename, expected_size in files:
                url = hf_hub_url(repo_id, filename)
                destination = tmp_dir / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                current_file_done = 0
                self.stage.emit(f"Скачивание файла: {filename}")

                with requests.get(url, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    if expected_size <= 0:
                        expected_size = int(response.headers.get("Content-Length", "0") or 0)
                    with open(destination, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=1024 * 512):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            chunk_len = len(chunk)
                            current_file_done += chunk_len
                            downloaded_now = downloaded_total + current_file_done
                            percent = int(min(100, downloaded_now * 100 / max(total_bytes, 1)))
                            self.progress.emit(percent, downloaded_now, total_bytes, filename)

                # Если размер файла был неизвестен в metadata, корректируем общий счетчик.
                if expected_size <= 0:
                    expected_size = destination.stat().st_size
                    total_bytes += expected_size
                downloaded_total += expected_size
                percent = int(min(100, downloaded_total * 100 / max(total_bytes, 1)))
                self.progress.emit(percent, downloaded_total, total_bytes, filename)

            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            tmp_dir.rename(target_dir)
            model_complete_marker(self.model_size).write_text(
                f"downloaded_at={datetime.now().isoformat()}\nrepo={repo_id}\n",
                encoding="utf-8",
            )
            self.progress.emit(100, total_bytes, total_bytes, "Готово")
            logging.info("Model downloaded with progress: %s -> %s", repo_id, target_dir)
            self.finished.emit(str(target_dir))
        except Exception as exc:
            logging.exception("Model download failed")
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                logging.exception("Failed to clean partial model directory")
            self.error.emit(str(exc))


class ModelDownloadDialog(QDialog):
    def __init__(self, model_size: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Загрузка модели распознавания")
        self.setWindowIcon(app_icon())
        self.setModal(True)
        self.setMinimumWidth(620)
        self._finished_ok = False
        self.thread = QThread(self)
        self.worker = ModelDownloadWorker(model_size)
        self.worker.moveToThread(self.thread)

        layout = QVBoxLayout(self)
        self.title_label = QLabel(f"Модель ещё не скачана: {model_size}")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.info_label = QLabel(
            "Для первой расшифровки нужно скачать модель Faster-Whisper.\n"
            "После загрузки она сохранится локально в папке приложения и повторно скачиваться обычно не будет."
        )
        self.info_label.setWordWrap(True)
        self.status_label = QLabel("Подготовка...")
        self.status_label.setWordWrap(True)
        self.progress_label = QLabel("0% | 0 КБ / 0 КБ")
        self.current_file_label = QLabel("Файл: —")
        self.current_file_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.setEnabled(False)

        layout.addWidget(self.title_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.current_file_label)
        layout.addWidget(self.btn_close, alignment=Qt.AlignRight)

        self.thread.started.connect(self.worker.run)
        self.worker.stage.connect(self.status_label.setText)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.btn_close.clicked.connect(self.accept)
        self.thread.start()

    def _on_progress(self, percent: int, downloaded_bytes: int, total_bytes: int, current_file: str):
        self.progress_bar.setValue(max(0, min(100, int(percent))))
        self.progress_label.setText(
            f"{int(percent)}% | {human_kb(int(downloaded_bytes))} / {human_kb(int(total_bytes))}"
        )
        self.current_file_label.setText(f"Файл: {current_file}")

    def _on_finished(self, path: str):
        self._finished_ok = True
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Модель успешно скачана.\n{path}")
        self.progress_label.setText("100% | загрузка завершена")
        self.current_file_label.setText("Файл: готово")
        self.btn_close.setText("Продолжить")
        self.btn_close.setEnabled(True)

    def _on_error(self, message: str):
        self.progress_bar.setValue(0)
        self.status_label.setText(
            "Не удалось скачать модель. Проверьте интернет, доступ к Hugging Face, "
            "свободное место на диске и антивирус/прокси.\n\n"
            f"Техническая ошибка: {message}"
        )
        self.btn_close.setText("Закрыть")
        self.btn_close.setEnabled(True)

    def closeEvent(self, event):
        if self.thread.isRunning() and not self._finished_ok:
            QMessageBox.warning(
                self,
                "Загрузка модели",
                "Модель ещё скачивается. Дождитесь завершения загрузки или закройте программу после окончания процесса."
            )
            event.ignore()
            return
        super().closeEvent(event)


class TranscriberWorker(QObject):
    status = Signal(str)
    text_ready = Signal(str)
    error = Signal(str)
    model_ready = Signal(str)
    finished = Signal()

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._model = None
        self._model_runtime = ""

    def _discard_pending_chunks(self) -> int:
        dropped = 0
        while True:
            try:
                old_path = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                os.remove(old_path)
                logging.info("Dropped pending audio chunk: %s", old_path)
            except OSError:
                logging.exception("Failed to remove pending audio chunk: %s", old_path)
            dropped += 1
        return dropped

    def enqueue(self, wav_path: str):
        # Распознавание может быть медленнее записи, особенно на CPU.
        # Ограничиваем очередь: старые фрагменты уже менее актуальны для живой диктовки
        # и иначе будут бесконечно копиться, нагружая CPU/диск и задерживая текст.
        while self._queue.qsize() >= MAX_PENDING_TRANSCRIPTION_CHUNKS:
            try:
                old_path = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                os.remove(old_path)
                logging.info("Dropped stale audio chunk from transcription queue: %s", old_path)
            except OSError:
                logging.exception("Failed to remove stale audio chunk: %s", old_path)
        self._queue.put(wav_path)

    def request_stop(self, drop_pending: bool = False):
        self._stop.set()
        if drop_pending:
            # При закрытии или повторной остановке важнее быстро завершить поток,
            # чем дообрабатывать устаревшие WAV-фрагменты в фоне.
            dropped = self._discard_pending_chunks()
            if dropped:
                logging.info("Dropped %d queued chunks during transcription stop", dropped)

    def _load_model(self, WhisperModel, model_path: Path):
        if cuda_is_available():
            for compute_type in ("int8_float16", "float16"):
                try:
                    self.status.emit(f"Загрузка модели на CUDA ({compute_type})")
                    model = WhisperModel(str(model_path), device="cuda", compute_type=compute_type)
                    logging.info("Faster-Whisper model initialized on CUDA with %s", compute_type)
                    return model, f"CUDA {compute_type}"
                except Exception:
                    # На части систем CUDA видна, но конкретный compute_type или runtime
                    # может не стартовать. Тогда пробуем более простой CUDA-режим,
                    # а затем откатываемся на CPU, чтобы приложение работало без видеокарты.
                    logging.exception("Failed to initialize Faster-Whisper on CUDA with %s", compute_type)

            self.status.emit("CUDA не запустилась, fallback на CPU int8")
            logging.warning("CUDA initialization failed, falling back to CPU int8")

        self.status.emit("Загрузка модели на CPU (int8)")
        model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
        logging.info("Faster-Whisper model initialized on CPU with int8")
        return model, "CPU int8"

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            model_size = model_size_for_mode(self.settings.accuracy_mode)
            if not is_model_downloaded(model_size):
                raise RuntimeError(
                    f"Модель {model_size} ещё не скачана. "
                    "Запустите диктовку повторно и дождитесь окна загрузки модели."
                )
            self.status.emit(f"Загрузка модели {model_size} в память")
            self._model, self._model_runtime = self._load_model(WhisperModel, model_local_dir(model_size))
            self.model_ready.emit(f"Модель загружена: {model_size} ({self._model_runtime})")
        except Exception as exc:
            logging.exception("Failed to load Whisper model")
            raise RuntimeError(
                "Не удалось загрузить Faster-Whisper. Проверьте, что модель скачана, "
                "установлены зависимости и есть свободное место на диске."
            ) from exc

    def run(self):
        try:
            self.status.emit("Обработка")
            self._ensure_model()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    wav_path = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    language = None if self.settings.language == "auto" else self.settings.language
                    # Для коротких чанков condition_on_previous_text=True часто даёт "прилипание"
                    # и повторение старого контекста. Для диктовки по фрагментам надёжнее False.
                    # beam_size=1 заметно снижает задержку для потоковой диктовки; более широкий
                    # поиск повышает нагрузку и легко создаёт хвост из необработанных WAV.
                    beam_size = 1
                    # VAD отбрасывает паузы и тишину до тяжёлого распознавания, чтобы модель
                    # не тратила время на пустые участки и очередь не росла без пользы.
                    vad_filter = True
                    segments, info = self._model.transcribe(
                        wav_path,
                        language=language,
                        beam_size=beam_size,
                        vad_filter=vad_filter,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=False,
                        temperature=0.0,
                        no_speech_threshold=0.65,
                        compression_ratio_threshold=2.4,
                    )
                    parts = []
                    for seg in segments:
                        if getattr(seg, "no_speech_prob", 0.0) > 0.80:
                            logging.info("Segment skipped as no speech: prob=%.2f text=%r", getattr(seg, "no_speech_prob", 0.0), seg.text)
                            continue
                        parts.append(seg.text.strip())
                    text = " ".join(parts)
                    detected_lang = getattr(info, "language", self.settings.language)
                    text = clean_text(text, self.settings.remove_fillers, detected_lang)
                    if not self.settings.autopunctuation:
                        # Whisper всё равно может отдавать знаки; это упрощённый режим без пунктуации.
                        for ch in ".,!?;:—-()[]{}\"«»":
                            text = text.replace(ch, "")
                    text = maybe_capitalize(text)
                    if text:
                        if self.settings.timestamps:
                            prefix = datetime.now().strftime("[%H:%M:%S] ")
                            text = prefix + text
                        self.text_ready.emit(text)
                    else:
                        logging.info("No text recognized from chunk: %s", wav_path)
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                except Exception as exc:
                    logging.exception("Transcription error")
                    self.error.emit(f"Ошибка распознавания: {exc}")
            self.status.emit("Ожидание")
        except Exception as exc:
            logging.exception("Transcriber fatal error")
            self.error.emit(str(exc))
            self.status.emit("Ожидание")
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle("Голосовая диктовка и расшифровка")
        self.setWindowIcon(app_icon())
        
        self.resize(1100, 760)

        self.rec_thread: Optional[QThread] = None
        self.rec_worker: Optional[RecorderWorker] = None
        self.tr_thread: Optional[QThread] = None
        self.tr_worker: Optional[TranscriberWorker] = None
        self.paused = False
        self._pending_start_after_download = False
        self._recorder_started = False
        self._stopping = False
        self._closing_after_stop = False

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(max(3, self.settings.autosave_interval_sec) * 1000)

        self._build_ui()
        self._build_menu()
        self._load_devices()
        self._apply_theme()
        self._connect_shortcuts()

        if AUTOSAVE_FILE.exists():
            try:
                text = AUTOSAVE_FILE.read_text(encoding="utf-8")
                if text.strip():
                    self.text_edit.setPlainText(text)
            except Exception:
                logging.exception("Failed to restore autosave")
        self.update_counts()
        self.refresh_history()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.btn_start = QPushButton("Начать диктовку")
        self.btn_pause = QPushButton("Пауза")
        self.btn_stop = QPushButton("Остановить")
        self.btn_clear = QPushButton("Очистить текст")
        self.btn_copy = QPushButton("Копировать текст")
        self.btn_txt = QPushButton("Экспорт в TXT")
        self.btn_docx = QPushButton("Экспорт в DOCX")
        for b in [self.btn_start, self.btn_pause, self.btn_stop, self.btn_clear, self.btn_copy, self.btn_txt, self.btn_docx]:
            top.addWidget(b)
        layout.addLayout(top)

        status_line = QHBoxLayout()
        self.status_label = QLabel("Ожидание")
        self.status_label.setStyleSheet("font-weight: bold;")
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #b36b00;")
        self.counter_label = QLabel("Слов: 0 | Символов: 0")
        status_line.addWidget(QLabel("Состояние:"))
        status_line.addWidget(self.status_label)
        status_line.addStretch(1)
        status_line.addWidget(self.warning_label)
        status_line.addStretch(1)
        status_line.addWidget(self.counter_label)
        layout.addLayout(status_line)

        self.text_edit = QTextEdit()
        self.text_edit.setAcceptRichText(False)
        self.text_edit.textChanged.connect(self.update_counts)
        layout.addWidget(self.text_edit, 1)

        settings_group = QGroupBox("Настройки распознавания")
        grid = QGridLayout(settings_group)

        self.microphone_combo = QComboBox()
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Русский", "Английский", "Автоопределение языка"])
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Быстрый", "Сбалансированный", "Максимальная точность"])

        self.chk_punct = QCheckBox("Автоматическая пунктуация")
        self.chk_spell = QCheckBox("Исправление орфографии")
        self.chk_fillers = QCheckBox("Удаление слов-паразитов")
        self.chk_autosave = QCheckBox("Автосохранение")
        self.chk_dark = QCheckBox("Тёмная тема")
        self.chk_timestamps = QCheckBox("Вставлять временные метки")

        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(3, 600)
        self.autosave_spin.setSuffix(" сек")
        self.folder_edit = QLineEdit(self.settings.save_folder)
        self.btn_folder = QPushButton("Выбрать папку")
        self.btn_download_model = QPushButton("Скачать / проверить модель")

        grid.addWidget(QLabel("Микрофон:"), 0, 0)
        grid.addWidget(self.microphone_combo, 0, 1, 1, 3)
        grid.addWidget(QLabel("Язык:"), 1, 0)
        grid.addWidget(self.language_combo, 1, 1)
        grid.addWidget(QLabel("Режим точности:"), 1, 2)
        grid.addWidget(self.mode_combo, 1, 3)
        grid.addWidget(self.chk_punct, 2, 0)
        grid.addWidget(self.chk_spell, 2, 1)
        grid.addWidget(self.chk_fillers, 2, 2)
        grid.addWidget(self.chk_timestamps, 2, 3)
        grid.addWidget(self.chk_autosave, 3, 0)
        grid.addWidget(self.autosave_spin, 3, 1)
        grid.addWidget(self.chk_dark, 3, 2)
        grid.addWidget(QLabel("Папка сохранения:"), 4, 0)
        grid.addWidget(self.folder_edit, 4, 1, 1, 2)
        grid.addWidget(self.btn_folder, 4, 3)
        grid.addWidget(self.btn_download_model, 5, 0, 1, 4)
        layout.addWidget(settings_group)

        history_group = QGroupBox("История последних сохранённых файлов")
        h_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        h_layout.addWidget(self.history_list)
        layout.addWidget(history_group)

        self.btn_start.clicked.connect(self.start_dictation)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_dictation)
        self.btn_clear.clicked.connect(self.clear_text)
        self.btn_copy.clicked.connect(self.copy_text)
        self.btn_txt.clicked.connect(self.export_txt)
        self.btn_docx.clicked.connect(self.export_docx)
        self.btn_folder.clicked.connect(self.choose_folder)
        self.btn_download_model.clicked.connect(self.download_current_model)
        self.chk_dark.stateChanged.connect(self.on_settings_changed)
        for widget in [self.language_combo, self.mode_combo, self.chk_punct, self.chk_spell, self.chk_fillers,
                       self.chk_autosave, self.autosave_spin, self.chk_timestamps, self.microphone_combo]:
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.on_settings_changed)
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self.on_settings_changed)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.on_settings_changed)
        self.folder_edit.textChanged.connect(self.on_settings_changed)

        self._settings_to_ui()
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def _settings_to_ui(self):
        self.language_combo.setCurrentIndex({"ru": 0, "en": 1, "auto": 2}.get(self.settings.language, 0))
        self.mode_combo.setCurrentIndex({"fast": 0, "balanced": 1, "max": 2}.get(self.settings.accuracy_mode, 1))
        self.chk_punct.setChecked(self.settings.autopunctuation)
        self.chk_spell.setChecked(self.settings.spell_correction)
        self.chk_fillers.setChecked(self.settings.remove_fillers)
        self.chk_autosave.setChecked(self.settings.autosave_enabled)
        self.autosave_spin.setValue(self.settings.autosave_interval_sec)
        self.chk_dark.setChecked(self.settings.dark_theme)
        self.chk_timestamps.setChecked(self.settings.timestamps)

    def _ui_to_settings(self):
        self.settings.language = ["ru", "en", "auto"][self.language_combo.currentIndex()]
        self.settings.accuracy_mode = ["fast", "balanced", "max"][self.mode_combo.currentIndex()]
        self.settings.autopunctuation = self.chk_punct.isChecked()
        self.settings.spell_correction = self.chk_spell.isChecked()
        self.settings.remove_fillers = self.chk_fillers.isChecked()
        self.settings.autosave_enabled = self.chk_autosave.isChecked()
        self.settings.autosave_interval_sec = self.autosave_spin.value()
        self.settings.save_folder = self.folder_edit.text().strip() or str(Path.home() / "Documents")
        self.settings.dark_theme = self.chk_dark.isChecked()
        self.settings.timestamps = self.chk_timestamps.isChecked()
        data = self.microphone_combo.currentData()
        self.settings.microphone_index = data if data is not None else None

    def on_settings_changed(self):
        self._ui_to_settings()
        save_settings(self.settings)
        self._apply_theme()
        if hasattr(self, "autosave_timer"):
            self.autosave_timer.setInterval(max(3, self.settings.autosave_interval_sec) * 1000)

    def _connect_shortcuts(self):
        act_start = QAction(self)
        act_start.setShortcut(QKeySequence("Ctrl+R"))
        act_start.triggered.connect(self.start_dictation)
        self.addAction(act_start)
        act_pause = QAction(self)
        act_pause.setShortcut(QKeySequence("Ctrl+P"))
        act_pause.triggered.connect(self.toggle_pause)
        self.addAction(act_pause)
        act_save = QAction(self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self.export_txt)
        self.addAction(act_save)

    def _build_menu(self):
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Справка")
        about_action = help_menu.addAction("О программе")
        about_action.triggered.connect(self.show_about_dialog)

    def _apply_theme(self):
        if self.settings.dark_theme:
            self.setStyleSheet("""
                QWidget { background: #1f1f1f; color: #f0f0f0; }
                QTextEdit, QComboBox, QLineEdit, QListWidget { background: #2b2b2b; color: #f0f0f0; border: 1px solid #555; }
                QPushButton { background: #333; color: #f0f0f0; border: 1px solid #666; padding: 6px; }
                QPushButton:hover { background: #444; }
                QGroupBox { border: 1px solid #555; margin-top: 8px; padding-top: 12px; }
            """)
        else:
            self.setStyleSheet("")

    def _load_devices(self):
        self.microphone_combo.clear()
        try:
            devices = sd.query_devices()
            default_in = None
            try:
                default_in = sd.default.device[0]
            except Exception:
                pass
            selected_index = self.settings.microphone_index
            current_combo_index = 0
            added = 0
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    name = f"{idx}: {dev.get('name', 'Микрофон')}"
                    self.microphone_combo.addItem(name, idx)
                    if selected_index == idx or (selected_index is None and idx == default_in):
                        current_combo_index = added
                    added += 1
            if added == 0:
                self.microphone_combo.addItem("Микрофон не найден", None)
                self.status_label.setText("Ошибка микрофона")
            else:
                self.microphone_combo.setCurrentIndex(current_combo_index)
        except Exception as exc:
            logging.exception("Failed to query devices")
            self.microphone_combo.addItem("Ошибка получения микрофонов", None)
            self.status_label.setText("Ошибка микрофона")
            QMessageBox.warning(self, "Микрофон", f"Не удалось получить список микрофонов: {exc}")

    def start_dictation(self):
        if self.rec_thread is not None or self.tr_thread is not None or self._stopping:
            return
        self.on_settings_changed()
        if self.settings.microphone_index is None:
            self.status_label.setText("Ошибка микрофона")
            QMessageBox.warning(self, "Микрофон", "Микрофон не выбран или не найден.")
            return

        model_size = model_size_for_mode(self.settings.accuracy_mode)
        if not is_model_downloaded(model_size):
            self.status_label.setText("Модель не скачана")
            QMessageBox.information(
                self,
                "Модель не скачана",
                f"Модель распознавания {model_size} ещё не скачана. "
                "Сейчас откроется окно загрузки. Диктовку можно начинать после завершения загрузки."
            )
            ok = self.download_current_model(show_success=True)
            if not ok:
                return

        self.warning_label.setText("")
        self.paused = False
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

        self.status_label.setText("Загрузка модели в память")
        self._recorder_started = False

        self.tr_thread = QThread(self)
        self.tr_worker = TranscriberWorker(self.settings)
        self.tr_worker.moveToThread(self.tr_thread)
        self.tr_thread.started.connect(self.tr_worker.run)
        self.tr_worker.text_ready.connect(self.append_transcript)
        self.tr_worker.status.connect(self._set_status_safely)
        self.tr_worker.error.connect(self.show_error)
        self.tr_worker.model_ready.connect(self._start_recorder_after_model_loaded)
        self.tr_worker.finished.connect(self.tr_thread.quit)
        self.tr_worker.finished.connect(self._on_transcriber_finished)
        self.tr_worker.finished.connect(self.tr_worker.deleteLater)
        self.tr_thread.finished.connect(self.tr_thread.deleteLater)
        self.tr_thread.start()

    def download_current_model(self, show_success: bool = True) -> bool:
        self.on_settings_changed()
        model_size = model_size_for_mode(self.settings.accuracy_mode)
        if is_model_downloaded(model_size):
            self.status_label.setText("Модель скачана")
            if show_success:
                QMessageBox.information(self, "Модель", f"Модель {model_size} уже скачана и готова к работе.")
            return True

        dialog = ModelDownloadDialog(model_size, self)
        dialog.exec()
        if dialog._finished_ok:
            self.status_label.setText("Модель скачана")
            if show_success:
                QMessageBox.information(self, "Модель", "Модель успешно скачана. Можно начинать диктовку.")
            return True

        self.status_label.setText("Модель не скачана")
        QMessageBox.warning(
            self,
            "Модель не скачана",
            "Модель распознавания ещё не скачана. Диктовка не запущена."
        )
        return False

    def _set_status_safely(self, msg: str):
        if self._stopping and msg not in ("Ожидание", "Завершение распознавания"):
            return
        # Не даём фоновому распознавателю показывать «Ожидание», пока запись реально идёт.
        if msg == "Ожидание" and self.rec_worker is not None:
            return
        self.status_label.setText(msg)

    def _chunk_seconds_for_mode(self) -> int:
        return {"fast": 2, "balanced": 4, "max": 6}.get(self.settings.accuracy_mode, 4)

    def _start_recorder_after_model_loaded(self, msg: str):
        logging.info(msg)
        if self._recorder_started or self.tr_worker is None:
            return
        self._recorder_started = True
        self.status_label.setText("Идёт запись")

        self.rec_thread = QThread(self)
        self.rec_worker = RecorderWorker(self.settings.microphone_index, chunk_sec=self._chunk_seconds_for_mode())
        self.rec_worker.moveToThread(self.rec_thread)
        self.rec_thread.started.connect(self.rec_worker.run)
        self.rec_worker.status.connect(self.status_label.setText)
        self.rec_worker.audio_ready.connect(
            lambda wav_path, db: self.tr_worker.enqueue(wav_path) if self.tr_worker else None,
            Qt.DirectConnection
        )
        self.rec_worker.audio_ready.connect(self.on_audio_ready_debug)
        self.rec_worker.warning.connect(self.show_warning)
        self.rec_worker.error.connect(self.show_error)
        self.rec_worker.finished.connect(self.rec_thread.quit)
        self.rec_worker.finished.connect(self._on_recorder_finished)
        self.rec_worker.finished.connect(self.rec_worker.deleteLater)
        self.rec_thread.finished.connect(self.rec_thread.deleteLater)
        self.rec_thread.start()


    def toggle_pause(self):
        if not self.rec_worker:
            return
        self.paused = not self.paused
        self.rec_worker.request_pause(self.paused)
        self.status_label.setText("Пауза" if self.paused else "Идёт запись")
        self.btn_pause.setText("Продолжить" if self.paused else "Пауза")

    def stop_dictation(self):
        if self.rec_worker is None and self.tr_worker is None:
            self.status_label.setText("Ожидание")
            return
        if self._stopping:
            return

        self._stopping = True
        self.status_label.setText("Остановка записи")
        self.warning_label.setText("")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)

        # Не вызываем QThread.wait() из UI-потока: пока Faster-Whisper завершает
        # текущий transcribe(), Windows считает неподвижное окно зависшим.
        # Вместо этого просим потоки остановиться и ждём их finished-сигналы.
        if self.rec_worker:
            self.rec_worker.request_stop()
        else:
            self._request_transcriber_stop(drop_pending=True)
        self._finish_stop_if_done()

    def _request_transcriber_stop(self, drop_pending: bool = False):
        if self.tr_worker:
            self.status_label.setText("Завершение распознавания")
            self.tr_worker.request_stop(drop_pending=drop_pending)

    def _on_recorder_finished(self):
        self.rec_worker = None
        self.rec_thread = None
        self._recorder_started = False
        if not self._stopping:
            self._stopping = True
        # После остановки микрофона новых WAV уже не будет. Разрешаем распознавателю
        # обработать максимум оставшийся актуальный фрагмент и затем завершиться.
        self._request_transcriber_stop(drop_pending=self._closing_after_stop)
        self._finish_stop_if_done()

    def _on_transcriber_finished(self):
        self.tr_worker = None
        self.tr_thread = None
        if self.rec_worker is not None and not self._stopping:
            self._stopping = True
            self.status_label.setText("Остановка записи")
            self.rec_worker.request_stop()
            return
        self._finish_stop_if_done()

    def _reset_idle_controls(self):
        self._stopping = False
        self._recorder_started = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("Пауза")

    def _finish_stop_if_done(self):
        if self.rec_worker is not None or self.tr_worker is not None:
            return
        if not self._stopping:
            self._reset_idle_controls()
            return
        self._reset_idle_controls()
        self.status_label.setText("Ожидание")
        self.autosave()
        if self._closing_after_stop:
            self._closing_after_stop = False
            self.close()

    def on_audio_ready_debug(self, wav_path: str, db: float):
        logging.info("Audio chunk ready: %s, level %.1f dB", wav_path, db)

    def append_transcript(self, text: str):
        logging.info("Transcript appended: %s", text[:200])
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        existing = self.text_edit.toPlainText()
        separator = "\n" if existing.strip().endswith((".", "!", "?", ":")) else " "
        if not existing.strip():
            separator = ""
        cursor.insertText(separator + text)
        self.text_edit.setTextCursor(cursor)
        self.autosave()

    def show_warning(self, msg: str):
        self.warning_label.setText(msg)
        logging.warning(msg)

    def show_error(self, msg: str):
        if self._closing_after_stop or self._stopping:
            logging.error(msg)
            return
        self.status_label.setText("Ошибка микрофона" if "микроф" in msg.lower() else "Ожидание")
        logging.error(msg)
        QMessageBox.warning(self, "Ошибка", msg)

    def clear_text(self):
        if self.text_edit.toPlainText().strip():
            reply = QMessageBox.question(self, "Очистить текст", "Очистить текущую расшифровку?")
            if reply != QMessageBox.Yes:
                return
        self.text_edit.clear()
        self.autosave()

    def copy_text(self):
        text = self.text_edit.textCursor().selectedText() or self.text_edit.toPlainText()
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Копирование", "Текст скопирован в буфер обмена.")

    def update_counts(self):
        text = self.text_edit.toPlainText()
        words = len([w for w in text.split() if w.strip()])
        chars = len(text)
        self.counter_label.setText(f"Слов: {words} | Символов: {chars}")

    def autosave(self):
        if not self.settings.autosave_enabled:
            return
        try:
            AUTOSAVE_FILE.write_text(self.text_edit.toPlainText(), encoding="utf-8")
        except Exception:
            logging.exception("Autosave failed")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку сохранения", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self.on_settings_changed()

    def _get_save_path(self, suffix: str, filter_name: str) -> Optional[str]:
        folder = self.settings.save_folder if Path(self.settings.save_folder).exists() else str(Path.home())
        default_name = f"dictation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{suffix}"
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", str(Path(folder) / default_name), filter_name)
        return path or None

    def export_txt(self):
        path = self._get_save_path("txt", "Text files (*.txt)")
        if not path:
            return
        try:
            Path(path).write_text(self.text_edit.toPlainText(), encoding="utf-8")
            append_history(path)
            self.refresh_history()
            QMessageBox.information(self, "Экспорт", "TXT-файл успешно сохранён.")
        except Exception as exc:
            logging.exception("TXT export failed")
            QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось сохранить TXT: {exc}")

    def export_docx(self):
        path = self._get_save_path("docx", "Word documents (*.docx)")
        if not path:
            return
        try:
            document = Document()
            style = document.styles["Normal"]
            style.font.name = "Calibri"
            style.font.size = Pt(12)
            text = self.text_edit.toPlainText()
            paragraphs = text.split("\n") or [""]
            for p in paragraphs:
                document.add_paragraph(p)
            document.save(path)
            append_history(path)
            self.refresh_history()
            QMessageBox.information(self, "Экспорт", "DOCX-файл успешно сохранён.")
        except Exception as exc:
            logging.exception("DOCX export failed")
            QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось сохранить DOCX: {exc}")

    def refresh_history(self):
        self.history_list.clear()
        for item in load_history():
            self.history_list.addItem(item)

    def closeEvent(self, event):
        if self.rec_worker is not None or self.tr_worker is not None or self._stopping:
            self._closing_after_stop = True
            try:
                if not self._stopping:
                    self.stop_dictation()
                elif self.tr_worker:
                    self._request_transcriber_stop(drop_pending=True)
            except Exception:
                logging.exception("Failed to request stop while closing")
            event.ignore()
            return
        self.on_settings_changed()
        self.autosave()
        try:
            for p in TMP_AUDIO_DIR.glob("*.wav"):
                p.unlink(missing_ok=True)
        except Exception:
            logging.exception("Failed to clean temp audio")
        super().closeEvent(event)

    def show_about_dialog(self):
        AboutDialog(self).exec()


def main():
    # Для корректной работы PyInstaller с multiprocessing-зависимостями.
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass
    set_windows_app_id()
    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе")
        self.setWindowIcon(app_icon())
        self.setModal(True)
        self.resize(500, 300)
        
        layout = QVBoxLayout(self)
        
        # Заголовок
        title_label = QLabel(APP_NAME)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Информация о программе
        info_text = QLabel(
            "Приложение для голосовой диктовки и расшифровки речи в текст.\n"
            "Использует современные технологии распознавания речи.\n\n"
        )
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        # Разработчик
        dev_label = QLabel("👤 Разработчик:")
        dev_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(dev_label)
        dev_info = QLabel(DEVELOPER_NAME)
        layout.addWidget(dev_info)
        
        layout.addSpacing(10)
        
        # Канал Telegram
        tg_label = QLabel("📱 Telegram канал:")
        tg_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(tg_label)
        
        tg_button = QPushButton(TELEGRAM_CHANNEL)
        tg_button.setStyleSheet("text-align: left; color: #0088cc; text-decoration: underline; border: none;")
        tg_button.setCursor(Qt.PointingHandCursor)
        tg_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(TELEGRAM_CHANNEL)))
        layout.addWidget(tg_button)
        
        layout.addSpacing(10)
        
        # Email
        email_label = QLabel("📧 Предложения и вопросы:")
        email_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(email_label)
        
        email_button = QPushButton(DEVELOPER_EMAIL)
        email_button.setStyleSheet("text-align: left; color: #0088cc; text-decoration: underline; border: none;")
        email_button.setCursor(Qt.PointingHandCursor)
        email_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(f"mailto:{DEVELOPER_EMAIL}")))
        layout.addWidget(email_button)
        
        layout.addStretch()
        
        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)


if __name__ == "__main__":
    main()
