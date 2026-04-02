"""Enterprise Audio tab — browse, play, transcribe, translate, TTS, export, import.

Features:
- 107K+ voice audio files indexed with paloc text linking (94.9% match)
- 3 voice languages identified: Korean (0005), Japanese (0006), English (0035)
- Category filter: Quest Dialogue, AI Ambient, etc.
- NPC voice type filter: Human Male/Female, Dwarf, Giant, etc.
- Language filter: KO, JA, EN
- Linked paloc text shown for each audio file in all game languages
- Linked paloc text in all 14 game languages
- TTS generation with multi-provider support
- Export WAV / Import WAV / Patch to Game
- Generated audio history with playback
- Virtual scrolling table for 100K+ files
"""

import os
import tempfile
from enum import Enum
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSplitter, QTableView, QHeaderView, QPlainTextEdit,
    QAbstractItemView, QApplication, QMenu, QSlider, QListWidget,
    QListWidgetItem,
)
from PySide6.QtCore import (
    Qt, Signal, QTimer, QAbstractTableModel, QModelIndex,
)
from PySide6.QtGui import QColor

from core.vfs_manager import VfsManager
from core.pamt_parser import PamtFileEntry
from core.audio_converter import wem_to_wav, get_audio_info
from core.audio_index import (
    AudioEntry, build_audio_index, build_paloc_lookup,
    get_all_categories, get_all_languages, VOICE_LANG_PACKAGES,
)
from ui.widgets.audio_player import AudioPlayerWidget
from ui.widgets.progress_widget import ProgressWidget
from ui.dialogs.file_picker import pick_directory, pick_file, pick_save_file
from ui.dialogs.confirmation import show_error, show_info, confirm_action
from utils.thread_worker import FunctionWorker
from utils.platform_utils import format_file_size
from utils.logger import get_logger

logger = get_logger("ui.tab_audio")

ALL_PACKAGES = "All"
ALL_CATEGORIES = "All Categories"
ALL_LANGUAGES = "All Languages"

_COL_FILE = 0
_COL_LANG = 1
_COL_CATEGORY = 2
_COL_TEXT = 3
_COL_SIZE = 4
_COL_NPC = 5
_COL_COUNT = 6
_HEADERS = ["File", "Lang", "Category", "Text", "Size", "NPC Voice"]


class _AudioModel(QAbstractTableModel):
    """Virtual model for audio file list with paloc text linking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all: list[AudioEntry] = []
        self._filtered: list[int] = []
        self._cat_filter = ""
        self._lang_filter = ""
        self._search = ""

    def set_data(self, entries: list[AudioEntry]):
        self.beginResetModel()
        self._all = entries
        self._refilter()
        self.endResetModel()

    def set_filter(self, category: str = "", language: str = "", search: str = ""):
        self.beginResetModel()
        self._cat_filter = category
        self._lang_filter = language
        self._search = search.strip().lower()
        self._refilter()
        self.endResetModel()

    def _refilter(self):
        cat = self._cat_filter
        lang = self._lang_filter
        search = self._search
        if not cat and not lang and not search:
            self._filtered = list(range(len(self._all)))
            return
        result = []
        for i, e in enumerate(self._all):
            if cat and e.category != cat:
                continue
            if lang and e.voice_lang != lang:
                continue
            if search:
                # Search across filename, key, all translation texts, and NPC voice
                all_texts = " ".join(e.text_translations.values()) if e.text_translations else e.text_original
                haystack = f"{e.entry.path} {e.paloc_key} {e.text_original} {all_texts} {e.voice_prefix}".lower()
                if search not in haystack:
                    continue
            result.append(i)
        self._filtered = result

    def row_at(self, view_row: int) -> AudioEntry:
        if 0 <= view_row < len(self._filtered):
            return self._all[self._filtered[view_row]]
        return None

    def rowCount(self, parent=QModelIndex()):
        return len(self._filtered)

    def columnCount(self, parent=QModelIndex()):
        return _COL_COUNT

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return _HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._filtered):
            return None
        e = self._all[self._filtered[index.row()]]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == _COL_FILE:
                return os.path.basename(e.entry.path)
            elif col == _COL_LANG:
                return e.voice_lang.upper() if e.voice_lang else ""
            elif col == _COL_CATEGORY:
                return e.category
            elif col == _COL_TEXT:
                return e.text_original[:80] if e.text_original else ""
            elif col == _COL_SIZE:
                return format_file_size(e.entry.orig_size)
            elif col == _COL_NPC:
                return e.voice_prefix

        elif role == Qt.ForegroundRole:
            if col == _COL_LANG:
                colors = {"ko": QColor("#f9e2af"), "ja": QColor("#f38ba8"), "en": QColor("#89b4fa")}
                return colors.get(e.voice_lang)
            if col == _COL_TEXT and e.text_original:
                return QColor("#a6e3a1")

        elif role == Qt.ToolTipRole:
            lines = [e.entry.path, f"Key: {e.paloc_key}"]
            if e.text_original:
                lines.append(f"Text: {e.text_original[:200]}")
            if e.npc_gender:
                lines.append(f"NPC: {e.npc_gender} {e.npc_class} ({e.npc_age})")
            return "\n".join(lines)

        return None

    def sort(self, column, order=Qt.AscendingOrder):
        self.beginResetModel()
        rev = order == Qt.DescendingOrder
        a = self._all
        if column == _COL_FILE:
            self._filtered.sort(key=lambda i: a[i].entry.path.lower(), reverse=rev)
        elif column == _COL_LANG:
            self._filtered.sort(key=lambda i: a[i].voice_lang, reverse=rev)
        elif column == _COL_CATEGORY:
            self._filtered.sort(key=lambda i: a[i].category, reverse=rev)
        elif column == _COL_TEXT:
            self._filtered.sort(key=lambda i: a[i].text_original, reverse=rev)
        elif column == _COL_SIZE:
            self._filtered.sort(key=lambda i: a[i].entry.orig_size, reverse=rev)
        elif column == _COL_NPC:
            self._filtered.sort(key=lambda i: a[i].voice_prefix, reverse=rev)
        self.endResetModel()

    @property
    def filtered_count(self): return len(self._filtered)
    @property
    def total_count(self): return len(self._all)


class AudioTab(QWidget):
    """Enterprise audio tab."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._vfs: VfsManager = None
        self._all_groups: list[str] = []
        self._tts_engine = None
        self._wav_cache: dict = {}
        self._generated_files: list[dict] = []
        self._temp_dir = tempfile.mkdtemp(prefix="crimsonforge_audio_")
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._apply_filter)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar ──
        tb = QHBoxLayout()

        tb.addWidget(QLabel("Language:"))
        self._lang_filter = QComboBox()
        self._lang_filter.addItem(ALL_LANGUAGES, "")
        self._lang_filter.addItem("Korean (KO)", "ko")
        self._lang_filter.addItem("Japanese (JA)", "ja")
        self._lang_filter.addItem("English (EN)", "en")
        self._lang_filter.currentIndexChanged.connect(lambda _: self._apply_filter())
        self._lang_filter.setMinimumWidth(120)
        tb.addWidget(self._lang_filter)

        tb.addWidget(QLabel("Category:"))
        self._cat_filter = QComboBox()
        self._cat_filter.addItem(ALL_CATEGORIES, "")
        self._cat_filter.currentIndexChanged.connect(lambda _: self._apply_filter())
        self._cat_filter.setMinimumWidth(140)
        tb.addWidget(self._cat_filter)

        tb.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search by filename, key, text, NPC voice...")
        self._search_input.textChanged.connect(lambda _: self._search_timer.start())
        tb.addWidget(self._search_input, 1)

        self._count_label = QLabel("0 files")
        self._count_label.setStyleSheet("color: #89b4fa; font-weight: 600; padding: 0 4px;")
        tb.addWidget(self._count_label)
        layout.addLayout(tb)

        # ── Main splitter ──
        splitter = QSplitter(Qt.Horizontal)

        # LEFT: Audio table
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)

        lh = QHBoxLayout()
        lbl = QLabel("Voice Audio Files")
        lbl.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px;")
        lh.addWidget(lbl)
        lh.addStretch()
        exp_btn = QPushButton("Export Selected WAV")
        exp_btn.setObjectName("primary")
        exp_btn.clicked.connect(self._export_selected)
        lh.addWidget(exp_btn)
        ll.addLayout(lh)

        self._model = _AudioModel(self)
        self._view = QTableView()
        self._view.setModel(self._model)
        self._view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._view.setAlternatingRowColors(True)
        self._view.setSortingEnabled(True)
        self._view.setShowGrid(False)
        self._view.verticalHeader().setVisible(False)
        self._view.verticalHeader().setDefaultSectionSize(22)
        self._view.horizontalHeader().setSectionResizeMode(_COL_FILE, QHeaderView.Interactive)
        self._view.horizontalHeader().setSectionResizeMode(_COL_TEXT, QHeaderView.Stretch)
        self._view.setColumnWidth(_COL_FILE, 280)
        self._view.setColumnWidth(_COL_LANG, 40)
        self._view.setColumnWidth(_COL_CATEGORY, 120)
        self._view.setColumnWidth(_COL_SIZE, 60)
        self._view.setColumnWidth(_COL_NPC, 140)
        self._view.clicked.connect(self._on_row_clicked)
        self._view.setContextMenuPolicy(Qt.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._show_context_menu)
        ll.addWidget(self._view, 1)
        splitter.addWidget(left)

        # CENTER: Player + Text + Generated
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)

        cl.addWidget(QLabel("Player"))
        self._audio_player = AudioPlayerWidget(standalone=True)
        cl.addWidget(self._audio_player)

        # Linked text display
        self._text_display = QPlainTextEdit()
        self._text_display.setReadOnly(True)
        self._text_display.setMaximumHeight(180)
        self._text_display.setPlaceholderText("Click an audio file to see linked dialogue text...")
        self._text_display.setStyleSheet("font-size: 12px;")
        cl.addWidget(self._text_display)

        # Generated audio history
        gh = QHBoxLayout()
        gh.addWidget(QLabel("Generated Audio"))
        gh.addStretch()
        QPushButton("Clear All", clicked=self._clear_generated).also = gh.addWidget(
            QPushButton("Clear All", clicked=self._clear_generated))
        cl.addLayout(gh)

        self._gen_list = QListWidget()
        self._gen_list.setAlternatingRowColors(True)
        self._gen_list.setMaximumHeight(120)
        self._gen_list.itemClicked.connect(self._play_generated)
        cl.addWidget(self._gen_list)

        splitter.addWidget(center)

        # RIGHT: TTS Generator
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        rl.addWidget(QLabel("TTS Generator"))

        pr = QHBoxLayout()
        pr.addWidget(QLabel("Provider:"))
        self._tts_provider = QComboBox()
        self._tts_provider.currentIndexChanged.connect(self._on_tts_provider_changed)
        pr.addWidget(self._tts_provider, 1)
        rl.addLayout(pr)

        self._tts_model_label = QLabel("Model: (select provider)")
        self._tts_model_label.setStyleSheet("color: #89b4fa; padding: 2px;")
        rl.addWidget(self._tts_model_label)

        vr = QHBoxLayout()
        vr.addWidget(QLabel("Voice:"))
        self._voice_combo = QComboBox()
        vr.addWidget(self._voice_combo, 1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(55)
        refresh_btn.clicked.connect(lambda: self._on_tts_provider_changed())
        vr.addWidget(refresh_btn)
        rl.addLayout(vr)

        lr = QHBoxLayout()
        lr.addWidget(QLabel("Language:"))
        self._tts_lang = QComboBox()
        self._tts_lang.setEditable(True)
        self._tts_lang.addItems([
            "en-US", "en-GB", "ar-SA", "ko-KR", "ja-JP", "zh-CN",
            "de-DE", "fr-FR", "es-ES", "it-IT", "pt-BR", "ru-RU",
        ])
        self._tts_lang.setCurrentText("en-US")
        self._tts_lang.currentTextChanged.connect(lambda _: self._on_tts_provider_changed())
        lr.addWidget(self._tts_lang, 1)
        rl.addLayout(lr)

        sr = QHBoxLayout()
        sr.addWidget(QLabel("Speed:"))
        self._speed = QSlider(Qt.Horizontal)
        self._speed.setRange(50, 200)
        self._speed.setValue(100)
        self._speed_label = QLabel("1.0x")
        self._speed.valueChanged.connect(lambda v: self._speed_label.setText(f"{v/100:.1f}x"))
        sr.addWidget(self._speed, 1)
        sr.addWidget(self._speed_label)
        rl.addLayout(sr)

        rl.addWidget(QLabel("Text:"))
        self._tts_text = QPlainTextEdit()
        self._tts_text.setPlaceholderText("Enter text or click audio to load linked text...")
        self._tts_text.setMaximumHeight(100)
        rl.addWidget(self._tts_text)

        btns = QHBoxLayout()
        gen_btn = QPushButton("Generate")
        gen_btn.setObjectName("primary")
        gen_btn.clicked.connect(self._generate_tts)
        btns.addWidget(gen_btn)
        patch_btn = QPushButton("Generate + Patch")
        patch_btn.clicked.connect(self._generate_and_patch)
        btns.addWidget(patch_btn)
        rl.addLayout(btns)
        rl.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([480, 300, 250])
        layout.addWidget(splitter, 1)

        self._progress = ProgressWidget()
        layout.addWidget(self._progress)

    # ── Initialization ──

    def initialize_from_game(self, vfs: VfsManager, groups: list[str]) -> None:
        self._vfs = vfs
        self._all_groups = groups

        self._progress.set_progress(0, "Building audio index...")
        QApplication.processEvents()

        # Build paloc lookup for text linking
        paloc_lookup = build_paloc_lookup(vfs, groups,
            progress_callback=lambda p, m: (self._progress.set_status(m), QApplication.processEvents()))

        self._progress.set_status("Indexing voice audio files...")
        QApplication.processEvents()

        # Build audio index with paloc linking
        audio_entries = build_audio_index(vfs, groups, paloc_lookup,
            progress_callback=lambda p, m: (self._progress.set_progress(p, m), QApplication.processEvents()))

        self._model.set_data(audio_entries)

        # Populate category filter
        cats = get_all_categories(audio_entries)
        self._cat_filter.blockSignals(True)
        self._cat_filter.clear()
        self._cat_filter.addItem(ALL_CATEGORIES, "")
        for c in cats:
            self._cat_filter.addItem(c, c)
        self._cat_filter.blockSignals(False)

        self._apply_filter()

        linked = sum(1 for e in audio_entries if e.text_original)
        self._progress.set_progress(100,
            f"Indexed {len(audio_entries):,} audio files, {linked:,} linked to text")

        # Init TTS
        from ai.tts_engine import TTSEngine, TTS_KEY_SHARING
        self._tts_engine = TTSEngine()
        self._tts_engine.initialize_from_config(self._config)

        self._tts_provider.blockSignals(True)
        self._tts_provider.clear()
        for p in self._tts_engine.list_providers():
            pid = p["id"]
            shared = TTS_KEY_SHARING.get(pid)
            if pid == "edge_tts":
                self._tts_provider.addItem(p["name"], pid)
            elif shared and self._config.get(f"ai_providers.{shared}.enabled", False):
                if self._config.get(f"ai_providers.{shared}.api_key", ""):
                    self._tts_provider.addItem(p["name"], pid)
            else:
                key = self._config.get(f"tts.{pid}_api_key", "")
                if key:
                    self._tts_provider.addItem(p["name"], pid)
        self._tts_provider.blockSignals(False)
        self._on_tts_provider_changed()


    # ── Filtering ──

    def _apply_filter(self):
        cat = self._cat_filter.currentData() or ""
        lang = self._lang_filter.currentData() or ""
        search = self._search_input.text()
        self._model.set_filter(cat, lang, search)
        self._count_label.setText(f"{self._model.filtered_count:,} / {self._model.total_count:,}")

    # ── Playback + Text Display ──

    def _on_row_clicked(self, index: QModelIndex):
        ae = self._model.row_at(index.row())
        if ae:
            self._play_and_show(ae)

    def _play_and_show(self, ae: AudioEntry):
        """Play audio and show linked text."""
        try:
            entry = ae.entry
            basename = os.path.basename(entry.path)

            # Cache decoded audio — key includes package group to distinguish languages
            ck = f"{ae.package_group}:{entry.path}"
            if ck in self._wav_cache:
                play_path = self._wav_cache[ck]
            else:
                # Use group prefix in temp filename to avoid overwriting between languages
                tmp = os.path.join(self._temp_dir, f"{ae.package_group}_{basename}")
                data = self._vfs.read_entry_data(entry)
                with open(tmp, "wb") as f:
                    f.write(data)
                play_path = tmp
                ext = os.path.splitext(basename)[1].lower()
                if ext in (".wem", ".bnk"):
                    wav_out = os.path.join(self._temp_dir, f"{ae.package_group}_{os.path.splitext(basename)[0]}.wav")
                    wav = wem_to_wav(tmp, wav_out)
                    if wav:
                        play_path = wav
                self._wav_cache[ck] = play_path

            self._audio_player.load_file(play_path)

            # Build text display
            lines = []
            lines.append(f"File: {entry.path}")
            lines.append(f"Language: {ae.voice_lang.upper()} | Category: {ae.category}")
            lines.append(f"NPC: {ae.npc_gender} {ae.npc_class} ({ae.npc_age})")
            lines.append(f"Paloc Key: {ae.paloc_key}")
            lines.append("")

            if ae.text_translations:
                lang_names = {
                    "ko": "Korean", "en": "English", "ja": "Japanese",
                    "ru": "Russian", "tr": "Turkish", "es": "Spanish",
                    "es-mx": "Spanish (MX)", "fr": "French", "de": "German",
                    "it": "Italian", "pl": "Polish", "pt-br": "Portuguese (BR)",
                    "zh-tw": "Chinese (TW)", "zh-cn": "Chinese (CN)",
                }
                for lang, text in sorted(ae.text_translations.items()):
                    name = lang_names.get(lang, lang.upper())
                    lines.append(f"[{name}] {text}")
            elif ae.text_original:
                lines.append(f"[Text] {ae.text_original}")
            else:
                lines.append("[No linked text found]")

            self._text_display.setPlainText("\n".join(lines))

            # Auto-load text into TTS input — use the selected TTS language if available
            tts_lang = self._tts_lang.currentText().strip()
            tts_lang_code = tts_lang.split("-")[0].lower() if tts_lang else ""
            tts_text = ""
            if ae.text_translations and tts_lang_code:
                # Try exact match first (e.g. "ar"), then try with region (e.g. "ar-sa")
                tts_text = ae.text_translations.get(tts_lang_code, "")
                if not tts_text:
                    # Try matching lang codes like "es" matching "es-mx"
                    for lk, lv in ae.text_translations.items():
                        if lk.startswith(tts_lang_code):
                            tts_text = lv
                            break
            if not tts_text:
                tts_text = ae.text_original
            if tts_text:
                self._tts_text.setPlainText(tts_text)

            self._progress.set_status(f"Playing: {basename}")

        except Exception as e:
            self._progress.set_status(f"Error: {e}")
            logger.error("Audio play error: %s", e)

    # ── Context Menu ──

    def _show_context_menu(self, pos):
        idx = self._view.indexAt(pos)
        if not idx.isValid():
            return
        ae = self._model.row_at(idx.row())
        if not ae:
            return

        menu = QMenu(self)
        entry = ae.entry

        menu.addAction("Play").triggered.connect(lambda: self._play_and_show(ae))
        menu.addSeparator()
        menu.addAction("Export as WAV").triggered.connect(lambda: self._export_wav(entry))
        menu.addAction("Import WAV (replace)").triggered.connect(lambda: self._import_wav(entry))
        menu.addAction("Import WAV + Patch to Game").triggered.connect(lambda: self._import_and_patch(entry))
        menu.addSeparator()
        menu.addAction("Copy paloc key").triggered.connect(
            lambda: QApplication.clipboard().setText(ae.paloc_key))
        menu.addAction("Copy text").triggered.connect(
            lambda: QApplication.clipboard().setText(ae.text_original))
        menu.addAction("Copy file path").triggered.connect(
            lambda: QApplication.clipboard().setText(ae.entry.path))

        menu.exec(self._view.viewport().mapToGlobal(pos))

    # ── Export / Import ──

    def _export_wav(self, entry: PamtFileEntry):
        basename = os.path.splitext(os.path.basename(entry.path))[0]
        save = pick_save_file(self, "Export as WAV", f"{basename}.wav", filters="WAV (*.wav)")
        if not save:
            return
        try:
            data = self._vfs.read_entry_data(entry)
            tmp = os.path.join(self._temp_dir, os.path.basename(entry.path))
            with open(tmp, "wb") as f:
                f.write(data)
            ext = os.path.splitext(entry.path)[1].lower()
            if ext in (".wem", ".bnk"):
                r = wem_to_wav(tmp, save)
                if not r:
                    show_error(self, "Error", "WEM decode failed")
                    return
            else:
                import shutil
                shutil.copy2(tmp, save)
            show_info(self, "Exported", f"Saved to:\n{save}")
        except Exception as e:
            show_error(self, "Error", str(e))

    def _export_selected(self):
        out = pick_directory(self, "Export Directory")
        if not out:
            return
        rows = sorted({i.row() for i in self._view.selectedIndexes()})
        ok = err = 0
        for r in rows:
            ae = self._model.row_at(r)
            if not ae:
                continue
            try:
                data = self._vfs.read_entry_data(ae.entry)
                bn = os.path.splitext(os.path.basename(ae.entry.path))[0]
                tmp = os.path.join(self._temp_dir, os.path.basename(ae.entry.path))
                with open(tmp, "wb") as f:
                    f.write(data)
                ext = os.path.splitext(ae.entry.path)[1].lower()
                wav_out = os.path.join(out, f"{bn}.wav")
                if ext in (".wem", ".bnk"):
                    wem_to_wav(tmp, wav_out)
                else:
                    import shutil
                    shutil.copy2(tmp, wav_out)
                ok += 1
            except Exception:
                err += 1
        show_info(self, "Batch Export", f"Exported {ok} files ({err} errors)")

    def _import_wav(self, entry: PamtFileEntry):
        wav = pick_file(self, "Select WAV", filters="Audio (*.wav *.ogg *.mp3);;All (*.*)")
        if not wav:
            return
        try:
            from core.audio_importer import import_audio
            orig = self._vfs.read_entry_data(entry)
            new = import_audio(wav, entry, orig)
            show_info(self, "Imported",
                      f"Original: {format_file_size(len(orig))}\nNew: {format_file_size(len(new))}")
        except Exception as e:
            show_error(self, "Error", str(e))

    def _import_and_patch(self, entry: PamtFileEntry):
        wav = pick_file(self, "Select Audio File",
                        filters="Audio (*.wav *.ogg *.wem);;All (*.*)")
        if not wav:
            return
        try:
            from core.repack_engine import RepackEngine, ModifiedFile
            from core.audio_converter import wav_to_wem
            orig = self._vfs.read_entry_data(entry)

            ext = os.path.splitext(wav)[1].lower()
            if ext == ".wem":
                # Already WEM — use directly
                with open(wav, "rb") as f:
                    new = f.read()
            else:
                # Check Wwise
                from utils.wwise_installer import is_wwise_installed
                if not is_wwise_installed():
                    show_error(self, "Wwise Required",
                               "Audio patching requires Wwise (free) for Vorbis encoding.\n\n"
                               "Install from audiokinetic.com (free account)")
                    return
                # Convert WAV/OGG → WEM via Wwise
                self._progress.set_status("Converting to WEM (Vorbis) via Wwise...")
                QApplication.processEvents()
                wem_path = wav_to_wem(wav, orig)
                if not wem_path:
                    show_error(self, "Error", "Wwise conversion failed.")
                    return
                with open(wem_path, "rb") as f:
                    new = f.read()

            if not confirm_action(self, "Patch Audio",
                                  f"Replace {entry.path}?\n"
                                  f"Orig: {format_file_size(len(orig))}\n"
                                  f"New: {format_file_size(len(new))}"):
                return
            game = os.path.dirname(os.path.dirname(entry.paz_file))
            papgt = os.path.join(game, "meta", "0.papgt")
            grp = os.path.basename(os.path.dirname(entry.paz_file))
            pamt = self._vfs.load_pamt(grp)
            mf = ModifiedFile(data=new, entry=entry, pamt_data=pamt, package_group=grp)
            result = RepackEngine(game).repack([mf], papgt_path=papgt)
            if result.success:
                show_info(self, "Patched", f"Patched {entry.path}")
            else:
                show_error(self, "Error", result.error)
        except Exception as e:
            show_error(self, "Error", str(e))

    # ── TTS ──

    def _on_tts_provider_changed(self, _=None):
        if not self._tts_engine:
            return
        pid = self._tts_provider.currentData()
        if not pid:
            return

        from ai.tts_engine import TTS_KEY_SHARING
        shared = TTS_KEY_SHARING.get(pid)
        cfg = shared or pid
        model = self._config.get(f"ai_providers.{cfg}.default_tts_model", "")
        self._tts_model_label.setText(f"Model: {model or '(set in Settings)'}")

        # Update API key
        provider = self._tts_engine.get_provider(pid)
        if provider and shared:
            key = self._config.get(f"ai_providers.{shared}.api_key", "")
            if key:
                provider.api_key = key

        # Fetch voices
        self._voice_combo.clear()
        lang = self._tts_lang.currentText().strip()
        try:
            voices = self._tts_engine.list_voices(pid, lang)
            for v in voices[:300]:
                label = v.name
                if v.gender:
                    label += f" ({v.gender})"
                self._voice_combo.addItem(label, v.voice_id)
        except Exception:
            pass
        if self._voice_combo.count() == 0:
            self._voice_combo.addItem("(check Settings)", "")

    def _generate_tts(self):
        text = self._tts_text.toPlainText().strip()
        if not text:
            show_error(self, "Error", "Enter text")
            return
        if not self._tts_engine:
            return

        pid = self._tts_provider.currentData() or "edge_tts"
        from ai.tts_engine import TTS_KEY_SHARING
        shared = TTS_KEY_SHARING.get(pid)
        cfg = shared or pid
        model = self._config.get(f"ai_providers.{cfg}.default_tts_model", "")
        voice = self._voice_combo.currentData() or ""
        lang = self._tts_lang.currentText().strip()
        spd = self._speed.value() / 100.0

        self._progress.set_status("Generating TTS...")
        QApplication.processEvents()

        result = self._tts_engine.synthesize(text, pid, model, voice, lang, spd)
        if result.success and result.audio_data:
            import time as _t
            fname = f"tts_{int(_t.time()*1000)}.wav"
            path = os.path.join(self._temp_dir, fname)
            with open(path, "wb") as f:
                f.write(result.audio_data)
            self._audio_player.load_file(path)

            voice_name = self._voice_combo.currentText().split(" (")[0]
            item = QListWidgetItem(
                f"{voice_name} | {format_file_size(len(result.audio_data))} | "
                f"{result.latency_ms:.0f}ms\n{text[:50]}")
            item.setData(Qt.UserRole, path)
            self._gen_list.insertItem(0, item)
            self._generated_files.insert(0, {"path": path, "text": text})
            self._progress.set_status(f"Generated: {format_file_size(len(result.audio_data))}")
        else:
            show_error(self, "TTS Error", result.error or "Failed")

    def _generate_and_patch(self):
        rows = sorted({i.row() for i in self._view.selectedIndexes()})
        if not rows:
            show_error(self, "Error", "Select an audio file first")
            return
        ae = self._model.row_at(rows[0])
        if not ae:
            return

        self._generate_tts()
        if not self._generated_files:
            return

        tts_wav_path = self._generated_files[0]["path"]
        try:
            from core.repack_engine import RepackEngine, ModifiedFile
            from core.audio_converter import wav_to_wem
            entry = ae.entry
            orig_data = self._vfs.read_entry_data(entry)

            # Convert TTS WAV → WEM (matching original format: Vorbis, 48kHz, mono)
            # Check if Wwise is installed for proper Vorbis encoding
            from utils.wwise_installer import is_wwise_installed
            if not is_wwise_installed():
                show_error(self, "Wwise Required",
                           "Audio patching requires Wwise (free) for Vorbis encoding.\n\n"
                           "The game only accepts Vorbis-encoded WEM audio.\n"
                           "Without Wwise, patched audio will be silent in-game.\n\n"
                           "Install Wwise:\n"
                           "1. Go to audiokinetic.com and create a free account\n"
                           "2. Download the Audiokinetic Launcher\n"
                           "3. Install Wwise (any version, ~2GB)\n"
                           "4. Restart CrimsonForge — it will auto-detect Wwise")
                return

            self._progress.set_status("Converting WAV to WEM (Vorbis)...")
            QApplication.processEvents()

            wem_path = wav_to_wem(tts_wav_path, orig_data)
            if not wem_path or not os.path.isfile(wem_path):
                show_error(self, "Error",
                           "WAV to WEM conversion failed.\n"
                           "Check Wwise installation and try again.")
                return

            with open(wem_path, "rb") as f:
                new_data = f.read()

            if not confirm_action(self, "Patch Audio",
                                  f"Replace {entry.path}?\n\n"
                                  f"Original: {format_file_size(len(orig_data))} (WEM Vorbis)\n"
                                  f"New: {format_file_size(len(new_data))} (WEM Vorbis)\n\n"
                                  f"ffmpeg converted TTS audio to match game format."):
                return

            game = os.path.dirname(os.path.dirname(entry.paz_file))
            papgt = os.path.join(game, "meta", "0.papgt")
            grp = os.path.basename(os.path.dirname(entry.paz_file))
            pamt = self._vfs.load_pamt(grp)
            mf = ModifiedFile(data=new_data, entry=entry, pamt_data=pamt, package_group=grp)

            self._progress.set_status("Patching to game...")
            QApplication.processEvents()

            result = RepackEngine(game).repack([mf], papgt_path=papgt)
            if result.success:
                self._progress.set_status(f"Patched: {entry.path}")
                show_info(self, "Patched",
                          f"TTS audio patched to {entry.path}\n\n"
                          f"Original: {format_file_size(len(orig_data))}\n"
                          f"New: {format_file_size(len(new_data))}\n\n"
                          f"Launch the game to hear your changes!")
            else:
                show_error(self, "Error", result.error)
        except Exception as e:
            show_error(self, "Error", str(e))

    def _play_generated(self, item):
        path = item.data(Qt.UserRole)
        if path and os.path.isfile(path):
            self._audio_player.load_file(path)

    def _clear_generated(self):
        self._gen_list.clear()
        self._generated_files.clear()
