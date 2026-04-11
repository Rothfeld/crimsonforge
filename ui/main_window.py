"""Main application window with tab bar and game discovery flow.

On first launch, only the Game Setup tab is active. The user browses
or auto-discovers the game packages directory. After the game is loaded:
- VFS is built, all PAMT indices scanned
- All paloc localization files are discovered
- All tabs are unlocked and auto-populated with game data
- No tab requires the user to browse for game paths again

Tabs: Game Setup | Explorer (Unpack+Browse+Edit) | Repack | Translate | Font Builder | Settings | About

Performance architecture (v1.16.2):
- Tabs are lazily instantiated: only constructed when first clicked.
- Game loading runs in a background QThread — UI stays responsive.
- PAMT scanning uses concurrent.futures for parallel I/O.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QLabel, QApplication,
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QGroupBox, QStackedWidget, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer

from utils.config import ConfigManager
from utils.platform_utils import auto_discover_game
from core.vfs_manager import VfsManager
from ai.provider_registry import ProviderRegistry
from ui.themes.dark import DARK_THEME
from ui.themes.light import LIGHT_THEME
from ui.dialogs.confirmation import show_error
from ui.dialogs.file_picker import pick_directory
from utils.thread_worker import FunctionWorker
from version import APP_VERSION, APP_NAME
from utils.logger import get_logger

logger = get_logger("ui.main_window")

# ---------------------------------------------------------------------------
# Tab registry — maps tab index to (module_path, class_name, tab_label,
# constructor_args_key).  Tabs are only imported and constructed on demand.
# ---------------------------------------------------------------------------
_TAB_REGISTRY: list[dict] = [
    # Index 0 — Setup tab is built inline, not lazy.
    {"label": "Game Setup", "lazy": False},
    {"label": "Explorer",          "module": "ui.tab_explorer",          "cls": "ExplorerTab",          "args": "config",    "lazy": True},
    {"label": "Item Catalog",      "module": "ui.tab_item_catalog",      "cls": "ItemCatalogTab",       "args": None,        "lazy": True},
    {"label": "Dialogue Catalog",  "module": "ui.tab_dialogue_catalog",  "cls": "DialogueCatalogTab",   "args": None,        "lazy": True},
    {"label": "Repack",            "module": "ui.tab_repack",            "cls": "RepackTab",            "args": "config",    "lazy": True},
    {"label": "Translate",         "module": "ui.tab_translate",         "cls": "TranslateTab",         "args": "config_registry", "lazy": True},
    {"label": "Audio",             "module": "ui.tab_audio",             "cls": "AudioTab",             "args": "config",    "lazy": True},
    {"label": "Font Builder",      "module": "ui.tab_font",             "cls": "FontTab",              "args": "config",    "lazy": True},
    {"label": "Settings",          "module": "ui.tab_settings",          "cls": "SettingsTab",          "args": "config_registry", "lazy": True},
    {"label": "About",             "module": "ui.tab_about",             "cls": "AboutTab",             "args": "config_kw", "lazy": True},
]


class _LazyPlaceholder(QWidget):
    """Invisible stand-in added to the QTabWidget until the real tab is needed."""
    pass


class MainWindow(QMainWindow):
    """Main application window.

    On first launch, only Game Setup + Settings + About are enabled.
    After game path is set and loaded, all tabs unlock.
    """

    def __init__(self, config: ConfigManager, registry: ProviderRegistry):
        super().__init__()
        self._config = config
        self._registry = registry
        self._game_loaded = False
        self._vfs: VfsManager = None
        self._packages_path = ""
        self._discovered_palocs: list[dict] = []
        self._all_groups: list[str] = []
        self._game_version = ""
        self._loader_worker: FunctionWorker = None

        # Lazy tab tracking: index → real widget (None until materialised)
        self._real_tabs: dict[int, QWidget] = {}

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - Crimson Desert Modding Studio")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        self._central_stack = QStackedWidget()
        self._tabs = QTabWidget()
        self._loading_page = self._build_loading_page()
        self._central_stack.addWidget(self._tabs)
        self._central_stack.addWidget(self._loading_page)
        self.setCentralWidget(self._central_stack)

        # --- Build tabs: Setup is eager, everything else is a placeholder ---
        self._setup_tab = self._build_setup_tab()
        self._tabs.addTab(self._setup_tab, "Game Setup")
        self._real_tabs[0] = self._setup_tab

        for i, entry in enumerate(_TAB_REGISTRY):
            if i == 0:
                continue  # already added Setup
            placeholder = _LazyPlaceholder()
            self._tabs.addTab(placeholder, entry["label"])

        # Eagerly create Settings + About (lightweight, always needed)
        self._materialise_tab(8)   # Settings
        self._materialise_tab(9)   # About

        # Connect signals after Settings tab exists
        settings_tab = self._real_tabs[8]
        settings_tab.theme_changed.connect(self._apply_theme)
        settings_tab.settings_changed.connect(self._on_settings_changed)

        # Lazy tab activation
        self._tabs.currentChanged.connect(self._on_tab_changed)

        status_bar = QStatusBar()
        self._status_label = QLabel("Ready")
        self._game_version_label = QLabel("")
        self._game_version_label.setStyleSheet("font-size: 11px; color: #a6adc8; padding: 0 8px;")
        self._files_label = QLabel("Files: 0")
        status_bar.addWidget(self._status_label, 1)
        status_bar.addPermanentWidget(self._game_version_label)
        status_bar.addPermanentWidget(self._files_label)
        self.setStatusBar(status_bar)

        theme = config.get("general.theme", "dark")
        self._apply_theme(theme)

        saved_path = config.get("general.last_game_path", "")
        if saved_path and self._validate_game_path(saved_path):
            self._show_loading_screen(
                "Loading Crimson Desert...",
                "Scanning game files and restoring your last session.",
            )
            QTimer.singleShot(100, lambda: self._activate_game(saved_path))
        else:
            self._lock_tabs()
            self._show_main_tabs()
            QTimer.singleShot(300, self._auto_discover_and_load)

    # ------------------------------------------------------------------
    # Lazy tab materialisation
    # ------------------------------------------------------------------
    def _materialise_tab(self, index: int) -> QWidget:
        """Import, construct, and swap in the real tab widget for *index*."""
        if index in self._real_tabs:
            return self._real_tabs[index]

        entry = _TAB_REGISTRY[index]
        if not entry.get("lazy", False):
            return self._tabs.widget(index)

        import importlib
        mod = importlib.import_module(entry["module"])
        cls = getattr(mod, entry["cls"])

        args_key = entry.get("args")
        if args_key == "config":
            widget = cls(self._config)
        elif args_key == "config_registry":
            widget = cls(self._config, self._registry)
        elif args_key == "config_kw":
            widget = cls(config=self._config)
        else:
            widget = cls()

        # Swap placeholder with the real widget, keeping the same index/label
        old = self._tabs.widget(index)
        label = self._tabs.tabText(index)
        enabled = self._tabs.isTabEnabled(index)
        self._tabs.removeTab(index)
        self._tabs.insertTab(index, widget, label)
        self._tabs.setTabEnabled(index, enabled)
        if old is not None:
            old.deleteLater()

        self._real_tabs[index] = widget
        logger.debug("Materialised tab %d (%s)", index, label)
        return widget

    def _on_tab_changed(self, index: int):
        """Materialise the tab on first click and initialise if game is loaded."""
        if index not in self._real_tabs:
            widget = self._materialise_tab(index)
            if self._game_loaded:
                self._init_tab_from_game(index, widget)

    def _tab(self, index: int) -> QWidget | None:
        """Return the real tab at *index* or None if not yet materialised."""
        return self._real_tabs.get(index)

    # ------------------------------------------------------------------
    # Setup tab (always eager)
    # ------------------------------------------------------------------
    def _build_setup_tab(self) -> QWidget:
        widget = QWidget()
        outer = QVBoxLayout(widget)
        outer.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setMaximumWidth(700)
        layout = QVBoxLayout(container)

        title = QLabel("CrimsonForge - Game Setup")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 16px;")
        layout.addWidget(title)

        desc = QLabel(
            "To get started, locate your Crimson Desert game installation.\n"
            "CrimsonForge will auto-discover Steam installations, or you can browse manually."
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; padding: 8px; color: #a6adc8;")
        layout.addWidget(desc)

        path_group = QGroupBox("Game Packages Directory")
        path_layout = QVBoxLayout(path_group)

        path_row = QHBoxLayout()
        self._setup_path = QLineEdit()
        self._setup_path.setPlaceholderText("Path to packages/ directory (contains meta/, 0012/, 0020/, ...)")
        self._setup_path.setToolTip(
            "Path to the game's packages directory.\n"
            "This folder contains numbered subdirectories (0000/, 0008/, 0012/, etc.) and a meta/ folder.\n"
            "Typically found at: Steam/steamapps/common/Crimson Desert/"
        )
        path_row.addWidget(self._setup_path, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.setToolTip("Open a folder picker to select the game packages directory.")
        browse_btn.clicked.connect(self._setup_browse)
        path_row.addWidget(browse_btn)
        path_layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        self._discover_btn = QPushButton("Auto-Discover")
        self._discover_btn.setObjectName("primary")
        self._discover_btn.setToolTip("Automatically scan Steam library folders to find the Crimson Desert installation.")
        self._discover_btn.clicked.connect(self._auto_discover)
        btn_row.addWidget(self._discover_btn)
        self._load_btn = QPushButton("Load Game")
        self._load_btn.setObjectName("primary")
        self._load_btn.setToolTip("Load the game from the specified directory.\nParses all package archives and enables the modding tools.")
        self._load_btn.clicked.connect(self._setup_load)
        btn_row.addWidget(self._load_btn)
        btn_row.addStretch()
        path_layout.addLayout(btn_row)

        self._setup_status = QLabel("")
        self._setup_status.setWordWrap(True)
        path_layout.addWidget(self._setup_status)

        layout.addWidget(path_group)
        layout.addStretch()
        outer.addWidget(container)
        return widget

    # ------------------------------------------------------------------
    # Loading page
    # ------------------------------------------------------------------
    def _build_loading_page(self) -> QWidget:
        widget = QWidget()
        outer = QVBoxLayout(widget)
        outer.setContentsMargins(48, 48, 48, 48)
        outer.addStretch()

        container = QWidget()
        container.setMaximumWidth(560)
        layout = QVBoxLayout(container)
        layout.setSpacing(18)

        title = QLabel("CrimsonForge")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: bold; padding-bottom: 4px;")
        layout.addWidget(title)

        self._loading_title = QLabel("Loading...")
        self._loading_title.setAlignment(Qt.AlignCenter)
        self._loading_title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(self._loading_title)

        self._loading_detail = QLabel("")
        self._loading_detail.setAlignment(Qt.AlignCenter)
        self._loading_detail.setWordWrap(True)
        self._loading_detail.setStyleSheet("font-size: 13px; color: #a6adc8;")
        layout.addWidget(self._loading_detail)

        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 100)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedWidth(320)
        self._loading_bar.setFixedHeight(18)
        layout.addWidget(self._loading_bar, 0, Qt.AlignHCenter)

        outer.addWidget(container, 0, Qt.AlignCenter)
        outer.addStretch()
        return widget

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _show_loading_screen(self, title: str, detail: str = "", pct: int = -1) -> None:
        self._loading_title.setText(title)
        self._loading_detail.setText(detail)
        if pct < 0:
            self._loading_bar.setRange(0, 0)  # indeterminate
        else:
            self._loading_bar.setRange(0, 100)
            self._loading_bar.setValue(pct)
        self._central_stack.setCurrentWidget(self._loading_page)
        self.setCursor(Qt.WaitCursor)
        self._status_label.setText(title)
        QApplication.processEvents()

    def _show_main_tabs(self) -> None:
        self._central_stack.setCurrentWidget(self._tabs)
        self.unsetCursor()

    def _lock_tabs(self) -> None:
        for i in range(self._tabs.count()):
            tab_text = self._tabs.tabText(i)
            if tab_text in ("Game Setup", "Settings", "About"):
                continue
            self._tabs.setTabEnabled(i, False)
        self._tabs.setCurrentIndex(0)
        self._status_label.setText("Select game location to get started")

    def _unlock_tabs(self) -> None:
        for i in range(self._tabs.count()):
            self._tabs.setTabEnabled(i, True)

    def _validate_game_path(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return os.path.isfile(os.path.join(path, "meta", "0.papgt"))

    # ------------------------------------------------------------------
    # Auto-discover
    # ------------------------------------------------------------------
    def _auto_discover_and_load(self) -> None:
        """Auto-discover game and load it immediately if found (first run)."""
        self._setup_status.setText("Scanning Steam libraries for Crimson Desert...")
        self._setup_status.setStyleSheet("color: #89b4fa;")
        self._discover_btn.setEnabled(False)
        QApplication.processEvents()

        path = auto_discover_game()
        self._discover_btn.setEnabled(True)

        if path:
            self._setup_path.setText(path)
            self._setup_status.setText(f"Found: {path}\nAuto-loading game...")
            self._setup_status.setStyleSheet("color: #a6e3a1;")
            QApplication.processEvents()
            self._show_loading_screen(
                "Loading Crimson Desert...",
                "Game auto-discovered. Reading package groups and preparing the workspace.",
            )
            QTimer.singleShot(0, lambda: self._activate_game(path))
        else:
            self._setup_status.setText(
                "Crimson Desert not found in Steam libraries.\n"
                "Use 'Browse...' to manually select the packages/ directory."
            )
            self._setup_status.setStyleSheet("color: #f9e2af;")

    def _auto_discover(self) -> None:
        self._setup_status.setText("Scanning Steam libraries for Crimson Desert...")
        self._setup_status.setStyleSheet("color: #89b4fa;")
        self._discover_btn.setEnabled(False)
        QApplication.processEvents()

        path = auto_discover_game()
        self._discover_btn.setEnabled(True)

        if path:
            self._setup_path.setText(path)
            self._setup_status.setText(f"Found: {path}\nClick 'Load Game' to continue.")
            self._setup_status.setStyleSheet("color: #a6e3a1;")
        else:
            self._setup_status.setText(
                "Crimson Desert not found in Steam libraries.\n"
                "Use 'Browse...' to manually select the packages/ directory."
            )
            self._setup_status.setStyleSheet("color: #f9e2af;")

    def _setup_browse(self) -> None:
        path = pick_directory(self, "Select Crimson Desert packages/ Directory")
        if path:
            self._setup_path.setText(path)

    def _setup_load(self) -> None:
        path = self._setup_path.text().strip()
        if not path:
            self._setup_status.setText("Enter or browse for a game packages directory.")
            self._setup_status.setStyleSheet("color: #f38ba8;")
            return
        if not self._validate_game_path(path):
            self._setup_status.setText(
                f"Invalid packages directory: {path}\n"
                f"The directory must contain meta/0.papgt."
            )
            self._setup_status.setStyleSheet("color: #f38ba8;")
            return
        self._show_loading_screen(
            "Loading Crimson Desert...",
            "Reading package groups and preparing the workspace.",
        )
        QTimer.singleShot(0, lambda: self._activate_game(path))

    # ------------------------------------------------------------------
    # Game version / update detection (cheap, runs on main thread)
    # ------------------------------------------------------------------
    def _detect_game_version(self, packages_path: str) -> str:
        try:
            papgt_path = os.path.join(packages_path, "meta", "0.papgt")
            if not os.path.isfile(papgt_path):
                return "Unknown"
            stat = os.stat(papgt_path)
            size = stat.st_size
            from datetime import datetime
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            from core.checksum_engine import pa_checksum
            with open(papgt_path, "rb") as f:
                data = f.read()
            crc = pa_checksum(data[12:]) if len(data) > 12 else 0
            return f"CRC:0x{crc:08X} | Modified:{mod_time} | Size:{size:,}B"
        except Exception as e:
            logger.warning("Failed to detect game version: %s", e)
            return "Unknown"

    def _check_game_updates(self, packages_path: str, groups: list[str]) -> dict:
        summary = {"new_groups": 0, "new_palocs": 0, "changed_palocs": 0}
        try:
            saved_paloc_count = self._config.get("game.last_paloc_count", 0)
            saved_group_count = self._config.get("game.last_group_count", 0)
            current_paloc_count = len(self._discovered_palocs)
            current_group_count = len(groups)

            if saved_group_count > 0:
                summary["new_groups"] = max(0, current_group_count - saved_group_count)
            if saved_paloc_count > 0:
                summary["new_palocs"] = max(0, current_paloc_count - saved_paloc_count)

            saved_fp = self._config.get("game.last_fingerprint", "")
            papgt_path = os.path.join(packages_path, "meta", "0.papgt")
            if os.path.isfile(papgt_path):
                from core.checksum_engine import pa_checksum
                with open(papgt_path, "rb") as f:
                    data = f.read()
                crc = pa_checksum(data[12:]) if len(data) > 12 else 0
                current_fp = f"{crc:08X}_{os.path.getsize(papgt_path)}"
                if saved_fp and current_fp != saved_fp:
                    summary["changed_palocs"] = 1
                self._config.set("game.last_fingerprint", current_fp)

            self._config.set("game.last_paloc_count", current_paloc_count)
            self._config.set("game.last_group_count", current_group_count)
        except Exception as e:
            logger.warning("Failed to check game updates: %s", e)
        return summary

    # ------------------------------------------------------------------
    # Background game loading (threaded)
    # ------------------------------------------------------------------
    def _activate_game(self, packages_path: str) -> None:
        """Kick off the background game loader thread."""
        self._packages_path = packages_path
        self._config.set("general.last_game_path", packages_path)
        self._config.save()

        self._show_loading_screen("Loading Crimson Desert...", "Reading package groups.", 0)

        worker = FunctionWorker(self._game_load_task, packages_path)
        worker.progress.connect(self._on_load_progress)
        worker.finished_result.connect(self._on_load_finished)
        worker.error_occurred.connect(self._on_load_error)
        self._loader_worker = worker
        worker.start()

    @staticmethod
    def _scan_paloc_files_parallel(vfs: VfsManager, groups: list[str], progress_cb) -> list[dict]:
        """Scan all package groups for .paloc files using parallel I/O."""
        paloc_lang_map = {
            "eng": "en", "kor": "ko", "jpn": "ja", "rus": "ru",
            "tur": "tr", "spa-es": "es", "spa-mx": "es-MX",
            "fre": "fr", "ger": "de", "ita": "it", "pol": "pl",
            "por-br": "pt-BR", "zho-tw": "zh-TW", "zho-cn": "zh",
            "tha": "th", "vie": "vi", "ind": "id", "ara": "ar",
        }
        results = []
        total = len(groups)

        def _scan_group(group: str) -> list[dict]:
            found = []
            try:
                pamt = vfs.load_pamt(group)
                for entry in pamt.file_entries:
                    if entry.path.lower().endswith(".paloc"):
                        basename = os.path.basename(entry.path)
                        name_part = basename.replace("localizationstring_", "").replace(".paloc", "")
                        lang_code = paloc_lang_map.get(name_part, name_part)
                        found.append({
                            "filename": basename,
                            "lang_code": lang_code,
                            "lang_key": name_part,
                            "group": group,
                            "entry": entry,
                        })
            except Exception as e:
                logger.warning("Error scanning group %s for palocs: %s", group, e)
            return found

        # Use up to 8 threads for parallel PAMT I/O
        with ThreadPoolExecutor(max_workers=min(8, total or 1)) as pool:
            futures = {pool.submit(_scan_group, g): g for g in groups}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                results.extend(future.result())
                if done_count % 4 == 0 or done_count == total:
                    pct = int((done_count / total) * 30) + 10  # 10-40%
                    progress_cb(pct, f"Scanning localization: {done_count}/{total} groups")
        return results

    def _game_load_task(self, worker: FunctionWorker, packages_path: str) -> dict:
        """Heavy I/O work that runs on the background thread.

        Returns a dict with everything the main thread needs to finish setup.
        """
        worker.report_progress(5, "Reading package groups.")
        vfs = VfsManager(packages_path)
        groups = vfs.list_package_groups()

        worker.report_progress(10, f"Scanning localization across {len(groups)} groups.")
        palocs = self._scan_paloc_files_parallel(vfs, groups, worker.report_progress)

        worker.report_progress(45, "Detecting game version.")
        game_version = self._detect_game_version(packages_path)

        worker.report_progress(50, "Background loading complete.")
        return {
            "vfs": vfs,
            "groups": groups,
            "palocs": palocs,
            "game_version": game_version,
            "packages_path": packages_path,
        }

    def _on_load_progress(self, pct: int, msg: str):
        self._show_loading_screen("Loading Crimson Desert...", msg, pct)

    def _on_load_finished(self, result: dict):
        """Runs on the main thread after the background loader finishes."""
        try:
            self._vfs = result["vfs"]
            self._all_groups = result["groups"]
            self._discovered_palocs = result["palocs"]
            self._game_version = result["game_version"]
            self._packages_path = result["packages_path"]
            groups = self._all_groups

            # ---- Initialise only the Explorer tab eagerly (it's the landing tab) ----
            self._show_loading_screen("Loading Crimson Desert...", "Building the Explorer file index.", 55)
            explorer = self._materialise_tab(1)
            explorer.initialize_from_game(self._vfs, groups)
            explorer.files_extracted.connect(self._on_files_extracted)
            explorer._game_initialized = True

            # ---- All other tabs initialise lazily on first click ----
            self._show_loading_screen("Loading Crimson Desert...", "Finalising.", 90)

            update_summary = self._check_game_updates(self._packages_path, groups)

            self._unlock_tabs()
            self._game_loaded = True

            # Restore translate session if it was the last active tab
            translate_tab = self._tab(5)  # Translate
            if translate_tab and hasattr(translate_tab, 'restore_state') and translate_tab.restore_state():
                self._tabs.setCurrentIndex(5)
            else:
                self._tabs.setCurrentIndex(1)

            paloc_count = len(self._discovered_palocs)
            self._game_version_label.setText(f"Game: {self._game_version}")
            self._status_label.setText(
                f"Game loaded: {len(groups)} package groups, {paloc_count} localization files"
            )
            self._files_label.setText(f"Groups: {len(groups)} | Languages: {paloc_count}")

            has_updates = (
                update_summary["new_groups"] > 0
                or update_summary["new_palocs"] > 0
                or update_summary["changed_palocs"] > 0
            )
            if has_updates:
                update_parts = []
                if update_summary["new_groups"] > 0:
                    update_parts.append(f"{update_summary['new_groups']} new package groups")
                if update_summary["new_palocs"] > 0:
                    update_parts.append(f"{update_summary['new_palocs']} new language files")
                if update_summary["changed_palocs"] > 0:
                    update_parts.append("game files modified since last session")
                update_msg = ", ".join(update_parts)
                self._status_label.setText(
                    f"Game loaded: {len(groups)} groups, {paloc_count} languages | "
                    f"Updates detected: {update_msg}"
                )
                logger.info("Game updates detected: %s", update_msg)

            self._show_main_tabs()
            logger.info(
                "Game activated: %s (%d groups, %d palocs) version=%s",
                self._packages_path, len(groups), paloc_count, self._game_version,
            )
        except Exception as e:
            self._on_load_error(str(e))

    def _on_load_error(self, error_msg: str):
        self._lock_tabs()
        self._show_main_tabs()
        self._game_loaded = False
        self._status_label.setText("Failed to load game")
        self._setup_status.setText(f"Failed to load game:\n{error_msg}")
        self._setup_status.setStyleSheet("color: #f38ba8;")
        logger.exception("Failed to activate game: %s", error_msg)
        show_error(self, "Load Error", error_msg)

    # ------------------------------------------------------------------
    # Per-tab lazy initialisation (called when a tab is first shown)
    # ------------------------------------------------------------------
    def _init_tab_from_game(self, index: int, widget: QWidget):
        """Initialise a single tab from game data. Called lazily on first click."""
        try:
            if index == 1:   # Explorer (usually already done in _on_load_finished)
                if not hasattr(widget, '_game_initialized'):
                    widget.initialize_from_game(self._vfs, self._all_groups)
                    widget.files_extracted.connect(self._on_files_extracted)
                    widget._game_initialized = True
            elif index == 2:  # Item Catalog
                widget.initialize_from_game(self._vfs)
            elif index == 3:  # Dialogue Catalog
                widget.initialize_from_game(self._vfs)
            elif index == 4:  # Repack
                widget.initialize_from_game(self._packages_path)
            elif index == 5:  # Translate
                widget.initialize_from_game(self._vfs, self._discovered_palocs)
            elif index == 6:  # Audio
                widget.initialize_from_game(self._vfs, self._all_groups)
            elif index == 7:  # Font
                widget.initialize_from_game(self._vfs)
        except Exception as e:
            logger.exception("Failed to initialise tab %d: %s", index, e)
            show_error(self, "Tab Init Error", f"Failed to initialise {_TAB_REGISTRY[index]['label']}: {e}")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _on_files_extracted(self, output_path: str):
        self._status_label.setText(f"Extracted to: {output_path}")

    def _apply_theme(self, theme_name: str):
        if theme_name == "light":
            QApplication.instance().setStyleSheet(LIGHT_THEME)
        else:
            QApplication.instance().setStyleSheet(DARK_THEME)
        self._config.set("general.theme", theme_name)

    def _on_settings_changed(self):
        translate_tab = self._tab(5)
        if translate_tab and hasattr(translate_tab, 'refresh_from_settings'):
            translate_tab.refresh_from_settings()
        audio_tab = self._tab(6)
        if audio_tab and hasattr(audio_tab, 'refresh_from_settings'):
            try:
                audio_tab.refresh_from_settings()
            except Exception:
                pass
        self._status_label.setText("Settings updated")

    def closeEvent(self, event):
        translate_tab = self._tab(5)
        if translate_tab and hasattr(translate_tab, 'save_state'):
            try:
                translate_tab.save_state()
            except Exception as e:
                logger.error("Failed to save translation state: %s", e)
        try:
            self._config.save()
        except Exception as e:
            logger.error("Failed to save config on exit: %s", e)
        event.accept()
