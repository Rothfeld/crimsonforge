"""Main application window with tab bar and game discovery flow.

On first launch, only the Game Setup tab is active. The user browses
or auto-discovers the game packages directory. After the game is loaded:
- VFS is built, all PAMT indices scanned
- All paloc localization files are discovered
- All tabs are unlocked and auto-populated with game data
- No tab requires the user to browse for game paths again

Tabs: Game Setup | Explorer (Unpack+Browse+Edit) | Repack | Translate | Font Builder | Settings | About
"""

import os
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
from ui.tab_explorer import ExplorerTab
from ui.tab_repack import RepackTab
from ui.tab_translate import TranslateTab
from ui.tab_font import FontTab
from ui.tab_settings import SettingsTab
from ui.tab_audio import AudioTab
from ui.tab_about import AboutTab
from ui.themes.dark import DARK_THEME
from ui.themes.light import LIGHT_THEME
from ui.dialogs.confirmation import show_error
from ui.dialogs.file_picker import pick_directory
from version import APP_VERSION, APP_NAME
from utils.logger import get_logger

logger = get_logger("ui.main_window")


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

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - Crimson Desert Modding Studio")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        self._central_stack = QStackedWidget()
        self._tabs = QTabWidget()
        self._loading_page = self._build_loading_page()
        self._central_stack.addWidget(self._tabs)
        self._central_stack.addWidget(self._loading_page)
        self.setCentralWidget(self._central_stack)

        self._setup_tab = self._build_setup_tab()
        self._explorer_tab = ExplorerTab(config)
        self._repack_tab = RepackTab(config)
        self._translate_tab = TranslateTab(config, registry)
        self._audio_tab = AudioTab(config)
        self._font_tab = FontTab(config)
        self._settings_tab = SettingsTab(config, registry)
        self._about_tab = AboutTab(config=config)

        self._tabs.addTab(self._setup_tab, "Game Setup")
        self._tabs.addTab(self._explorer_tab, "Explorer")
        self._tabs.addTab(self._repack_tab, "Repack")
        self._tabs.addTab(self._translate_tab, "Translate")
        self._tabs.addTab(self._audio_tab, "Audio")
        self._tabs.addTab(self._font_tab, "Font Builder")
        self._tabs.addTab(self._settings_tab, "Settings")
        self._tabs.addTab(self._about_tab, "About")

        self._explorer_tab.files_extracted.connect(self._on_files_extracted)
        self._settings_tab.theme_changed.connect(self._apply_theme)
        self._settings_tab.settings_changed.connect(self._on_settings_changed)

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
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedWidth(320)
        self._loading_bar.setFixedHeight(18)
        layout.addWidget(self._loading_bar, 0, Qt.AlignHCenter)

        outer.addWidget(container, 0, Qt.AlignCenter)
        outer.addStretch()
        return widget

    def _show_loading_screen(self, title: str, detail: str = "") -> None:
        self._loading_title.setText(title)
        self._loading_detail.setText(detail)
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
            # Auto-load the game immediately
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

    def _detect_game_version(self, packages_path: str) -> str:
        """Detect game version from PAPGT metadata and file stats."""
        try:
            papgt_path = os.path.join(packages_path, "meta", "0.papgt")
            if not os.path.isfile(papgt_path):
                return "Unknown"
            stat = os.stat(papgt_path)
            size = stat.st_size
            from datetime import datetime
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            # Compute CRC fingerprint for version identification
            from core.checksum_engine import pa_checksum
            with open(papgt_path, "rb") as f:
                data = f.read()
            crc = pa_checksum(data[12:]) if len(data) > 12 else 0
            return f"CRC:0x{crc:08X} | Modified:{mod_time} | Size:{size:,}B"
        except Exception as e:
            logger.warning("Failed to detect game version: %s", e)
            return "Unknown"

    def _check_game_updates(self, packages_path: str, groups: list[str]) -> dict:
        """Check for new/changed files since last session. Returns a summary dict."""
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

            # Check game fingerprint for any changes
            saved_fp = self._config.get("game.last_fingerprint", "")
            papgt_path = os.path.join(packages_path, "meta", "0.papgt")
            if os.path.isfile(papgt_path):
                from core.checksum_engine import pa_checksum
                with open(papgt_path, "rb") as f:
                    data = f.read()
                crc = pa_checksum(data[12:]) if len(data) > 12 else 0
                current_fp = f"{crc:08X}_{os.path.getsize(papgt_path)}"
                if saved_fp and current_fp != saved_fp:
                    summary["changed_palocs"] = 1  # game files changed
                self._config.set("game.last_fingerprint", current_fp)

            self._config.set("game.last_paloc_count", current_paloc_count)
            self._config.set("game.last_group_count", current_group_count)
        except Exception as e:
            logger.warning("Failed to check game updates: %s", e)
        return summary

    def _activate_game(self, packages_path: str) -> None:
        """Load game data and unlock all tabs with auto-populated data."""
        try:
            self._packages_path = packages_path
            self._config.set("general.last_game_path", packages_path)
            self._config.save()

            self._show_loading_screen(
                "Loading Crimson Desert...",
                "Reading package groups.",
            )
            self._vfs = VfsManager(packages_path)
            groups = self._vfs.list_package_groups()

            self._show_loading_screen(
                "Loading Crimson Desert...",
                f"Scanning localization files across {len(groups)} package groups.",
            )
            self._discovered_palocs = self._scan_paloc_files(packages_path, groups)

            # Detect game version
            self._show_loading_screen(
                "Loading Crimson Desert...",
                "Detecting game version and checking for updates.",
            )
            game_version = self._detect_game_version(packages_path)
            update_summary = self._check_game_updates(packages_path, groups)

            self._show_loading_screen(
                "Loading Crimson Desert...",
                "Building the Explorer file index.",
            )
            self._explorer_tab.initialize_from_game(self._vfs, groups)

            self._show_loading_screen(
                "Loading Crimson Desert...",
                "Initializing Repack, Translate, and Font tools.",
            )
            self._repack_tab.initialize_from_game(packages_path)
            self._translate_tab.initialize_from_game(self._vfs, self._discovered_palocs)
            self._font_tab.initialize_from_game(self._vfs)
            self._audio_tab.initialize_from_game(self._vfs, groups)

            self._unlock_tabs()
            self._game_loaded = True

            self._show_loading_screen(
                "Loading Crimson Desert...",
                "Restoring your last translation session.",
            )
            if self._translate_tab.restore_state():
                self._tabs.setCurrentIndex(3)
            else:
                self._tabs.setCurrentIndex(1)

            paloc_count = len(self._discovered_palocs)
            self._game_version_label.setText(f"Game: {game_version}")
            self._status_label.setText(
                f"Game loaded: {len(groups)} package groups, {paloc_count} localization files"
            )
            self._files_label.setText(f"Groups: {len(groups)} | Languages: {paloc_count}")

            # Show update notification if game files changed
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
                packages_path, len(groups), paloc_count, game_version,
            )
        except Exception as e:
            self._lock_tabs()
            self._show_main_tabs()
            self._game_loaded = False
            self._status_label.setText("Failed to load game")
            self._setup_status.setText(f"Failed to load game:\n{e}")
            self._setup_status.setStyleSheet("color: #f38ba8;")
            logger.exception("Failed to activate game: %s", packages_path)
            show_error(self, "Load Error", str(e))

    def _scan_paloc_files(self, packages_path: str, groups: list[str]) -> list[dict]:
        """Scan all package groups for .paloc localization files."""
        paloc_lang_map = {
            "eng": "en", "kor": "ko", "jpn": "ja", "rus": "ru",
            "tur": "tr", "spa-es": "es", "spa-mx": "es-MX",
            "fre": "fr", "ger": "de", "ita": "it", "pol": "pl",
            "por-br": "pt-BR", "zho-tw": "zh-TW", "zho-cn": "zh",
            "tha": "th", "vie": "vi", "ind": "id", "ara": "ar",
        }
        results = []
        for i, group in enumerate(groups, start=1):
            if i == 1 or i == len(groups) or i % 4 == 0:
                self._show_loading_screen(
                    "Loading Crimson Desert...",
                    f"Scanning localization files: group {group} ({i}/{len(groups)}).",
                )
            try:
                pamt = self._vfs.load_pamt(group)
                for entry in pamt.file_entries:
                    if entry.path.lower().endswith(".paloc"):
                        basename = os.path.basename(entry.path)
                        name_part = basename.replace("localizationstring_", "").replace(".paloc", "")
                        lang_code = paloc_lang_map.get(name_part, name_part)
                        results.append({
                            "filename": basename,
                            "lang_code": lang_code,
                            "lang_key": name_part,
                            "group": group,
                            "entry": entry,
                        })
            except Exception as e:
                logger.warning("Error scanning group %s for palocs: %s", group, e)
        return results

    def _on_files_extracted(self, output_path: str):
        self._status_label.setText(f"Extracted to: {output_path}")

    def _apply_theme(self, theme_name: str):
        if theme_name == "light":
            QApplication.instance().setStyleSheet(LIGHT_THEME)
        else:
            QApplication.instance().setStyleSheet(DARK_THEME)
        self._config.set("general.theme", theme_name)

    def _on_settings_changed(self):
        self._status_label.setText("Settings updated")

    def closeEvent(self, event):
        try:
            self._translate_tab.save_state()
        except Exception as e:
            logger.error("Failed to save translation state: %s", e)
        try:
            self._config.save()
        except Exception as e:
            logger.error("Failed to save config on exit: %s", e)
        event.accept()
