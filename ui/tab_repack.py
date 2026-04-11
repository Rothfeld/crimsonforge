"""Repack tab - modify and repack files with full checksum chain.

Game path is set by the main window on startup. User only needs to
select modified files directory and click repack.
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QHeaderView, QGroupBox,
)
from PySide6.QtCore import Qt

from core.repack_engine import RepackEngine, ModifiedFile
from core.pamt_parser import parse_pamt, find_file_entry
from ui.widgets.progress_widget import ProgressWidget
from ui.dialogs.file_picker import pick_directory
from ui.dialogs.confirmation import show_error, show_info, confirm_action
from utils.thread_worker import FunctionWorker
from utils.platform_utils import format_file_size
from utils.logger import get_logger

logger = get_logger("ui.tab_repack")


class RepackTab(QWidget):
    """Tab for repacking modified files into game archives."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._game_path = ""
        self._worker: FunctionWorker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Modified Files:"))
        self._source_path = QLineEdit()
        self._source_path.setPlaceholderText("Directory containing modified files to repack...")
        src_row.addWidget(self._source_path, 1)
        src_browse = QPushButton("Browse...")
        src_browse.clicked.connect(self._browse_source)
        src_row.addWidget(src_browse)
        layout.addLayout(src_row)

        backup_row = QHBoxLayout()
        backup_row.addWidget(QLabel("Backup Dir:"))
        self._backup_path = QLineEdit(self._config.get("repack.backup_dir", ""))
        self._backup_path.setPlaceholderText("Directory for backups (auto-created if empty)...")
        backup_row.addWidget(self._backup_path, 1)
        bk_browse = QPushButton("Browse...")
        bk_browse.clicked.connect(self._browse_backup)
        backup_row.addWidget(bk_browse)
        layout.addLayout(backup_row)

        self._file_tree = QTreeWidget()
        self._file_tree.setHeaderLabels(["", "File", "Size", "Target PAZ", "Status"])
        self._file_tree.setUniformRowHeights(True)
        self._file_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self._file_tree.setAlternatingRowColors(True)
        layout.addWidget(self._file_tree, 1)

        scan_btn = QPushButton("Scan Modified Files")
        scan_btn.clicked.connect(self._scan_files)
        layout.addWidget(scan_btn)

        crc_group = QGroupBox("Checksum Chain Status")
        crc_layout = QHBoxLayout(crc_group)
        self._paz_crc_label = QLabel("PAZ CRC:  Pending")
        self._pamt_crc_label = QLabel("PAMT CRC: Pending")
        self._papgt_crc_label = QLabel("PAPGT CRC: Pending")
        crc_layout.addWidget(self._paz_crc_label)
        crc_layout.addWidget(self._pamt_crc_label)
        crc_layout.addWidget(self._papgt_crc_label)
        layout.addWidget(crc_group)

        opt_row = QHBoxLayout()
        self._backup_check = QCheckBox("Create Backup")
        self._backup_check.setChecked(self._config.get("repack.auto_backup", True))
        opt_row.addWidget(self._backup_check)
        self._verify_check = QCheckBox("Verify After Repack")
        self._verify_check.setChecked(self._config.get("repack.verify_after_repack", True))
        opt_row.addWidget(self._verify_check)
        self._timestamp_check = QCheckBox("Preserve Timestamps")
        self._timestamp_check.setChecked(self._config.get("repack.preserve_timestamps", True))
        opt_row.addWidget(self._timestamp_check)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        btn_row = QHBoxLayout()
        self._repack_btn = QPushButton("Repack Selected")
        self._repack_btn.setObjectName("primary")
        self._repack_btn.clicked.connect(self._repack)
        btn_row.addWidget(self._repack_btn)
        restore_btn = QPushButton("Restore Backup")
        restore_btn.clicked.connect(self._restore_backup)
        btn_row.addWidget(restore_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._progress = ProgressWidget()
        layout.addWidget(self._progress)

    def initialize_from_game(self, packages_path: str) -> None:
        """Called by main_window after game is loaded."""
        self._game_path = packages_path

    def set_game_path(self, path: str):
        """Backward compat alias."""
        self._game_path = path

    def _browse_source(self):
        path = pick_directory(self, "Select Modified Files Directory")
        if path:
            self._source_path.setText(path)

    def _browse_backup(self):
        path = pick_directory(self, "Select Backup Directory")
        if path:
            self._backup_path.setText(path)

    def _scan_files(self):
        src = self._source_path.text().strip()
        if not src or not os.path.isdir(src):
            show_error(self, "Error", "Select a valid directory with modified files.")
            return
        self._file_tree.clear()
        for root, dirs, files in os.walk(src):
            for fname in files:
                fpath = os.path.join(root, fname)
                size = os.path.getsize(fpath)
                item = QTreeWidgetItem()
                item.setCheckState(0, Qt.Checked)
                item.setText(1, os.path.relpath(fpath, src))
                item.setText(2, format_file_size(size))
                item.setText(3, "Auto-detect")
                item.setText(4, "Ready")
                item.setData(0, Qt.UserRole, fpath)
                self._file_tree.addTopLevelItem(item)
        self._progress.set_status(f"Found {self._file_tree.topLevelItemCount()} files")

    def _repack(self):
        if not self._game_path or not os.path.isdir(self._game_path):
            show_error(self, "Error",
                       "Game path not set. Return to Game Setup tab and load the game first.")
            return

        checked_files = []
        for i in range(self._file_tree.topLevelItemCount()):
            item = self._file_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                checked_files.append(item.data(0, Qt.UserRole))

        if not checked_files:
            show_error(self, "Error", "No files selected for repacking.")
            return

        if not confirm_action(self, "Confirm Repack",
                              f"Repack {len(checked_files)} files into game archives?\n"
                              f"This will modify game files."):
            return

        self._repack_btn.setEnabled(False)
        self._progress.set_progress(0, "Resolving file entries...")

        papgt_path = os.path.join(self._game_path, "meta", "0.papgt")
        if not os.path.isfile(papgt_path):
            show_error(self, "Error",
                       f"PAPGT root index not found: {papgt_path}\n"
                       f"Ensure the game is loaded correctly.")
            self._repack_btn.setEnabled(True)
            return

        modified_list = []
        resolve_errors = []
        for file_path in checked_files:
            basename = os.path.basename(file_path)
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
            except OSError as e:
                resolve_errors.append(f"Cannot read {basename}: {e}")
                continue

            pamt_data, entry = self._resolve_file_entry(basename)
            if not pamt_data or not entry:
                resolve_errors.append(
                    f"Cannot locate {basename} in any PAMT index. "
                    f"The file must match a known game file name."
                )
                continue

            group_dir = os.path.basename(os.path.dirname(pamt_data.path))
            modified_list.append(ModifiedFile(
                data=file_data,
                entry=entry,
                pamt_data=pamt_data,
                package_group=group_dir,
            ))

        if resolve_errors:
            error_text = "\n".join(resolve_errors)
            if not modified_list:
                show_error(self, "Resolve Error", f"No files could be resolved:\n{error_text}")
                self._repack_btn.setEnabled(True)
                return
            if not confirm_action(self, "Partial Resolve",
                                  f"{len(resolve_errors)} file(s) could not be resolved:\n"
                                  f"{error_text}\n\nContinue with {len(modified_list)} resolved file(s)?"):
                self._repack_btn.setEnabled(True)
                return

        backup_dir = self._backup_path.text().strip()
        engine = RepackEngine(self._game_path, backup_dir=backup_dir)

        def do_repack(worker, _files=modified_list, _papgt=papgt_path, _engine=engine):
            def progress_cb(pct, msg):
                worker.report_progress(pct, msg)
            return _engine.repack(
                modified_files=_files,
                papgt_path=_papgt,
                create_backup=self._backup_check.isChecked(),
                verify_after=self._verify_check.isChecked(),
                preserve_timestamps=self._timestamp_check.isChecked(),
                progress_callback=progress_cb,
            )

        self._worker = FunctionWorker(do_repack)
        self._worker.progress.connect(lambda p, m: self._progress.set_progress(p, m))
        self._worker.finished_result.connect(self._on_repack_done)
        self._worker.error_occurred.connect(
            lambda e: (show_error(self, "Repack Error", e), self._repack_btn.setEnabled(True))
        )
        self._worker.start()

    def _resolve_file_entry(self, basename):
        for item in sorted(os.listdir(self._game_path)):
            group_dir = os.path.join(self._game_path, item)
            pamt_path = os.path.join(group_dir, "0.pamt")
            if not os.path.isfile(pamt_path):
                continue
            try:
                pamt_data = parse_pamt(pamt_path, paz_dir=group_dir)
                entry = find_file_entry(pamt_data, basename)
                if entry:
                    return pamt_data, entry
            except Exception as e:
                logger.warning("Error scanning %s: %s", item, e)
        return None, None

    def _on_repack_done(self, result):
        self._repack_btn.setEnabled(True)
        self._paz_crc_label.setText(f"PAZ CRC:  0x{result.paz_crc:08X}")
        self._pamt_crc_label.setText(f"PAMT CRC: 0x{result.pamt_crc:08X}")
        self._papgt_crc_label.setText(f"PAPGT CRC: 0x{result.papgt_crc:08X}")

        if result.success:
            self._progress.set_progress(100, f"Repack complete: {result.files_repacked} files")
            msg = f"Successfully repacked {result.files_repacked} files.\n"
            if result.backup_dir:
                msg += f"Backup: {result.backup_dir}\n"
            msg += (
                f"\nChecksum chain:\n"
                f"  PAZ:   0x{result.paz_crc:08X}\n"
                f"  PAMT:  0x{result.pamt_crc:08X}\n"
                f"  PAPGT: 0x{result.papgt_crc:08X}"
            )
            show_info(self, "Repack Complete", msg)
        else:
            error_text = "\n".join(result.errors) if result.errors else "Unknown error"
            self._progress.set_progress(100, f"Repack failed: {error_text}")
            show_error(self, "Repack Failed", error_text)

    def _restore_backup(self):
        backup_dir = self._backup_path.text().strip()
        if not backup_dir:
            backup_dir = pick_directory(self, "Select Backup Directory")
        if not backup_dir:
            return
        if confirm_action(self, "Restore Backup",
                          "Restore game files from this backup?\n"
                          "This will overwrite current game files."):
            try:
                from core.backup_manager import BackupManager
                bm = BackupManager(backup_dir)
                backups = bm.list_backups()
                if backups:
                    restored = bm.restore_backup(backups[0]["backup_dir"])
                    show_info(self, "Restore Complete", f"Restored {len(restored)} files.")
                else:
                    show_error(self, "Error", "No backups found in the selected directory.")
            except Exception as e:
                show_error(self, "Restore Error", str(e))
