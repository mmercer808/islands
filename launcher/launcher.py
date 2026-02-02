"""
PyLauncher - Drag & Drop Python Script Runner
Drop .py files to execute with captured output, environment management, and history tracking.
"""
## python.ex
import sys
import os
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QListWidget, QListWidgetItem, QSplitter,
    QLabel, QPushButton, QFrame, QMenu, QToolBar, QStatusBar,
    QMessageBox, QFileDialog
)
from PySide6.QtCore import (
    Qt, QProcess, QMimeData, Signal, Slot, QTimer, QSize
)
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QFont, QColor, QPalette,
    QAction, QIcon, QTextCharFormat, QTextCursor
)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class LaunchRecord:
    """Record of a script launch."""
    path: str
    timestamp: str
    exit_code: Optional[int] = None
    venv_used: Optional[str] = None
    requirements_installed: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LaunchRecord':
        return cls(**data)


class HistoryManager:
    """Manages persistent history of launched scripts."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.records: list[LaunchRecord] = []
        self._load()
    
    def _load(self):
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                self.records = [LaunchRecord.from_dict(r) for r in data]
            except (json.JSONDecodeError, KeyError):
                self.records = []
    
    def _save(self):
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.records]
        self.history_file.write_text(json.dumps(data, indent=2))
    
    def add(self, record: LaunchRecord):
        self.records.insert(0, record)
        # Keep last 100 entries
        self.records = self.records[:100]
        self._save()
    
    def update_last(self, exit_code: int):
        if self.records:
            self.records[0].exit_code = exit_code
            self._save()
    
    def get_recent(self, limit: int = 20) -> list[LaunchRecord]:
        return self.records[:limit]
    
    def clear(self):
        self.records = []
        self._save()


# ============================================================================
# Environment Manager
# ============================================================================

class EnvironmentManager:
    """Manages Python environments and requirements."""
    
    def __init__(self, envs_dir: Path):
        self.envs_dir = envs_dir
        self.envs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_project_env_dir(self, script_path: Path) -> Path:
        """Get the venv directory for a project."""
        # Use parent directory name + hash for uniqueness
        project_dir = script_path.parent
        env_name = f"{project_dir.name}_{hash(str(project_dir)) % 10000:04d}"
        return self.envs_dir / env_name
    
    def find_requirements(self, script_path: Path) -> Optional[Path]:
        """Find requirements.txt in script directory or parent."""
        search_dirs = [script_path.parent, script_path.parent.parent]
        for d in search_dirs:
            req_file = d / "requirements.txt"
            if req_file.exists():
                return req_file
        return None
    
    def find_existing_venv(self, script_path: Path) -> Optional[Path]:
        """Find an existing venv in project directory."""
        project_dir = script_path.parent
        venv_names = ["venv", ".venv", "env", ".env"]
        for name in venv_names:
            venv_path = project_dir / name
            python_exe = self._get_python_in_venv(venv_path)
            if python_exe and python_exe.exists():
                return venv_path
        return None
    
    def _get_python_in_venv(self, venv_path: Path) -> Optional[Path]:
        """Get Python executable path in a venv."""
        if sys.platform == "win32":
            python = venv_path / "Scripts" / "python.exe"
        else:
            python = venv_path / "bin" / "python"
        return python if python.exists() else None
    
    def create_venv(self, env_dir: Path) -> bool:
        """Create a new virtual environment."""
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(env_dir)],
                check=True, capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_python_executable(self, script_path: Path, use_venv: bool = True) -> tuple[str, Optional[Path]]:
        """
        Get the Python executable to use for a script.
        Returns (python_path, venv_path_if_used)
        """
        if not use_venv:
            return sys.executable, None
        
        # Check for existing project venv first
        existing_venv = self.find_existing_venv(script_path)
        if existing_venv:
            python = self._get_python_in_venv(existing_venv)
            if python:
                return str(python), existing_venv
        
        # Use system Python
        return sys.executable, None
    
    def install_requirements(self, python_exe: str, requirements_path: Path) -> tuple[bool, str]:
        """
        Install requirements using pip.
        Returns (success, output_message)
        """
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", "-r", str(requirements_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Installation timed out after 5 minutes"
        except Exception as e:
            return False, str(e)
    
    def check_requirements_installed(self, python_exe: str, requirements_path: Path) -> bool:
        """Check if requirements are already satisfied."""
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", "--dry-run", "-r", str(requirements_path)],
                capture_output=True,
                text=True
            )
            # If dry-run shows "Would install", requirements aren't met
            return "Would install" not in result.stdout
        except:
            return False


# ============================================================================
# Output Console Widget
# ============================================================================

class OutputConsole(QPlainTextEdit):
    """Console widget for displaying script output."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        
        # Monospace font
        font = QFont("Consolas", 10)
        if not font.exactMatch():
            font = QFont("Monaco", 10)
        if not font.exactMatch():
            font = QFont("Courier New", 10)
        self.setFont(font)
        
        # Dark theme
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: none;
                padding: 8px;
            }
        """)
        
        # Text formats
        self.stdout_format = QTextCharFormat()
        self.stdout_format.setForeground(QColor("#d4d4d4"))
        
        self.stderr_format = QTextCharFormat()
        self.stderr_format.setForeground(QColor("#f48771"))
        
        self.info_format = QTextCharFormat()
        self.info_format.setForeground(QColor("#569cd6"))
        
        self.success_format = QTextCharFormat()
        self.success_format.setForeground(QColor("#4ec9b0"))
    
    def append_stdout(self, text: str):
        self._append_formatted(text, self.stdout_format)
    
    def append_stderr(self, text: str):
        self._append_formatted(text, self.stderr_format)
    
    def append_info(self, text: str):
        self._append_formatted(text, self.info_format)
    
    def append_success(self, text: str):
        self._append_formatted(text, self.success_format)
    
    def _append_formatted(self, text: str, fmt: QTextCharFormat):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text, fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


# ============================================================================
# History Panel
# ============================================================================

class HistoryPanel(QFrame):
    """Panel showing launch history."""
    
    script_selected = Signal(str)  # Emits script path
    
    def __init__(self, history_manager: HistoryManager, parent=None):
        super().__init__(parent)
        self.history_manager = history_manager
        self._setup_ui()
        self.refresh()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("📜 History")
        header.setStyleSheet("""
            QLabel {
                color: #569cd6;
                font-weight: bold;
                padding: 8px;
                background: #252526;
            }
        """)
        layout.addWidget(header)
        
        # List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:hover {
                background-color: #2a2d2e;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
        """)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.list_widget)
        
        # Clear button
        clear_btn = QPushButton("Clear History")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #333;
                color: #d4d4d4;
                border: none;
                padding: 6px;
            }
            QPushButton:hover {
                background: #444;
            }
        """)
        clear_btn.clicked.connect(self._clear_history)
        layout.addWidget(clear_btn)
    
    def refresh(self):
        self.list_widget.clear()
        for record in self.history_manager.get_recent():
            # Format display
            name = Path(record.path).name
            time = record.timestamp[:16].replace("T", " ")
            
            status = "⏳"
            if record.exit_code is not None:
                status = "✅" if record.exit_code == 0 else "❌"
            
            text = f"{status} {name}\n   {time}"
            
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, record.path)
            item.setToolTip(record.path)
            self.list_widget.addItem(item)
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        if path and Path(path).exists():
            self.script_selected.emit(path)
    
    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #454545;
            }
            QMenu::item:selected {
                background: #094771;
            }
        """)
        
        run_action = menu.addAction("▶️ Run Again")
        open_folder = menu.addAction("📁 Open Folder")
        
        action = menu.exec_(self.list_widget.mapToGlobal(pos))
        path = item.data(Qt.UserRole)
        
        if action == run_action and path:
            self.script_selected.emit(path)
        elif action == open_folder and path:
            folder = Path(path).parent
            if sys.platform == "darwin":
                subprocess.run(["open", str(folder)])
            elif sys.platform == "win32":
                subprocess.run(["explorer", str(folder)])
            else:
                subprocess.run(["xdg-open", str(folder)])
    
    def _clear_history(self):
        reply = QMessageBox.question(
            self, "Clear History",
            "Clear all launch history?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.history_manager.clear()
            self.refresh()


# ============================================================================
# Drop Zone Widget
# ============================================================================

class DropZone(QFrame):
    """Central drop zone for drag & drop."""
    
    file_dropped = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 200)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setStyleSheet("""
            DropZone {
                background-color: #252526;
                border: 2px dashed #454545;
                border-radius: 8px;
            }
            DropZone:hover {
                border-color: #569cd6;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        icon_label = QLabel("🐍")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        text_label = QLabel("Drop Python File Here")
        text_label.setStyleSheet("""
            color: #888;
            font-size: 16px;
            font-weight: bold;
        """)
        text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(text_label)
        
        hint_label = QLabel("or double-click history item to re-run")
        hint_label.setStyleSheet("color: #666; font-size: 11px;")
        hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_label)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().endswith('.py') for url in urls):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    DropZone {
                        background-color: #1e3a1e;
                        border: 2px dashed #4ec9b0;
                        border-radius: 8px;
                    }
                """)
    
    def dragLeaveEvent(self, event):
        self._setup_ui()  # Reset style
    
    def dropEvent(self, event: QDropEvent):
        self._setup_ui()  # Reset style
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.endswith('.py'):
                self.file_dropped.emit(path)
                break


# ============================================================================
# Main Window
# ============================================================================

class PyLauncher(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Data directory
        self.data_dir = Path.home() / ".pylauncher"
        self.data_dir.mkdir(exist_ok=True)
        
        # Managers
        self.history_manager = HistoryManager(self.data_dir / "history.json")
        self.env_manager = EnvironmentManager(self.data_dir / "envs")
        
        # Process
        self.process: Optional[QProcess] = None
        self.current_script: Optional[Path] = None
        
        self._setup_ui()
        self._setup_process()
    
    def _setup_ui(self):
        self.setWindowTitle("PyLauncher")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QSplitter::handle {
                background: #333;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left: History panel
        self.history_panel = HistoryPanel(self.history_manager)
        self.history_panel.setFixedWidth(250)
        self.history_panel.script_selected.connect(self.run_script)
        splitter.addWidget(self.history_panel)
        
        # Right: Main area
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        splitter.addWidget(right_widget)
        
        # Toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.script_label = QLabel("No script loaded")
        self.script_label.setStyleSheet("color: #888; font-size: 12px;")
        toolbar_layout.addWidget(self.script_label)
        
        toolbar_layout.addStretch()
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #6e2020;
                color: #fff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #8e3030;
            }
            QPushButton:disabled {
                background: #333;
                color: #666;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_process)
        toolbar_layout.addWidget(self.stop_btn)
        
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: #0e639c;
                color: #fff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #1177bb;
            }
            QPushButton:disabled {
                background: #333;
                color: #666;
            }
        """)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._rerun_script)
        toolbar_layout.addWidget(self.run_btn)
        
        open_btn = QPushButton("📂 Open")
        open_btn.setStyleSheet("""
            QPushButton {
                background: #333;
                color: #d4d4d4;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #444;
            }
        """)
        open_btn.clicked.connect(self._open_file_dialog)
        toolbar_layout.addWidget(open_btn)
        
        right_layout.addWidget(toolbar)
        
        # Stacked: Drop zone / Console
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.run_script)
        
        self.console = OutputConsole()
        self.console.hide()
        
        right_layout.addWidget(self.drop_zone)
        right_layout.addWidget(self.console)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #252526;
                color: #888;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - drop a .py file to run")
    
    def _setup_process(self):
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
    
    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Python Script",
            str(Path.home()),
            "Python Files (*.py)"
        )
        if path:
            self.run_script(path)
    
    @Slot(str)
    def run_script(self, script_path: str):
        """Run a Python script."""
        path = Path(script_path)
        
        if not path.exists():
            QMessageBox.warning(self, "Error", f"File not found:\n{path}")
            return
        
        # Stop any running process
        if self.process.state() == QProcess.Running:
            self.process.kill()
            self.process.waitForFinished()
        
        self.current_script = path
        
        # Show console, hide drop zone
        self.drop_zone.hide()
        self.console.show()
        self.console.clear()
        
        # Update UI
        self.script_label.setText(f"📄 {path.name}")
        self.script_label.setToolTip(str(path))
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Log info
        self.console.append_info(f"═══ PyLauncher ═══\n")
        self.console.append_info(f"Script: {path}\n")
        self.console.append_info(f"Directory: {path.parent}\n")
        
        # Find environment
        python_exe, venv_path = self.env_manager.get_python_executable(path)
        if venv_path:
            self.console.append_info(f"Using venv: {venv_path}\n")
        
        # Check for requirements
        req_file = self.env_manager.find_requirements(path)
        requirements_installed = False
        if req_file:
            self.console.append_info(f"Found requirements: {req_file}\n")
            
            # Check if requirements need to be installed
            if not self.env_manager.check_requirements_installed(python_exe, req_file):
                reply = QMessageBox.question(
                    self, "Install Requirements?",
                    f"Found requirements.txt that may need installation:\n{req_file}\n\nInstall now?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.console.append_info("Installing requirements...\n")
                    QApplication.processEvents()  # Update UI
                    success, output = self.env_manager.install_requirements(python_exe, req_file)
                    if success:
                        self.console.append_success("Requirements installed successfully.\n")
                        requirements_installed = True
                    else:
                        self.console.append_stderr(f"Failed to install requirements:\n{output}\n")
        
        self.console.append_info(f"Python: {python_exe}\n")
        self.console.append_info(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
        self.console.append_info(f"{'═' * 40}\n\n")
        
        # Create history record
        record = LaunchRecord(
            path=str(path),
            timestamp=datetime.now().isoformat(),
            venv_used=str(venv_path) if venv_path else None,
            requirements_installed=requirements_installed
        )
        self.history_manager.add(record)
        self.history_panel.refresh()
        
        # Start process
        self.process.setWorkingDirectory(str(path.parent))
        
        # Set up environment
        env = self.process.processEnvironment()
        if env.isEmpty():
            env = self.process.systemEnvironment()
        
        # Unbuffered output
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        
        self.status_bar.showMessage(f"Running: {path.name}")
        self.process.start(python_exe, [str(path)])
    
    def _rerun_script(self):
        if self.current_script:
            self.run_script(str(self.current_script))
    
    def _stop_process(self):
        if self.process.state() == QProcess.Running:
            self.process.kill()
            self.console.append_stderr("\n\n[Process killed by user]\n")
    
    def _read_stdout(self):
        data = self.process.readAllStandardOutput().data()
        text = data.decode('utf-8', errors='replace')
        self.console.append_stdout(text)
    
    def _read_stderr(self):
        data = self.process.readAllStandardError().data()
        text = data.decode('utf-8', errors='replace')
        self.console.append_stderr(text)
    
    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        
        # Update history
        self.history_manager.update_last(exit_code)
        self.history_panel.refresh()
        
        # Log completion
        self.console.append_info(f"\n{'═' * 40}\n")
        if exit_code == 0:
            self.console.append_success(f"✓ Completed successfully (exit code: {exit_code})\n")
            self.status_bar.showMessage("Completed successfully")
        else:
            self.console.append_stderr(f"✗ Exited with code: {exit_code}\n")
            self.status_bar.showMessage(f"Exited with code: {exit_code}")
    
    def _process_error(self, error: QProcess.ProcessError):
        error_msgs = {
            QProcess.FailedToStart: "Failed to start process",
            QProcess.Crashed: "Process crashed",
            QProcess.Timedout: "Process timed out",
            QProcess.WriteError: "Write error",
            QProcess.ReadError: "Read error",
            QProcess.UnknownError: "Unknown error",
        }
        msg = error_msgs.get(error, "Unknown error")
        self.console.append_stderr(f"\n[Error: {msg}]\n")
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
    
    def closeEvent(self, event):
        if self.process.state() == QProcess.Running:
            reply = QMessageBox.question(
                self, "Process Running",
                "A script is still running. Kill it and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished()
        event.accept()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1e1e1e"))
    palette.setColor(QPalette.WindowText, QColor("#d4d4d4"))
    palette.setColor(QPalette.Base, QColor("#252526"))
    palette.setColor(QPalette.AlternateBase, QColor("#2d2d2d"))
    palette.setColor(QPalette.Text, QColor("#d4d4d4"))
    palette.setColor(QPalette.Button, QColor("#333"))
    palette.setColor(QPalette.ButtonText, QColor("#d4d4d4"))
    palette.setColor(QPalette.Highlight, QColor("#094771"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    
    window = PyLauncher()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
