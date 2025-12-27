#!/usr/bin/env python3
"""
AQUA v2 - Unified Application

Integrates:
- Document class for all text display
- DockLayout for flexible tabbed interface
- PromptStage for command input
- LLM streaming (from proc_streamer)
- Tab type registry for extensibility

No QSplitter sliders. Tabs in any position with drag-drop.
"""

from __future__ import annotations
import sys
import os
import json
import uuid
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Qt imports
from PySide6.QtCore import Qt, Signal, QObject, QSize, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QStatusBar, QLabel, QPushButton, QToolBar, 
    QMessageBox, QLineEdit, QComboBox, QMenu
)

# Our modules
from document import Document, EditableDocument, SpanStyle, Styles
from dock_layout import DockLayout, TabDockArea, DockPosition, TabTypeRegistry
from prompt_stage import PromptStage, PromptStyles

# Optional: requests for LLM
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    requests = None
    HAVE_REQUESTS = False


# =============================================================================
# SETTINGS
# =============================================================================

SETTINGS_FILE = Path("aqua_settings.json")

@dataclass
class Theme:
    name: str = "dark"
    font_family: str = "Consolas"
    font_size: int = 12
    accent: str = "#007acc"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def dark() -> Theme:
        return Theme(name="dark", font_family="Consolas", font_size=12, accent="#007acc")

    @staticmethod
    def light() -> Theme:
        return Theme(name="light", font_family="Consolas", font_size=12, accent="#2962ff")


class Settings:
    """Application settings."""
    
    def __init__(self):
        self.theme = Theme.dark()
        self.llm = {
            "provider": "ollama",
            "url": "http://127.0.0.1:11434",
            "model": "llama3:latest",
            "api_key": "",
            "timeout": 60,
            "stream": True,
        }
        self.load()
    
    def load(self):
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                if "theme" in data:
                    self.theme = Theme(**data["theme"])
                if "llm" in data:
                    self.llm.update(data["llm"])
            except Exception:
                pass
    
    def save(self):
        data = {
            "theme": self.theme.to_dict(),
            "llm": self.llm,
        }
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))


# =============================================================================
# LLM CHANNEL (from proc_streamer)
# =============================================================================

class LLMChannel(QObject):
    """
    LLM communication channel with streaming support.
    Adapted from proc_streamer's AssistChannel.
    """
    
    chunk = Signal(str)
    complete = Signal(str)
    error = Signal(str)
    status = Signal(bool, str)  # connected, message
    
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._connected = False
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    def connect_llm(self, url: str = None, model: str = None):
        """Connect to LLM service."""
        if url:
            self.settings.llm["url"] = url
        if model:
            self.settings.llm["model"] = model
        
        ok, msg = self._healthcheck()
        self._connected = ok
        self.status.emit(ok, msg)
    
    def disconnect_llm(self):
        """Disconnect from LLM."""
        self._connected = False
        self._stop_flag = True
        self.status.emit(False, "Disconnected")
    
    def query(self, prompt: str):
        """Send query to LLM."""
        if not self._connected:
            self.error.emit("Not connected to LLM")
            return
        if not HAVE_REQUESTS:
            self.error.emit("requests library not installed")
            return
        
        self._stop_flag = False
        self._thread = threading.Thread(
            target=self._run_query, 
            args=(prompt,), 
            daemon=True
        )
        self._thread.start()
    
    def _healthcheck(self) -> Tuple[bool, str]:
        """Check LLM service health."""
        if not HAVE_REQUESTS:
            return False, "Install 'requests' to connect"
        
        provider = self.settings.llm["provider"].lower()
        
        if provider == "ollama":
            try:
                url = self.settings.llm["url"].rstrip("/") + "/api/tags"
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json() or {}
                models = [m.get("name") for m in data.get("models", [])]
                model = self.settings.llm.get("model")
                
                if model and models:
                    if model not in models:
                        # Check partial match
                        partial = any(m.startswith(model + ":") or m == model for m in models)
                        if not partial:
                            return False, f"Model '{model}' not found. Available: {', '.join(models)}"
                
                return True, f"Connected to Ollama ({len(models)} models)"
            except Exception as e:
                return False, f"Ollama connect failed: {e}"
        else:
            # OpenAI-compatible
            try:
                url = self.settings.llm["url"].rstrip("/") + "/v1/models"
                headers = {}
                if self.settings.llm["api_key"]:
                    headers["Authorization"] = f"Bearer {self.settings.llm['api_key']}"
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                return True, f"Connected to OpenAI-compatible API"
            except Exception as e:
                return False, f"Connect failed: {e}"
    
    def _run_query(self, prompt: str):
        """Run query in thread."""
        provider = self.settings.llm["provider"].lower()
        try:
            if provider == "ollama":
                self._query_ollama(prompt)
            else:
                self._query_openai_compat(prompt)
        except Exception as e:
            self.error.emit(str(e))
    
    def _query_ollama(self, prompt: str):
        """Query Ollama API."""
        url = self.settings.llm["url"].rstrip("/") + "/api/generate"
        model = self.settings.llm["model"]
        payload = {"model": model, "prompt": prompt, "stream": True}
        
        with requests.post(url, json=payload, stream=True, timeout=self.settings.llm["timeout"]) as r:
            r.raise_for_status()
            full = []
            for line in r.iter_lines(decode_unicode=True):
                if self._stop_flag:
                    break
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        tok = obj["response"]
                        full.append(tok)
                        self.chunk.emit(tok)
                    if obj.get("done"):
                        break
                except Exception:
                    full.append(line)
                    self.chunk.emit(line)
            self.complete.emit("".join(full))
    
    def _query_openai_compat(self, prompt: str):
        """Query OpenAI-compatible API."""
        url = self.settings.llm["url"].rstrip("/") + "/v1/chat/completions"
        model = self.settings.llm["model"]
        headers = {}
        if self.settings.llm["api_key"]:
            headers["Authorization"] = f"Bearer {self.settings.llm['api_key']}"
        
        data = {
            "model": model, 
            "stream": True, 
            "messages": [{"role": "user", "content": prompt}]
        }
        
        with requests.post(url, headers=headers, json=data, stream=True, timeout=self.settings.llm["timeout"]) as r:
            r.raise_for_status()
            full = []
            for line in r.iter_lines(decode_unicode=True):
                if self._stop_flag:
                    break
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            full.append(delta)
                            self.chunk.emit(delta)
                    except Exception:
                        pass
            self.complete.emit("".join(full))


# =============================================================================
# REGISTER TAB TYPES
# =============================================================================

def register_tab_types():
    """Register available tab types."""
    
    # Document (read-only display)
    TabTypeRegistry.register("document", lambda: Document())
    
    # Editor (editable document)
    TabTypeRegistry.register("editor", lambda: EditableDocument())
    
    # Prompt
    TabTypeRegistry.register("prompt", lambda: PromptStage())
    
    # Assistant (read-only document for LLM responses)
    def create_assistant():
        doc = Document(read_only=True)
        doc.append_line("Assistant", Styles.INFO)
        doc.append_line("Ask questions via the prompt.", Styles.MUTED)
        return doc
    TabTypeRegistry.register("assistant", create_assistant)
    
    # Console (output only)
    def create_console():
        doc = Document(read_only=True)
        doc.append_line("Console Output", Styles.INFO)
        return doc
    TabTypeRegistry.register("console", create_console)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.settings = Settings()
        self.llm_channel = LLMChannel(self.settings)
        
        # Setup
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_connections()
        self._create_initial_tabs()
        self._apply_theme()
        
        # Auto-connect to LLM
        QTimer.singleShot(500, self._auto_connect_llm)
    
    def _setup_ui(self):
        """Setup main UI."""
        self.setWindowTitle("AQUA v2")
        self.resize(1400, 900)
        
        # Register tab types
        register_tab_types()
        
        # Create dock layout
        self.dock_layout = DockLayout()
        self.setCentralWidget(self.dock_layout)
    
    def _setup_toolbar(self):
        """Setup toolbar."""
        tb = QToolBar("Main")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        
        # File actions
        act_new = QAction("New", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._new_document)
        
        act_open = QAction("Open", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_document)
        
        act_save = QAction("Save", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_document)
        
        tb.addActions([act_new, act_open, act_save])
        tb.addSeparator()
        
        # Tab creation
        tb.addWidget(QLabel(" New: "))
        
        self.new_tab_combo = QComboBox()
        self.new_tab_combo.addItems(TabTypeRegistry.get_types())
        tb.addWidget(self.new_tab_combo)
        
        btn_add_tab = QPushButton("Add Tab")
        btn_add_tab.clicked.connect(self._add_tab_from_combo)
        tb.addWidget(btn_add_tab)
        
        tb.addSeparator()
        
        # LLM controls
        act_connect = QAction("Connect LLM", self)
        act_connect.triggered.connect(self._connect_llm)
        
        act_disconnect = QAction("Disconnect", self)
        act_disconnect.triggered.connect(self._disconnect_llm)
        
        tb.addActions([act_connect, act_disconnect])
        
        tb.addSeparator()
        
        # Theme toggle
        act_theme = QAction("Toggle Theme", self)
        act_theme.triggered.connect(self._toggle_theme)
        tb.addAction(act_theme)
    
    def _setup_statusbar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_msg = QLabel("Ready")
        self.status_llm = QLabel("LLM: Disconnected")
        
        self.status_bar.addWidget(self.status_msg)
        self.status_bar.addPermanentWidget(self.status_llm)
    
    def _setup_connections(self):
        """Setup signal connections."""
        # LLM channel
        self.llm_channel.status.connect(self._on_llm_status)
        self.llm_channel.chunk.connect(self._on_llm_chunk)
        self.llm_channel.complete.connect(self._on_llm_complete)
        self.llm_channel.error.connect(self._on_llm_error)
    
    def _create_initial_tabs(self):
        """Create initial tabs."""
        # Main editor in center
        editor = EditableDocument()
        editor.append_line("Welcome to AQUA v2", Styles.INFO)
        editor.append_line("")
        editor.append_line("This is an editable document.", Styles.DEFAULT)
        editor.append_line("Drag tabs between areas to rearrange.", Styles.MUTED)
        self.dock_layout.add_tab(editor, "Document", DockPosition.CENTER, "editor")
        
        # Assistant in right
        self.assistant = Document(read_only=True)
        self.assistant.append_line("Assistant", Styles.INFO)
        self.assistant.append_line("LLM responses appear here.", Styles.MUTED)
        self.dock_layout.add_tab(self.assistant, "Assistant", DockPosition.RIGHT, "assistant")
        
        # Prompt at bottom
        self.prompt = PromptStage()
        self._connect_prompt()
        self.dock_layout.add_tab(self.prompt, "Prompt", DockPosition.BOTTOM, "prompt")
        
        # Set sizes
        self.dock_layout.set_sizes(right_width=350, bottom_height=250)
    
    def _connect_prompt(self):
        """Connect prompt signals."""
        self.prompt.llm_query.connect(self._on_prompt_llm_query)
        self.prompt.system_command.connect(self._on_prompt_system_command)
        self.prompt.special_command.connect(self._on_prompt_special_command)
    
    # -------------------------------------------------------------------------
    # PROMPT HANDLERS
    # -------------------------------------------------------------------------
    
    def _on_prompt_llm_query(self, query: str):
        """Handle LLM query from prompt."""
        # Show in assistant
        self.assistant.append_line(f"You: {query}", Styles.USER_INPUT)
        self.assistant.append_text("🤖 ", Styles.INFO)
        
        # Start stream
        self._assistant_stream_id = self.assistant.stream_start(Styles.LLM_RESPONSE)
        
        # Send to LLM
        self.llm_channel.query(query)
    
    def _on_prompt_system_command(self, command: str):
        """Handle system command from prompt."""
        import subprocess
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if result.stdout:
                self.prompt.output_result(result.stdout)
            if result.stderr:
                self.prompt.output_error(result.stderr)
        except subprocess.TimeoutExpired:
            self.prompt.output_error("Command timed out")
        except Exception as e:
            self.prompt.output_error(str(e))
    
    def _on_prompt_special_command(self, command: str, args: tuple):
        """Handle special command from prompt."""
        if command == "connect":
            url = args[0] if args else None
            self._connect_llm(url)
        elif command == "disconnect":
            self._disconnect_llm()
        elif command == "model":
            if args:
                self.settings.llm["model"] = args[0]
                self.prompt.output_success(f"Model set to: {args[0]}")
            else:
                self.prompt.output_info(f"Current model: {self.settings.llm['model']}")
        elif command == "theme":
            self._toggle_theme()
        elif command == "new":
            tab_type = args[0] if args else "editor"
            self._create_tab(tab_type)
        else:
            self.prompt.output_error(f"Unknown command: /{command}")
    
    # -------------------------------------------------------------------------
    # LLM HANDLERS
    # -------------------------------------------------------------------------
    
    def _on_llm_status(self, connected: bool, message: str):
        """Handle LLM status change."""
        status = "Connected" if connected else "Disconnected"
        self.status_llm.setText(f"LLM: {status}")
        self.prompt.set_llm_connected(connected, message)
    
    def _on_llm_chunk(self, chunk: str):
        """Handle LLM chunk."""
        # To assistant
        if hasattr(self, '_assistant_stream_id'):
            self.assistant.stream_chunk(chunk, self._assistant_stream_id)
        # To prompt
        self.prompt.receive_llm_chunk(chunk)
    
    def _on_llm_complete(self, full_response: str):
        """Handle LLM complete."""
        # To assistant
        if hasattr(self, '_assistant_stream_id'):
            self.assistant.stream_complete(self._assistant_stream_id)
            self.assistant.append_line("")
            delattr(self, '_assistant_stream_id')
        # To prompt
        self.prompt.receive_llm_complete(full_response)
    
    def _on_llm_error(self, error: str):
        """Handle LLM error."""
        self.assistant.append_error(f"Error: {error}")
        self.prompt.receive_llm_error(error)
    
    # -------------------------------------------------------------------------
    # ACTIONS
    # -------------------------------------------------------------------------
    
    def _new_document(self):
        """Create new document."""
        editor = EditableDocument()
        self.dock_layout.add_tab(editor, "Untitled", DockPosition.CENTER, "editor")
    
    def _open_document(self):
        """Open document from file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", 
            "Text files (*.txt);;Markdown (*.md);;All files (*.*)"
        )
        if not path:
            return
        
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            editor = EditableDocument()
            editor.set_text(text)
            title = Path(path).name
            self.dock_layout.add_tab(editor, title, DockPosition.CENTER, "editor")
            self.status_msg.setText(f"Opened: {title}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")
    
    def _save_document(self):
        """Save current document."""
        # Get current center tab
        center = self.dock_layout.get_area(DockPosition.CENTER)
        if not center:
            return
        
        tab_id = center.current_tab_id()
        if not tab_id:
            return
        
        tab_info = center.get_tab_by_id(tab_id)
        if not tab_info or tab_info.tab_type != "editor":
            return
        
        editor = tab_info.widget
        path, _ = QFileDialog.getSaveFileName(
            self, "Save", tab_info.title,
            "Text files (*.txt);;Markdown (*.md);;All files (*.*)"
        )
        if not path:
            return
        
        try:
            Path(path).write_text(editor.get_text(), encoding="utf-8")
            self.status_msg.setText(f"Saved: {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save file: {e}")
    
    def _add_tab_from_combo(self):
        """Add tab of selected type."""
        tab_type = self.new_tab_combo.currentText()
        self._create_tab(tab_type)
    
    def _create_tab(self, tab_type: str, position: DockPosition = DockPosition.CENTER):
        """Create and add a tab."""
        widget = TabTypeRegistry.create(tab_type)
        if widget:
            title = tab_type.title()
            self.dock_layout.add_tab(widget, title, position, tab_type)
    
    def _connect_llm(self, url: str = None):
        """Connect to LLM."""
        self.llm_channel.connect_llm(url)
    
    def _disconnect_llm(self):
        """Disconnect from LLM."""
        self.llm_channel.disconnect_llm()
    
    def _auto_connect_llm(self):
        """Auto-connect to LLM on startup."""
        self.llm_channel.connect_llm()
    
    def _toggle_theme(self):
        """Toggle light/dark theme."""
        if self.settings.theme.name == "dark":
            self.settings.theme = Theme.light()
        else:
            self.settings.theme = Theme.dark()
        self.settings.save()
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply current theme."""
        t = self.settings.theme
        
        if t.name == "dark":
            base_bg = "#1e1e1e"
            base_fg = "#d4d4d4"
            panel_bg = "#0f1115"
            accent = t.accent
        else:
            base_bg = "#f7f7f8"
            base_fg = "#1b1b1f"
            panel_bg = "#ffffff"
            accent = t.accent
        
        self.setStyleSheet(f"""
            QMainWindow {{ background: {base_bg}; color: {base_fg}; }}
            QToolBar {{ background: {panel_bg}; border: none; spacing: 5px; padding: 5px; }}
            QToolBar QLabel {{ color: {base_fg}; }}
            QPushButton {{ 
                background: {panel_bg}; 
                color: {base_fg}; 
                border: 1px solid #3e3e42; 
                padding: 5px 10px; 
                border-radius: 4px; 
            }}
            QPushButton:hover {{ border-color: {accent}; }}
            QComboBox {{ 
                background: {panel_bg}; 
                color: {base_fg}; 
                border: 1px solid #3e3e42; 
                padding: 5px; 
            }}
            QStatusBar {{ background: {accent}; color: white; }}
        """)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AQUA v2")
    app.setApplicationVersion("2.0")
    
    window = MainWindow()
    window.show()
    
    print("AQUA v2 started!")
    print("- Drag tabs between areas to rearrange")
    print("- Type /help in prompt for commands")
    print("- Type /connect to connect to LLM")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
