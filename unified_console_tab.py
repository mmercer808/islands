#!/usr/bin/env python3
"""
Unified Console Tab - Single widget handling both commands and LLM queries
Key Features:
- Single QTextEdit widget with cursor position tracking
- Smart routing: LLM queries vs system commands
- Text formatting for input/output separation
- Command history with up/down navigation
- Special commands (/help, /connect, /clear, etc.)
- Global settings integration
"""

import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from PySide6.QtWidgets import (
    QTextEdit, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QPushButton, QTabWidget, QApplication, QMainWindow, QWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import (
    QFont, QTextCharFormat, QColor, QTextCursor, QKeySequence
)


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

class GlobalSettings:
    """Global application settings"""
    
    def __init__(self):
        self.llm_settings = {
            'default_url': 'http://localhost:11434',
            'default_model': 'phi',
            'streaming': True,
            'timeout': 30,
            'auto_connect': False,
            'api_key': '',
            'provider': 'local'  # 'local', 'openai', 'anthropic', 'huggingface'
        }
        
        self.console_settings = {
            'history_size': 100,
            'command_prefixes': ['$', 'cmd:', 'shell:', '!'],
            'special_prefixes': ['/'],
            'default_to_llm': True,  # If LLM connected, default everything to LLM
            'show_timestamps': False,
            'auto_scroll': True
        }
        
        self.ui_settings = {
            'theme': 'dark',
            'font_family': 'Consolas',
            'font_size': 12,
            'prompt_symbol': '> ',
            'llm_symbol': '🤖 ',
            'error_symbol': '❌ ',
            'success_symbol': '✅ '
        }
    
    def save_to_file(self, file_path: str = "console_settings.json"):
        """Save settings to file"""
        all_settings = {
            'llm': self.llm_settings,
            'console': self.console_settings,
            'ui': self.ui_settings
        }
        
        with open(file_path, 'w') as f:
            json.dump(all_settings, f, indent=2)
    
    def load_from_file(self, file_path: str = "console_settings.json"):
        """Load settings from file"""
        try:
            with open(file_path, 'r') as f:
                all_settings = json.load(f)
            
            self.llm_settings.update(all_settings.get('llm', {}))
            self.console_settings.update(all_settings.get('console', {}))
            self.ui_settings.update(all_settings.get('ui', {}))
        except FileNotFoundError:
            # Use defaults
            pass


# =============================================================================
# SIGNAL BRIDGE (Enhanced)
# =============================================================================

class EnhancedSignalBridge(QObject):
    """Enhanced bridge between your signal system and PySide"""
    
    # PySide signals for immediate UI updates
    console_output = Signal(str, str)  # message, message_type
    llm_response_chunk = Signal(str)   # chunk_content
    llm_response_complete = Signal(str)  # full_response
    command_executed = Signal(str, dict)  # command, result
    connection_status_changed = Signal(bool, str)  # connected, message
    
    def __init__(self, your_signal_emitter=None):
        super().__init__()
        self.your_signal_emitter = your_signal_emitter
        
    def emit_to_both_systems(self, signal_type: str, source_id: str, data: Dict[str, Any]):
        """Emit to both PySide and your signal system"""
        
        # 1. Immediate PySide signal for UI responsiveness
        if signal_type == "llm_chunk":
            self.llm_response_chunk.emit(data.get('content', ''))
        elif signal_type == "llm_complete":
            self.llm_response_complete.emit(data.get('response', ''))
        elif signal_type == "command_result":
            self.command_executed.emit(data.get('command', ''), data.get('result', {}))
        elif signal_type == "connection_status":
            self.connection_status_changed.emit(data.get('connected', False), data.get('message', ''))
        
        # 2. Your signal system for business logic
        if self.your_signal_emitter:
            self.your_signal_emitter.emit_signal(signal_type, source_id, data)
        
        # Always emit general console output
        self.console_output.emit(data.get('message', ''), data.get('type', 'output'))


# =============================================================================
# UNIFIED CONSOLE TAB
# =============================================================================

class UnifiedConsoleTab(QTextEdit):
    """Single widget console handling both commands and LLM queries"""
    
    # Signals for external integration
    command_executed = Signal(str, dict)
    llm_query_sent = Signal(str)
    
    def __init__(self, signal_bridge: EnhancedSignalBridge, settings: GlobalSettings):
        super().__init__()
        self.signal_bridge = signal_bridge
        self.settings = settings
        
        # Console state
        self.prompt_text = self.settings.ui_settings['prompt_symbol']
        self.input_start_pos = 0  # Where user input begins
        self.command_history = []
        self.history_index = -1
        self.current_directory = Path.cwd()
        
        # LLM connection state
        self.llm_connected = False
        self.llm_url = self.settings.llm_settings['default_url']
        self.llm_model = self.settings.llm_settings['default_model']
        self.waiting_for_llm = False
        
        # Setup
        self.setup_console()
        self.setup_signal_connections()
        self.show_welcome_message()
        self.show_prompt()
    
    def setup_console(self):
        """Setup console appearance and behavior"""
        # Terminal-like font
        font = QFont(self.settings.ui_settings['font_family'], 
                    self.settings.ui_settings['font_size'])
        font.setFixedPitch(True)
        self.setFont(font)
        
        # Dark terminal theme
        if self.settings.ui_settings['theme'] == 'dark':
            self.setStyleSheet("""
                QTextEdit {
                    background-color: #0c0c0c;
                    color: #cccccc;
                    border: none;
                    selection-background-color: #264f78;
                    selection-color: #ffffff;
                }
            """)
        
        # Configure behavior
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setAcceptRichText(True)  # For formatting
    
    def setup_signal_connections(self):
        """Setup signal connections"""
        # Connect to signal bridge
        self.signal_bridge.llm_response_chunk.connect(self.on_llm_chunk_received)
        self.signal_bridge.llm_response_complete.connect(self.on_llm_response_complete)
        self.signal_bridge.connection_status_changed.connect(self.on_connection_status_changed)
    
    def show_welcome_message(self):
        """Show initial welcome message"""
        welcome = f"""Unified Console - Commands & LLM
Working Directory: {self.current_directory}
LLM Model: {self.llm_model} | Connected: {self.llm_connected}

Commands:
  /help     - Show help
  /connect  - Connect to LLM
  /clear    - Clear console
  /cd <dir> - Change directory
  /settings - Open settings

Usage:
  Regular text → LLM query (if connected)
  $ command  → System command
  /command   → Special command

"""
        self.append_formatted_text(welcome, "info")
    
    def show_prompt(self):
        """Show prompt and position cursor for input"""
        # Add working directory info
        if self.settings.console_settings.get('show_pwd', False):
            pwd_text = f"[{self.current_directory.name}] "
            self.append_formatted_text(pwd_text, "path")
        
        self.append_formatted_text(self.prompt_text, "prompt")
        
        # Mark where user input starts
        cursor = self.textCursor()
        self.input_start_pos = cursor.position()
    
    def keyPressEvent(self, event):
        """Handle keyboard input with cursor restrictions"""
        cursor = self.textCursor()
        current_pos = cursor.position()
        
        # Prevent editing before the prompt
        if current_pos < self.input_start_pos and event.key() not in [Qt.Key.Key_End, Qt.Key.Key_Down]:
            cursor.setPosition(self.input_start_pos)
            self.setTextCursor(cursor)
            if event.key() not in [Qt.Key.Key_Up, Qt.Key.Key_Down]:
                return
        
        if event.key() == Qt.Key.Key_Return and not event.modifiers():
            self.execute_current_input()
            
        elif event.key() == Qt.Key.Key_Up:
            self.navigate_history(-1)
            
        elif event.key() == Qt.Key.Key_Down:
            self.navigate_history(1)
            
        elif event.key() == Qt.Key.Key_Backspace:
            if current_pos <= self.input_start_pos:
                return
            super().keyPressEvent(event)
            
        elif event.key() == Qt.Key.Key_Left:
            if current_pos <= self.input_start_pos:
                return
            super().keyPressEvent(event)
            
        elif event.key() == Qt.Key.Key_Home:
            cursor.setPosition(self.input_start_pos)
            self.setTextCursor(cursor)
            
        elif event.key() == Qt.Key.Key_Tab:
            # Tab completion (future feature)
            event.accept()
            
        else:
            # Normal text input
            super().keyPressEvent(event)
    
    def execute_current_input(self):
        """Execute current input with smart routing"""
        input_text = self.get_current_input().strip()
        
        if not input_text:
            self.move_to_new_line()
            self.show_prompt()
            return
        
        # Add to history
        if input_text and (not self.command_history or self.command_history[-1] != input_text):
            self.command_history.append(input_text)
            if len(self.command_history) > self.settings.console_settings['history_size']:
                self.command_history.pop(0)
        
        self.history_index = len(self.command_history)
        
        # Move to new line
        self.move_to_new_line()
        
        # Route the input
        if self.is_special_command(input_text):
            self.execute_special_command(input_text)
        elif self.is_explicit_system_command(input_text):
            self.execute_system_command(input_text)
        elif self.llm_connected and self.settings.console_settings['default_to_llm']:
            self.execute_llm_query(input_text)
        else:
            # No LLM connection, try as system command
            self.execute_system_command(input_text)
        
        # Show new prompt (unless waiting for LLM)
        if not self.waiting_for_llm:
            self.show_prompt()
    
    def get_current_input(self) -> str:
        """Get current input text"""
        cursor = self.textCursor()
        cursor.setPosition(self.input_start_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        return cursor.selectedText()
    
    def move_to_new_line(self):
        """Move cursor to new line"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n")
        self.setTextCursor(cursor)
    
    def clear_current_input(self):
        """Clear current input line"""
        cursor = self.textCursor()
        cursor.setPosition(self.input_start_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
    
    def navigate_history(self, direction: int):
        """Navigate command history"""
        if not self.command_history:
            return
        
        self.clear_current_input()
        
        self.history_index = max(0, min(len(self.command_history), self.history_index + direction))
        
        if self.history_index < len(self.command_history):
            history_text = self.command_history[self.history_index]
            cursor = self.textCursor()
            cursor.setPosition(self.input_start_pos)
            cursor.insertText(history_text)
    
    def is_special_command(self, text: str) -> bool:
        """Check if text is a special command (/help, /clear, etc.)"""
        return any(text.startswith(prefix) for prefix in self.settings.console_settings['special_prefixes'])
    
    def is_explicit_system_command(self, text: str) -> bool:
        """Check if text is explicitly marked as system command"""
        prefixes = self.settings.console_settings['command_prefixes']
        return any(text.startswith(prefix) for prefix in prefixes)
    
    def execute_special_command(self, command: str):
        """Execute special commands (/help, /clear, etc.)"""
        parts = command[1:].split()  # Remove leading /
        cmd = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            self.show_help()
        elif cmd == "clear":
            self.clear_console()
        elif cmd == "connect":
            url = args[0] if args else self.llm_url
            self.connect_to_llm(url)
        elif cmd == "disconnect":
            self.disconnect_from_llm()
        elif cmd == "model":
            model = args[0] if args else "phi2"
            self.set_model(model)
        elif cmd == "status":
            self.show_status()
        elif cmd == "history":
            self.show_history()
        elif cmd == "cd":
            directory = args[0] if args else str(Path.home())
            self.change_directory(directory)
        elif cmd == "pwd":
            self.append_formatted_text(f"{self.current_directory}\n", "output")
        elif cmd == "settings":
            self.append_formatted_text("Settings dialog would open here\n", "info")
        else:
            self.append_formatted_text(f"Unknown command: {command}\n", "error")
    
    def execute_system_command(self, command: str):
        """Execute system command"""
        # Remove explicit prefixes
        for prefix in self.settings.console_settings['command_prefixes']:
            if command.startswith(prefix):
                command = command[len(prefix):].strip()
                break
        
        if not command:
            return
        
        try:
            self.append_formatted_text(f"$ {command}\n", "command")
            
            # Execute in current directory
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.current_directory)
            )
            
            if result.stdout:
                self.append_formatted_text(result.stdout, "output")
            if result.stderr:
                self.append_formatted_text(result.stderr, "error")
            
            # Emit signal for logging/monitoring
            self.command_executed.emit(command, {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            })
            
        except subprocess.TimeoutExpired:
            self.append_formatted_text("Command timed out\n", "error")
        except Exception as e:
            self.append_formatted_text(f"Error executing command: {str(e)}\n", "error")
    
    def execute_llm_query(self, query: str):
        """Execute LLM query"""
        if not self.llm_connected:
            self.append_formatted_text("❌ Not connected to LLM. Use /connect <url>\n", "error")
            return
        
        self.append_formatted_text(f"🤖 Thinking...\n", "status")
        self.waiting_for_llm = True
        
        # Emit to signal system for LLM processing
        self.signal_bridge.emit_to_both_systems(
            "llm_query_start",
            "unified_console",
            {
                "query": query,
                "model": self.llm_model,
                "url": self.llm_url,
                "stream": self.settings.llm_settings['streaming']
            }
        )
        
        # Emit local signal
        self.llm_query_sent.emit(query)
    
    def connect_to_llm(self, url: str):
        """Connect to LLM service"""
        self.llm_url = url
        self.append_formatted_text(f"Connecting to {url}...\n", "info")
        
        # Emit connection request
        self.signal_bridge.emit_to_both_systems(
            "llm_connect_request",
            "unified_console",
            {
                "url": url,
                "model": self.llm_model,
                "api_key": self.settings.llm_settings.get('api_key', ''),
                "provider": self.settings.llm_settings.get('provider', 'local')
            }
        )
    
    def disconnect_from_llm(self):
        """Disconnect from LLM"""
        self.llm_connected = False
        self.append_formatted_text("Disconnected from LLM\n", "info")
        
        self.signal_bridge.emit_to_both_systems(
            "llm_disconnect",
            "unified_console",
            {"url": self.llm_url}
        )
    
    def set_model(self, model: str):
        """Set LLM model"""
        self.llm_model = model
        self.append_formatted_text(f"Model set to: {model}\n", "success")
        
        self.signal_bridge.emit_to_both_systems(
            "llm_model_change",
            "unified_console",
            {"model": model, "url": self.llm_url}
        )
    
    def change_directory(self, path: str):
        """Change working directory"""
        try:
            new_dir = Path(path).resolve()
            if new_dir.exists() and new_dir.is_dir():
                self.current_directory = new_dir
                self.append_formatted_text(f"Changed to: {new_dir}\n", "success")
            else:
                self.append_formatted_text(f"Directory not found: {path}\n", "error")
        except Exception as e:
            self.append_formatted_text(f"Error changing directory: {str(e)}\n", "error")
    
    def show_help(self):
        """Show help text"""
        help_text = """
Unified Console Help:

Special Commands:
  /help          - Show this help
  /clear         - Clear console
  /connect <url> - Connect to LLM (e.g., /connect http://localhost:8080)
  /disconnect    - Disconnect from LLM
  /model <name>  - Set LLM model (e.g., /model phi2)
  /status        - Show connection and system status
  /history       - Show command history
  /cd <dir>      - Change directory
  /pwd           - Show current directory
  /settings      - Open settings dialog

Input Routing:
  Regular text   → LLM query (if connected)
  $ command      → Force system command
  ! command      → Force system command
  cmd: command   → Force system command
  /command       → Special console command

Navigation:
  ↑/↓ arrows     - Navigate command history
  Home           - Go to start of input line
  Ctrl+C         - Cancel current operation (future)

Examples:
  hello world                    → LLM query
  $ ls -la                      → System command
  /connect http://localhost:8080 → Connect to local LLM
  what files are in this folder? → LLM query

"""
        self.append_formatted_text(help_text, "help")
    
    def show_status(self):
        """Show system status"""
        status_text = f"""
System Status:
  Working Directory: {self.current_directory}
  LLM Connected:     {self.llm_connected}
  LLM URL:          {self.llm_url}
  LLM Model:        {self.llm_model}
  Command History:  {len(self.command_history)} entries
  Waiting for LLM:  {self.waiting_for_llm}

Settings:
  Default to LLM:   {self.settings.console_settings['default_to_llm']}
  History Size:     {self.settings.console_settings['history_size']}
  Font:            {self.settings.ui_settings['font_family']} {self.settings.ui_settings['font_size']}

"""
        self.append_formatted_text(status_text, "info")
    
    def show_history(self):
        """Show command history"""
        if not self.command_history:
            self.append_formatted_text("No command history\n", "info")
            return
        
        self.append_formatted_text("Command History:\n", "info")
        for i, cmd in enumerate(self.command_history[-10:], 1):  # Last 10
            self.append_formatted_text(f"  {i:2d}. {cmd}\n", "history")
    
    def clear_console(self):
        """Clear console output"""
        self.clear()
        self.show_prompt()
    
    def append_formatted_text(self, text: str, text_type: str = "output"):
        """Append formatted text to console"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Create format based on type
        format = QTextCharFormat()
        
        color_map = {
            "prompt": "#00ff00",      # Green
            "command": "#ffff00",     # Yellow
            "output": "#cccccc",      # Light gray
            "error": "#ff4444",       # Red
            "info": "#888888",        # Dark gray
            "success": "#44ff44",     # Bright green
            "status": "#ffaa00",      # Orange
            "llm": "#44aaff",         # Blue
            "help": "#cc88ff",        # Purple
            "history": "#88ccff",     # Light blue
            "path": "#ff88cc"         # Pink
        }
        
        if text_type in color_map:
            format.setForeground(QColor(color_map[text_type]))
        
        if text_type == "prompt":
            format.setFontWeight(QFont.Weight.Bold)
        elif text_type == "error":
            format.setFontWeight(QFont.Weight.Bold)
        
        cursor.setCharFormat(format)
        cursor.insertText(text)
        
        # Auto-scroll to bottom
        if self.settings.console_settings.get('auto_scroll', True):
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    # Signal handlers
    def on_llm_chunk_received(self, chunk: str):
        """Handle streaming LLM response chunk"""
        self.append_formatted_text(chunk, "llm")
    
    def on_llm_response_complete(self, response: str):
        """Handle complete LLM response"""
        self.waiting_for_llm = False
        self.append_formatted_text("\n", "output")  # Add newline
        self.show_prompt()
    
    def on_connection_status_changed(self, connected: bool, message: str):
        """Handle LLM connection status change"""
        self.llm_connected = connected
        status = "✅ Connected" if connected else "❌ Disconnected"
        self.append_formatted_text(f"{status}: {message}\n", "success" if connected else "error")


# =============================================================================
# SETTINGS DIALOG
# =============================================================================

class ConsoleSettingsDialog(QDialog):
    """Global settings dialog for console configuration"""
    
    def __init__(self, settings: GlobalSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Console Settings")
        self.setModal(True)
        self.resize(500, 600)
        self.setup_ui()
        self.load_current_settings()
    
    def setup_ui(self):
        """Setup settings dialog UI"""
        layout = QVBoxLayout(self)
        
        # LLM Connection Settings
        llm_group = QGroupBox("LLM Connection")
        llm_layout = QGridLayout(llm_group)
        
        llm_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["local", "openai", "anthropic", "huggingface"])
        llm_layout.addWidget(self.provider_combo, 0, 1)
        
        llm_layout.addWidget(QLabel("URL:"), 1, 0)
        self.url_edit = QLineEdit()
        llm_layout.addWidget(self.url_edit, 1, 1)
        
        llm_layout.addWidget(QLabel("Model:"), 2, 0)
        self.model_edit = QLineEdit()
        llm_layout.addWidget(self.model_edit, 2, 1)
        
        llm_layout.addWidget(QLabel("API Key:"), 3, 0)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        llm_layout.addWidget(self.api_key_edit, 3, 1)
        
        self.streaming_cb = QCheckBox("Enable streaming responses")
        llm_layout.addWidget(self.streaming_cb, 4, 0, 1, 2)
        
        self.auto_connect_cb = QCheckBox("Auto-connect on startup")
        llm_layout.addWidget(self.auto_connect_cb, 5, 0, 1, 2)
        
        layout.addWidget(llm_group)
        
        # Console Behavior Settings
        console_group = QGroupBox("Console Behavior")
        console_layout = QGridLayout(console_group)
        
        console_layout.addWidget(QLabel("History Size:"), 0, 0)
        self.history_size_spin = QSpinBox()
        self.history_size_spin.setRange(10, 1000)
        console_layout.addWidget(self.history_size_spin, 0, 1)
        
        self.default_llm_cb = QCheckBox("Default input to LLM (when connected)")
        console_layout.addWidget(self.default_llm_cb, 1, 0, 1, 2)
        
        self.auto_scroll_cb = QCheckBox("Auto-scroll to bottom")
        console_layout.addWidget(self.auto_scroll_cb, 2, 0, 1, 2)
        
        self.show_timestamps_cb = QCheckBox("Show timestamps")
        console_layout.addWidget(self.show_timestamps_cb, 3, 0, 1, 2)
        
        layout.addWidget(console_group)
        
        # UI Settings
        ui_group = QGroupBox("Appearance")
        ui_layout = QGridLayout(ui_group)
        
        ui_layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        ui_layout.addWidget(self.theme_combo, 0, 1)
        
        ui_layout.addWidget(QLabel("Font Family:"), 1, 0)
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(["Consolas", "Courier New", "Monaco", "Ubuntu Mono"])
        ui_layout.addWidget(self.font_family_combo, 1, 1)
        
        ui_layout.addWidget(QLabel("Font Size:"), 2, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        ui_layout.addWidget(self.font_size_spin, 2, 1)
        
        layout.addWidget(ui_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def load_current_settings(self):
        """Load current settings into dialog"""
        # LLM settings
        self.provider_combo.setCurrentText(self.settings.llm_settings.get('provider', 'local'))
        self.url_edit.setText(self.settings.llm_settings.get('default_url', 'http://localhost:8080'))
        self.model_edit.setText(self.settings.llm_settings.get('default_model', 'phi2'))
        self.api_key_edit.setText(self.settings.llm_settings.get('api_key', ''))
        self.streaming_cb.setChecked(self.settings.llm_settings.get('streaming', True))
        self.auto_connect_cb.setChecked(self.settings.llm_settings.get('auto_connect', False))
        
        # Console settings
        self.history_size_spin.setValue(self.settings.console_settings.get('history_size', 100))
        self.default_llm_cb.setChecked(self.settings.console_settings.get('default_to_llm', True))
        self.auto_scroll_cb.setChecked(self.settings.console_settings.get('auto_scroll', True))
        self.show_timestamps_cb.setChecked(self.settings.console_settings.get('show_timestamps', False))
        
        # UI settings
        self.theme_combo.setCurrentText(self.settings.ui_settings.get('theme', 'dark'))
        self.font_family_combo.setCurrentText(self.settings.ui_settings.get('font_family', 'Consolas'))
        self.font_size_spin.setValue(self.settings.ui_settings.get('font_size', 12))
    
    def save_settings(self):
        """Save settings and close dialog"""
        # Update LLM settings
        self.settings.llm_settings.update({
            'provider': self.provider_combo.currentText(),
            'default_url': self.url_edit.text(),
            'default_model': self.model_edit.text(),
            'api_key': self.api_key_edit.text(),
            'streaming': self.streaming_cb.isChecked(),
            'auto_connect': self.auto_connect_cb.isChecked()
        })
        
        # Update console settings
        self.settings.console_settings.update({
            'history_size': self.history_size_spin.value(),
            'default_to_llm': self.default_llm_cb.isChecked(),
            'auto_scroll': self.auto_scroll_cb.isChecked(),
            'show_timestamps': self.show_timestamps_cb.isChecked()
        })
        
        # Update UI settings
        self.settings.ui_settings.update({
            'theme': self.theme_combo.currentText(),
            'font_family': self.font_family_combo.currentText(),
            'font_size': self.font_size_spin.value()
        })
        
        # Save to file
        self.settings.save_to_file()
        
        self.accept()


# =============================================================================
# ENHANCED TAB WIDGET FOR MIXED TAB TYPES
# =============================================================================

class SmartTabWidget(QTabWidget):
    """Enhanced tab widget that handles different tab types"""
    
    def __init__(self, settings: GlobalSettings):
        super().__init__()
        self.settings = settings
        self.signal_bridge = EnhancedSignalBridge()
        self.tab_types = {}  # Track tab types by index
        self.tab_counters = {
            'editor': 0,
            'console': 0,
            'notebook': 0
        }
        
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)
    
    def add_text_editor_tab(self, title: str = None, content: str = ""):
        """Add a text editor tab"""
        from simple_word_processor import SimpleTextEditor  # Import your existing editor
        
        self.tab_counters['editor'] += 1
        title = title or f"Document {self.tab_counters['editor']}"
        
        # Create editor (you'll need to adapt this to your existing SimpleTextEditor)
        editor = SimpleTextEditor(self.signal_bridge, f"editor_{self.tab_counters['editor']}")
        if content:
            editor.setPlainText(content)
        
        index = self.addTab(editor, title)
        self.tab_types[index] = 'editor'
        self.setCurrentIndex(index)
        
        return editor
    
    def add_console_tab(self, title: str = None):
        """Add a unified console tab"""
        self.tab_counters['console'] += 1
        title = title or f"Console {self.tab_counters['console']}"
        
        console = UnifiedConsoleTab(self.signal_bridge, self.settings)
        
        index = self.addTab(console, title)
        self.tab_types[index] = 'console'
        self.setCurrentIndex(index)
        
        return console
    
    def add_notebook_tab(self, title: str = None):
        """Add a notebook-style tab (future implementation)"""
        self.tab_counters['notebook'] += 1
        title = title or f"Notebook {self.tab_counters['notebook']}"
        
        # Placeholder for future notebook implementation
        from PySide6.QtWidgets import QLabel
        placeholder = QLabel("Notebook tab - Coming soon!")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        index = self.addTab(placeholder, title)
        self.tab_types[index] = 'notebook'
        self.setCurrentIndex(index)
        
        return placeholder
    
    def close_tab(self, index: int):
        """Close a tab with type-specific cleanup"""
        if index < 0 or index >= self.count():
            return
        
        tab_type = self.tab_types.get(index, 'unknown')
        widget = self.widget(index)
        
        # Type-specific cleanup
        if tab_type == 'editor':
            # Check for unsaved changes
            if hasattr(widget, 'document') and widget.document().isModified():
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    "This document has unsaved changes. Close anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
        
        elif tab_type == 'console':
            # Disconnect from LLM if connected
            if hasattr(widget, 'llm_connected') and widget.llm_connected:
                widget.disconnect_from_llm()
        
        # Remove tab
        self.removeTab(index)
        
        # Update tab_types mapping (shift indices)
        new_tab_types = {}
        for tab_index, tab_type_val in self.tab_types.items():
            if tab_index < index:
                new_tab_types[tab_index] = tab_type_val
            elif tab_index > index:
                new_tab_types[tab_index - 1] = tab_type_val
        self.tab_types = new_tab_types
    
    def get_current_tab_type(self) -> str:
        """Get the type of the current tab"""
        return self.tab_types.get(self.currentIndex(), 'unknown')
    
    def get_current_tab_widget(self):
        """Get the current tab widget"""
        return self.currentWidget()


# =============================================================================
# INTEGRATION WITH MAIN WORD PROCESSOR
# =============================================================================

class EnhancedWordProcessor(QMainWindow):
    """Enhanced word processor with unified console tabs"""
    
    def __init__(self):
        super().__init__()
        
        # Load global settings
        self.settings = GlobalSettings()
        self.settings.load_from_file()
        
        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        
        # Create initial tabs
        self.create_initial_tabs()
    
    def setup_ui(self):
        """Setup main UI"""
        self.setWindowTitle("Enhanced Word Processor with Unified Console")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with smart tab system
        self.tab_widget = SmartTabWidget(self.settings)
        self.setCentralWidget(self.tab_widget)
        
        # Apply theme
        self.apply_theme()
    
    def setup_menu_bar(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_editor_action = file_menu.addAction("New Text Document")
        new_editor_action.setShortcut(QKeySequence.StandardKey.New)
        new_editor_action.triggered.connect(lambda: self.tab_widget.add_text_editor_tab())
        
        new_console_action = file_menu.addAction("New Console")
        new_console_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        new_console_action.triggered.connect(lambda: self.tab_widget.add_console_tab())
        
        new_notebook_action = file_menu.addAction("New Notebook")
        new_notebook_action.setShortcut(QKeySequence("Ctrl+Shift+N"))
        new_notebook_action.triggered.connect(lambda: self.tab_widget.add_notebook_tab())
        
        file_menu.addSeparator()
        
        settings_action = file_menu.addAction("Settings...")
        settings_action.triggered.connect(self.open_settings)
        
        # Console menu
        console_menu = menubar.addMenu("Console")
        
        connect_action = console_menu.addAction("Connect to LLM...")
        connect_action.triggered.connect(self.quick_connect_llm)
        
        clear_action = console_menu.addAction("Clear Current Console")
        clear_action.setShortcut(QKeySequence("Ctrl+L"))
        clear_action.triggered.connect(self.clear_current_console)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        console_help_action = help_menu.addAction("Console Help")
        console_help_action.triggered.connect(self.show_console_help)
    
    def setup_toolbar(self):
        """Setup toolbar"""
        toolbar = self.addToolBar("Main")
        
        # Tab creation buttons
        new_editor_btn = QPushButton("📄 Text")
        new_editor_btn.clicked.connect(lambda: self.tab_widget.add_text_editor_tab())
        toolbar.addWidget(new_editor_btn)
        
        new_console_btn = QPushButton("💻 Console")
        new_console_btn.clicked.connect(lambda: self.tab_widget.add_console_tab())
        toolbar.addWidget(new_console_btn)
        
        toolbar.addSeparator()
        
        # Console controls
        connect_btn = QPushButton("🔗 Connect LLM")
        connect_btn.clicked.connect(self.quick_connect_llm)
        toolbar.addWidget(connect_btn)
        
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self.open_settings)
        toolbar.addWidget(settings_btn)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Tab type indicator
        self.tab_type_label = QLabel("No tabs")
        self.status_bar.addPermanentWidget(self.tab_type_label)
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.update_status_bar)
    
    def apply_theme(self):
        """Apply global theme"""
        if self.settings.ui_settings['theme'] == 'dark':
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                }
                QTabWidget::pane {
                    border: 1px solid #3e3e42;
                    background-color: #1e1e1e;
                }
                QTabBar::tab {
                    background-color: #2d2d30;
                    color: #d4d4d4;
                    padding: 8px 16px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                QTabBar::tab:selected {
                    background-color: #007acc;
                    color: white;
                }
                QPushButton {
                    background-color: #3c3c3c;
                    border: 1px solid #565656;
                    color: #d4d4d4;
                    padding: 6px 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #484848;
                }
                QStatusBar {
                    background-color: #007acc;
                    color: white;
                }
            """)
    
    def create_initial_tabs(self):
        """Create initial tabs"""
        # Create a welcome text editor
        editor = self.tab_widget.add_text_editor_tab("Welcome")
        editor.setPlainText("""Welcome to Enhanced Word Processor!

This application supports multiple tab types:

1. Text Editor Tabs - Traditional text editing
2. Console Tabs - Command execution and LLM interaction
3. Notebook Tabs - Jupyter-style cells (coming soon)

Try creating a new console tab to get started with LLM interactions!

Console Commands:
- /help - Show console help
- /connect <url> - Connect to LLM service
- /clear - Clear console
- Regular text - Send to LLM (when connected)
- $ command - Execute system command

""")
        
        # Create a console tab
        self.tab_widget.add_console_tab("Main Console")
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = ConsoleSettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Settings were saved, apply any immediate changes
            self.apply_theme()
            
            # Notify all console tabs of settings change
            for i in range(self.tab_widget.count()):
                widget = self.tab_widget.widget(i)
                if isinstance(widget, UnifiedConsoleTab):
                    # You could add a method to refresh settings
                    widget.setup_console()  # Reapply console settings
    
    def quick_connect_llm(self):
        """Quick connect to LLM using default settings"""
        current_widget = self.tab_widget.get_current_tab_widget()
        
        if isinstance(current_widget, UnifiedConsoleTab):
            url = self.settings.llm_settings['default_url']
            current_widget.connect_to_llm(url)
        else:
            # Create a new console tab and connect
            console = self.tab_widget.add_console_tab("LLM Console")
            url = self.settings.llm_settings['default_url']
            console.connect_to_llm(url)
    
    def clear_current_console(self):
        """Clear current console if it's a console tab"""
        current_widget = self.tab_widget.get_current_tab_widget()
        
        if isinstance(current_widget, UnifiedConsoleTab):
            current_widget.clear_console()
    
    def show_console_help(self):
        """Show console help in a new tab or current console"""
        current_widget = self.tab_widget.get_current_tab_widget()
        
        if isinstance(current_widget, UnifiedConsoleTab):
            current_widget.show_help()
        else:
            # Create new console and show help
            console = self.tab_widget.add_console_tab("Console Help")
            console.show_help()
    
    def update_status_bar(self, index: int):
        """Update status bar based on current tab"""
        tab_type = self.tab_widget.get_current_tab_type()
        widget = self.tab_widget.get_current_tab_widget()
        
        if tab_type == 'editor':
            self.tab_type_label.setText("Text Editor")
        elif tab_type == 'console':
            llm_status = "Connected" if hasattr(widget, 'llm_connected') and widget.llm_connected else "Disconnected"
            self.tab_type_label.setText(f"Console - LLM: {llm_status}")
        elif tab_type == 'notebook':
            self.tab_type_label.setText("Notebook")
        else:
            self.tab_type_label.setText("Unknown")


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Enhanced Word Processor")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("Enhanced Systems")
    
    # Create and show main window
    window = EnhancedWordProcessor()
    window.show()
    
    print("🚀 Enhanced Word Processor with Unified Console started!")
    print("💻 Console tabs support both commands and LLM queries")
    print("⚙️ Use File → Settings to configure LLM connections")
    print("📚 Try creating different tab types from the toolbar")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())