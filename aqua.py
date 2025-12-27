#!/usr/bin/env python3
"""
PROJECT AQUA
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
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field

from PySide6.QtWidgets import (
    QTextEdit, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QPushButton, QTabWidget, QApplication, QMainWindow, QWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import (
    QFont, QTextCharFormat, QColor, QTextCursor, QKeySequence
)
from PySide6.QtCore import Slot

from matts.signal_system import SignalLine, SignalPayload, SignalType, SignalPriority, Observer, CallbackObserver, CircuitBreaker, ObserverWorkerPool, PriorityDispatcher


import logging
# Configure logging
logger = logging.getLogger(__name__)

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
            'auto_connect': True,
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
# LLM-SPECIFIC SIGNAL TYPES
# =============================================================================

class LLMSignalType(Enum):
    """LLM-specific signal types extending the base signal system."""
    
    # Connection events
    LLM_CONNECTING = "llm_connecting"
    LLM_CONNECTED = "llm_connected"
    LLM_DISCONNECTED = "llm_disconnected"
    LLM_CONNECTION_FAILED = "llm_connection_failed"
    LLM_RECONNECTING = "llm_reconnecting"
    
    # Query lifecycle
    LLM_QUERY_QUEUED = "llm_query_queued"
    LLM_QUERY_STARTED = "llm_query_started"
    LLM_QUERY_PROGRESS = "llm_query_progress"
    LLM_QUERY_COMPLETED = "llm_query_completed"
    LLM_QUERY_FAILED = "llm_query_failed"
    LLM_QUERY_CANCELLED = "llm_query_cancelled"
    LLM_QUERY_TIMEOUT = "llm_query_timeout"
    
    # Streaming events
    LLM_STREAM_STARTED = "llm_stream_started"
    LLM_STREAM_CHUNK = "llm_stream_chunk"
    LLM_STREAM_ENDED = "llm_stream_ended"
    LLM_STREAM_ERROR = "llm_stream_error"
    
    # Model management
    LLM_MODEL_LOADING = "llm_model_loading"
    LLM_MODEL_LOADED = "llm_model_loaded"
    LLM_MODEL_UNLOADED = "llm_model_unloaded"
    LLM_MODEL_ERROR = "llm_model_error"
    LLM_MODELS_LISTED = "llm_models_listed"
    
    # Provider-specific
    LLM_PROVIDER_STATUS = "llm_provider_status"
    LLM_PROVIDER_ERROR = "llm_provider_error"
    LLM_PROVIDER_METRICS = "llm_provider_metrics"
    
    # System events
    LLM_SYSTEM_READY = "llm_system_ready"
    LLM_SYSTEM_SHUTDOWN = "llm_system_shutdown"


# =============================================================================
# SIMPLIFIED SIGNAL SYSTEM (NO WORKERS)
# =============================================================================

class SimpleSignalType:
    """Simple signal types for the word processor"""
    
    # Document operations
    DOCUMENT_OPENED = "document_opened"
    DOCUMENT_SAVED = "document_saved"
    DOCUMENT_MODIFIED = "document_modified"
    DOCUMENT_CLOSED = "document_closed"
    
    # Text operations
    TEXT_INSERTED = "text_inserted"
    TEXT_DELETED = "text_deleted"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    
    # Plugin operations
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_EXECUTED = "plugin_executed"
    
    # System operations
    SYSTEM_STARTED = "system_started"
    SYSTEM_SHUTDOWN = "system_shutdown"


@dataclass
class SimpleSignal:
    """Simple signal payload"""
    signal_type: str
    source_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SimpleSignalEmitter(QObject):
    """Simple signal emitter that just logs signals and emits Qt signals"""
    
    # Qt signal for UI updates
    signal_emitted = Signal(str, str, dict)  # signal_type, source_id, data
    
    def __init__(self):
        super().__init__()
        self.signals_emitted = 0
        self.signal_history = []
        self.max_history = 50
        self.observers = []
    
    def add_observer(self, observer):
        """Add an observer function"""
        self.observers.append(observer)
    
    def emit_signal(self, signal_type: str, source_id: str, data: dict = None):
        """Emit a simple signal"""
        data = data or {}
        self.signals_emitted += 1
        
        # Create signal
        signal = SimpleSignal(
            signal_type=signal_type,
            source_id=source_id,
            data=data
        )
        
        # Store in history
        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]
        
        # Print for debugging
        print(f"📡 [{self.signals_emitted}] {signal_type} from {source_id}")
        if data:
            print(f"    Data: {list(data.keys())}")
        
        # Notify observers
        for observer in self.observers:
            try:
                observer(signal)
            except Exception as e:
                print(f"Observer error: {e}")
        
        # Emit Qt signal for UI
        self.signal_emitted.emit(signal_type, source_id, data)
    
    def get_stats(self):
        """Get simple statistics"""
        return {
            'signals_emitted': self.signals_emitted,
            'observers': len(self.observers),
            'recent_signals': [
                {
                    'type': s.signal_type,
                    'source': s.source_id,
                    'time': s.timestamp.strftime('%H:%M:%S')
                }
                for s in self.signal_history[-5:]
            ]
        }




# =============================================================================
# SIMPLE PLUGIN SYSTEM
# =============================================================================

class Plugin:
    """Base class for plugins"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
    
    def execute(self, word_processor, *args, **kwargs):
        """Execute plugin functionality"""
        pass


class WordCountPlugin(Plugin):
    """Word count plugin"""
    
    def __init__(self):
        super().__init__("Word Count", "1.0.0")
    
    def execute(self, word_processor, *args, **kwargs):
        """Execute word count analysis"""
        editor = word_processor.get_active_editor()
        if not editor:
            return {}
        
        text = editor.toPlainText()
        words = text.split()
        
        stats = {
            'characters': len(text),
            'characters_no_spaces': len(text.replace(' ', '')),
            'words': len(words),
            'sentences': len([s for s in text.split('.') if s.strip()]),
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'lines': text.count('\n') + 1 if text else 0,
            'reading_time_minutes': round(len(words) / 200, 1)  # 200 words per minute
        }
        
        return stats


class StatsPlugin(Plugin):
    """Signal statistics plugin"""
    
    def __init__(self):
        super().__init__("Signal Stats", "1.0.0")
    
    def execute(self, word_processor, *args, **kwargs):
        """Show signal statistics"""
        return word_processor.signal_emitter.get_stats()


class SimplePluginManager(QObject):
    """Simple plugin manager"""
    
    plugin_executed = Signal(str, dict)  # plugin_name, result
    
    def __init__(self, signal_emitter: SimpleSignalEmitter):
        super().__init__()
        self.signal_emitter = signal_emitter
        self.plugins: Dict[str, Plugin] = {}
        self.load_default_plugins()
    
    def load_default_plugins(self):
        """Load default plugins"""
        self.register_plugin(WordCountPlugin())
        self.register_plugin(StatsPlugin())
    
    def register_plugin(self, plugin: Plugin):
        """Register a plugin"""
        self.plugins[plugin.name] = plugin
        
        self.signal_emitter.emit_signal(
            SimpleSignalType.PLUGIN_LOADED,
            "plugin_manager",
            {
                'plugin_name': plugin.name,
                'plugin_version': plugin.version
            }
        )
    
    def execute_plugin(self, plugin_name: str, word_processor, *args, **kwargs):
        """Execute plugin"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            if plugin.enabled:
                try:
                    result = plugin.execute(word_processor, *args, **kwargs)
                    
                    self.plugin_executed.emit(plugin_name, result or {})
                    
                    self.signal_emitter.emit_signal(
                        SimpleSignalType.PLUGIN_EXECUTED,
                        "plugin_manager",
                        {
                            'plugin_name': plugin_name,
                            'success': True,
                            'result_keys': list(result.keys()) if result else []
                        }
                    )
                    
                    return result
                
                except Exception as e:
                    print(f"Plugin error: {e}")
                    self.signal_emitter.emit_signal(
                        SimpleSignalType.PLUGIN_EXECUTED,
                        "plugin_manager",
                        {
                            'plugin_name': plugin_name,
                            'success': False,
                            'error': str(e)
                        }
                    )
        
        return None
    
    def get_plugin_list(self) -> List[str]:
        """Get list of available plugins"""
        return list(self.plugins.keys())



# =============================================================================
# SIGNAL BRIDGE (Enhanced)
# =============================================================================

class SignalManager(QObject):
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


###################################################################################
# TABS - TEXT EDITOR, CONSOLE, NOTEBOOK
###################################################################################


class SimpleTextEditor(QTextEdit):
    """Simple text editor with signal integration"""
    
    # Qt signals for immediate UI responsiveness
    text_inserted = Signal(str, int)  # text, position
    cursor_moved = Signal(int, int)   # line, column
    selection_changed = Signal(int, int)  # start, end
    
    def __init__(self, signal_emitter: SimpleSignalEmitter, document_id: str):
        super().__init__()
        self.signal_emitter = signal_emitter
        self.document_id = document_id
        self.setup_editor()
        self.setup_signals()
        self.setup_shortcuts()
        
        # State tracking
        self.last_cursor_position = 0
        
        # Debounced modification timer
        self.modification_timer = QTimer()
        self.modification_timer.timeout.connect(self.emit_modification_signal)
        self.modification_timer.setSingleShot(True)
    
    def setup_editor(self):
        """Setup editor appearance"""
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
                padding: 15px;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }
            QTextEdit:focus {
                border-color: #007acc;
            }
        """)
        
        # Set font
        font = QFont("Consolas", 14)
        if not font.exactMatch():
            font = QFont("Courier New", 14)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        
        # Configure editor
        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.setTabStopDistance(40)  # 4 spaces
    
    def setup_signals(self):
        """Setup signal connections"""
        self.textChanged.connect(self.on_text_changed)
        self.cursorPositionChanged.connect(self.on_cursor_changed)
        self.selectionChanged.connect(self.on_selection_changed)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Duplicate line
        self.duplicate_line_action = QAction("Duplicate Line", self)
        self.duplicate_line_action.setShortcut(QKeySequence("Ctrl+D"))
        self.duplicate_line_action.triggered.connect(self.duplicate_line)
        self.addAction(self.duplicate_line_action)
    
    def on_text_changed(self):
        """Handle text changes"""
        cursor = self.textCursor()
        position = cursor.position()
        self.text_inserted.emit("", position)
        
        # Start debounced timer
        self.modification_timer.start(300)  # 300ms delay
    
    def emit_modification_signal(self):
        """Emit modification signal (debounced)"""
        content = self.toPlainText()
        word_count = len(content.split()) if content else 0
        char_count = len(content)
        line_count = self.document().lineCount()
        
        # Emit signal
        self.signal_emitter.emit_signal(
            SimpleSignalType.DOCUMENT_MODIFIED,
            f"editor_{self.document_id}",
            {
                'document_id': self.document_id,
                'word_count': word_count,
                'char_count': char_count,
                'line_count': line_count,
                'content_preview': content[:50] + '...' if len(content) > 50 else content
            }
        )
    
    def on_cursor_changed(self):
        """Handle cursor position changes"""
        cursor = self.textCursor()
        block = cursor.block()
        line = block.blockNumber() + 1
        col = cursor.columnNumber() + 1
        
        if cursor.position() != self.last_cursor_position:
            self.last_cursor_position = cursor.position()
            self.cursor_moved.emit(line, col)
            
            # Emit signal (less frequently)
            if cursor.position() % 10 == 0:  # Only every 10 characters
                self.signal_emitter.emit_signal(
                    SimpleSignalType.CURSOR_MOVED,
                    f"editor_{self.document_id}",
                    {
                        'document_id': self.document_id,
                        'line': line,
                        'column': col,
                        'position': cursor.position()
                    }
                )
    
    def on_selection_changed(self):
        """Handle selection changes"""
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        self.selection_changed.emit(start, end)
        
        if cursor.hasSelection():
            self.signal_emitter.emit_signal(
                SimpleSignalType.SELECTION_CHANGED,
                f"editor_{self.document_id}",
                {
                    'document_id': self.document_id,
                    'selection_start': start,
                    'selection_end': end,
                    'selected_length': end - start
                }
            )
    
    def duplicate_line(self):
        """Duplicate current line"""
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        line_text = cursor.selectedText()
        
        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine)
        cursor.insertText('\n' + line_text)


# =============================================================================
# UNIFIED CONSOLE TAB
# =============================================================================

class UnifiedConsoleTab(QTextEdit):
    """Single widget console handling both commands and LLM queries"""
    
    # Signals for external integration
    command_executed = Signal(str, dict)
    llm_query_sent = Signal(str)
    
    def __init__(self, signal_bridge: SignalManager, settings: GlobalSettings):
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


# launch the tabs from here
class SmartTabWidget(QTabWidget):
    """Enhanced tab widget that handles different tab types"""
    
    def __init__(self, settings: GlobalSettings):
        super().__init__()
        self.settings = settings
        self.signal_bridge = SignalManager()
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

class AquaWordProcessor(QMainWindow):
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
        self.setWindowTitle("AQUA")
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
        
        # connect this
        new_debug_action = file_menu.addAction("Debug")
        new_debug_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
        # new_debug_action.triggered.connect()
        
        save_action = file_menu.addAction("Save")
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        # save_action.triggered.connect(self.save_current_document)
        
        save_as_action = file_menu.addAction("Save As...")
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        # save_as_action.triggered.connect(self.save_current_document_as)
        

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
        editor.setPlainText("""Welcome to AQUA Processor!

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



# plugin manager
class SimpleWordProcessor(QMainWindow):
    """Simple word processor with optional signal system"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize simple signal system
        self.signal_emitter = SimpleSignalEmitter()
        
        # Add a simple observer for demonstration
        def demo_observer(signal):
            if signal.signal_type == SimpleSignalType.DOCUMENT_SAVED:
                print(f"✅ Document saved: {signal.data.get('title', 'Unknown')}")
        
        self.signal_emitter.add_observer(demo_observer)
        
        # Initialize plugin manager
        self.plugin_manager = SimplePluginManager(self.signal_emitter)
        
        # Document management
        self.editors: Dict[str, SimpleTextEditor] = {}
        self.document_states: Dict[str, DocumentState] = {}
        self.active_editor_id: Optional[str] = None
        
        # Setup UI
        self.setup_ui()
        self.setup_signal_connections()
        
        # Emit startup signal
        self.signal_emitter.emit_signal(
            SimpleSignalType.SYSTEM_STARTED,
            "main_window",
            {'timestamp': datetime.now().isoformat()}
        )
        
        # Create initial document
        self.create_new_document()
    
    def setup_ui(self):
        """Setup the UI"""
        self.setWindowTitle("Simple Word Processor with Signals")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply theme
        self.apply_theme()
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create editor area
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_document_tab)
        self.editor_tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.editor_tabs)
        
        # Create status bar
        self.create_status_bar()
    
    def apply_theme(self):
        """Apply dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QToolBar {
                background-color: #2d2d30;
                border-bottom: 1px solid #3e3e42;
                spacing: 6px;
                padding: 6px;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #565656;
                color: #d4d4d4;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 70px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #484848;
                border-color: #6c6c6c;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
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
            QTabBar::tab:hover {
                background-color: #3e3e42;
            }
            QStatusBar {
                background-color: #007acc;
                color: white;
                border: none;
                font-weight: 500;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #565656;
                color: #d4d4d4;
                padding: 4px 8px;
                border-radius: 4px;
                min-width: 120px;
            }
            QComboBox:hover {
                background-color: #484848;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #d4d4d4;
                width: 0;
                height: 0;
            }
        """)
    
    def create_toolbar(self):
        """Create toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        
        # File operations
        new_btn = QPushButton("📄 New")
        new_btn.clicked.connect(self.create_new_document)
        toolbar.addWidget(new_btn)
        
        open_btn = QPushButton("📂 Open")
        open_btn.clicked.connect(self.open_document)
        toolbar.addWidget(open_btn)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_document)
        toolbar.addWidget(save_btn)
        
        toolbar.addSeparator()
        
        # Plugin operations
        toolbar.addWidget(QLabel("Plugin:"))
        self.plugin_combo = QComboBox()
        self.plugin_combo.addItems(self.plugin_manager.get_plugin_list())
        toolbar.addWidget(self.plugin_combo)
        
        execute_plugin_btn = QPushButton("▶ Execute")
        execute_plugin_btn.clicked.connect(self.execute_plugin)
        toolbar.addWidget(execute_plugin_btn)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Main status
        self.status_bar.showMessage("Ready")
        
        # Document info
        self.doc_info_label = QLabel("No document")
        self.status_bar.addPermanentWidget(self.doc_info_label)
        
        # Stats
        self.stats_label = QLabel("Signals: 0")
        self.status_bar.addPermanentWidget(self.stats_label)
        
        # Cursor position
        self.cursor_label = QLabel("Ln 1, Col 1")
        self.cursor_label.setStyleSheet("color: white; font-family: monospace; font-weight: bold;")
        self.status_bar.addPermanentWidget(self.cursor_label)
        
        # Update stats periodically
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(2000)  # Every 2 seconds
    
    def setup_signal_connections(self):
        """Setup signal connections"""
        # Plugin manager signals
        self.plugin_manager.plugin_executed.connect(self.on_plugin_executed)
        
        # Signal emitter signals
        self.signal_emitter.signal_emitted.connect(self.on_signal_emitted)
    
    def create_new_document(self):
        """Create a new document"""
        document_id = str(uuid.uuid4())
        
        # Create document state
        doc_state = DocumentState(document_id=document_id)
        self.document_states[document_id] = doc_state
        
        # Create editor
        editor = SimpleTextEditor(self.signal_emitter, document_id)
        self.editors[document_id] = editor
        
        # Connect editor signals
        editor.cursor_moved.connect(self.update_cursor_position)
        editor.text_inserted.connect(lambda text, pos: self.on_text_changed(document_id))
        
        # Add to tabs
        tab_index = self.editor_tabs.addTab(editor, "Untitled")
        self.editor_tabs.setCurrentIndex(tab_index)
        
        self.active_editor_id = document_id
        
        # Emit document opened signal
        self.signal_emitter.emit_signal(
            SimpleSignalType.DOCUMENT_OPENED,
            "main_window",
            {
                'document_id': document_id,
                'title': 'Untitled'
            }
        )
        
        self.update_document_info()
        return document_id
    
    def open_document(self):
        """Open an existing document"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                document_id = str(uuid.uuid4())
                
                # Create document state
                doc_state = DocumentState(
                    document_id=document_id,
                    file_path=file_path,
                    title=Path(file_path).name,
                    content=content
                )
                self.document_states[document_id] = doc_state
                
                # Create editor with content
                editor = SimpleTextEditor(self.signal_emitter, document_id)
                editor.setPlainText(content)
                self.editors[document_id] = editor
                
                # Connect signals
                editor.cursor_moved.connect(self.update_cursor_position)
                editor.text_inserted.connect(lambda text, pos: self.on_text_changed(document_id))
                
                # Add to tabs
                tab_index = self.editor_tabs.addTab(editor, Path(file_path).name)
                self.editor_tabs.setCurrentIndex(tab_index)
                
                self.active_editor_id = document_id
                
                # Emit signal
                self.signal_emitter.emit_signal(
                    SimpleSignalType.DOCUMENT_OPENED,
                    "main_window",
                    {
                        'document_id': document_id,
                        'file_path': file_path,
                        'title': Path(file_path).name
                    }
                )
                
                self.update_document_info()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
    
    def save_document(self):
        """Save current document"""
        if not self.active_editor_id or self.active_editor_id not in self.editors:
            return
        
        editor = self.editors[self.active_editor_id]
        doc_state = self.document_states[self.active_editor_id]
        
        file_path = doc_state.file_path
        
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Document", "", "Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return
        
        try:
            content = editor.toPlainText()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update document state
            doc_state.file_path = file_path
            doc_state.title = Path(file_path).name
            doc_state.content = content
            doc_state.modified = False
            
            # Update tab title
            current_tab = self.editor_tabs.currentIndex()
            self.editor_tabs.setTabText(current_tab, doc_state.title)
            
            # Emit signal
            self.signal_emitter.emit_signal(
                SimpleSignalType.DOCUMENT_SAVED,
                "main_window",
                {
                    'document_id': self.active_editor_id,
                    'file_path': file_path,
                    'title': doc_state.title
                }
            )
            
            self.status_bar.showMessage(f"Saved: {file_path}", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def close_document_tab(self, index: int):
        """Close a document tab"""
        if 0 <= index < self.editor_tabs.count():
            widget = self.editor_tabs.widget(index)
            
            # Find document ID
            document_id = None
            for doc_id, editor in self.editors.items():
                if editor == widget:
                    document_id = doc_id
                    break
            
            if document_id:
                # Remove tab and cleanup
                self.editor_tabs.removeTab(index)
                del self.editors[document_id]
                del self.document_states[document_id]
                
                # Update active document
                if document_id == self.active_editor_id:
                    self.active_editor_id = None
                    if self.editor_tabs.count() > 0:
                        self.on_tab_changed(self.editor_tabs.currentIndex())
                
                # Emit signal
                self.signal_emitter.emit_signal(
                    SimpleSignalType.DOCUMENT_CLOSED,
                    "main_window",
                    {'document_id': document_id}
                )
    
    def on_tab_changed(self, index: int):
        """Handle tab change"""
        if 0 <= index < self.editor_tabs.count():
            widget = self.editor_tabs.widget(index)
            
            # Find document ID
            for doc_id, editor in self.editors.items():
                if editor == widget:
                    self.active_editor_id = doc_id
                    break
            
            self.update_document_info()
    
    def on_text_changed(self, document_id: str):
        """Handle text changes"""
        if document_id in self.editors and document_id in self.document_states:
            doc_state = self.document_states[document_id]
            doc_state.modified = True
            
            # Update tab title
            for i in range(self.editor_tabs.count()):
                if self.editor_tabs.widget(i) == self.editors[document_id]:
                    title = doc_state.title
                    if not title.endswith(" •"):
                        self.editor_tabs.setTabText(i, f"{title} •")
                    break
    
    def update_cursor_position(self, line: int, col: int):
        """Update cursor position display"""
        self.cursor_label.setText(f"Ln {line}, Col {col}")
    
    def update_document_info(self):
        """Update document info"""
        if self.active_editor_id and self.active_editor_id in self.document_states:
            doc_state = self.document_states[self.active_editor_id]
            title = doc_state.title
            if doc_state.modified:
                title += " •"
            self.doc_info_label.setText(title)
        else:
            self.doc_info_label.setText("No document")
    
    def update_stats(self):
        """Update signal statistics"""
        stats = self.signal_emitter.get_stats()
        self.stats_label.setText(f"Signals: {stats['signals_emitted']}")
    
    def execute_plugin(self):
        """Execute selected plugin"""
        plugin_name = self.plugin_combo.currentText()
        if plugin_name:
            result = self.plugin_manager.execute_plugin(plugin_name, self)
            if result:
                self.status_bar.showMessage(f"Plugin '{plugin_name}' executed", 2000)
    
    def on_plugin_executed(self, plugin_name: str, result: Dict[str, Any]):
        """Handle plugin execution results"""
        if plugin_name == "Word Count":
            if result:
                stats_text = "\n".join([
                    f"{k.replace('_', ' ').title()}: {v}" 
                    for k, v in result.items()
                ])
                QMessageBox.information(self, "Word Count Results", stats_text)
        
        elif plugin_name == "Signal Stats":
            if result:
                recent_signals = result.get('recent_signals', [])
                signals_text = "Recent Signals:\n\n" + "\n".join([
                    f"{s['time']} - {s['type']} from {s['source']}"
                    for s in recent_signals
                ])
                signals_text += f"\n\nTotal Signals: {result.get('signals_emitted', 0)}"
                signals_text += f"\nObservers: {result.get('observers', 0)}"
                
                QMessageBox.information(self, "Signal Statistics", signals_text)
    
    def on_signal_emitted(self, signal_type: str, source_id: str, data: dict):
        """Handle signal emissions for debugging"""
        # This runs on every signal emission - useful for debugging
        pass  # Could add specific UI responses here
    
    def get_active_editor(self) -> Optional[SimpleTextEditor]:
        """Get the active editor"""
        if self.active_editor_id and self.active_editor_id in self.editors:
            return self.editors[self.active_editor_id]
        return None
    
    def closeEvent(self, event):
        """Handle application close"""
        # Check for unsaved documents
        unsaved_docs = []
        for doc_state in self.document_states.values():
            if doc_state.modified:
                unsaved_docs.append(doc_state.title)
        
        if unsaved_docs:
            reply = QMessageBox.question(
                self, "Unsaved Documents",
                f"You have {len(unsaved_docs)} unsaved document(s). Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        # Emit shutdown signal
        self.signal_emitter.emit_signal(
            SimpleSignalType.SYSTEM_SHUTDOWN,
            "main_window",
            {
                'documents_count': len(self.editors),
                'signals_emitted': self.signal_emitter.signals_emitted
            }
        )
        
        event.accept()

# ============= UI Integration =============
# word_processor_integration.py
class SmartWordProcessor(QMainWindow):
    """Main word processor window with LLM integration"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Word Processor")
        
        # Initialize document model
        self.document = DocumentModel()
        
        # Setup UI
        self.editor = QTextEdit()
        self.setCentralWidget(self.editor)
        
        # Connect signals using Qt slots
        self.document.textChanged.connect(self._on_text_changed)
        self.document.llmSuggestionReceived.connect(self._show_suggestions)
        self.document.documentSaved.connect(self._on_document_saved)
        
        # Connect editor to model
        self.editor.textChanged.connect(self._sync_to_model)
        
        # Setup status monitoring
        self.status_monitor = QTimer()
        self.status_monitor.timeout.connect(self._update_status)
        self.status_monitor.start(1000)  # Update every second
    
    @Slot()
    def _on_text_changed(self, text: str):
        """Handle text changes from model"""
        if self.editor.toPlainText() != text:
            # Preserve cursor position
            cursor = self.editor.textCursor()
            pos = cursor.position()
            
            self.editor.setPlainText(text)
            
            cursor.setPosition(min(pos, len(text)))
            self.editor.setTextCursor(cursor)
    
    @Slot()
    def _sync_to_model(self):
        """Sync editor changes to model"""
        self.document.content = self.editor.toPlainText()
        self.document._cursor_position = self.editor.textCursor().position()
    
    @Slot(list)
    def _show_suggestions(self, suggestions: List[str]):
        """Display LLM suggestions"""
        # Implementation depends on your UI design
        # Could show in a popup, sidebar, or inline
        print(f"Suggestions: {suggestions}")
    
    @Slot()
    def _on_document_saved(self):
        """Handle document saved signal"""
        self.statusBar().showMessage("Document saved", 2000)
    
    @Slot()
    def _update_status(self):
        """Update status bar with connection info"""
        if hasattr(self.document._llm_wrapper, '_connection_status'):
            status = self.document._llm_wrapper._connection_status
            active = self.document._llm_wrapper._active_provider
            
            if active and active in status:
                msg = f"LLM: {active} - {status[active]}"
                self.statusBar().showMessage(msg)




# observers from llm_provider_signals.py

# =============================================================================
# INTEGRATION WITH UNIFIED CONSOLE
# =============================================================================

class LLMConsoleIntegration:
    """Integration layer between LLM system and unified console."""
    
    def __init__(self, console_tab, signal_line: Optional[SignalLine] = None):
        self.console_tab = console_tab
        self.signal_line = signal_line
        self.llm_manager = LLMManager(signal_line)
        
        # Observers
        self.console_observer = None
        self.metrics_observer = MetricsObserver()
        
        # State
        self.current_query_id = None
        self.response_buffer = ""
        
    async def initialize(self, signal_line: SignalLine):
        """Initialize with signal line."""
        self.signal_line = signal_line
        await self.llm_manager.set_signal_line(signal_line)
        
        # Create console observer with callback to console
        self.console_observer = ConsoleObserver(self._on_llm_message)
        
        # Register observers
        await self.llm_manager.add_observer_to_all(self.console_observer)
        await self.llm_manager.add_observer_to_all(self.metrics_observer)
        
        logger.info("LLM Console Integration initialized")
    
    def _on_llm_message(self, message: str, signal_type: str):
        """Handle LLM messages and forward to console."""
        if hasattr(self.console_tab, 'append_formatted_text'):
            # Determine message type for formatting
            if signal_type == LLMSignalType.LLM_STREAM_CHUNK.value:
                self.console_tab.append_formatted_text(message, "llm")
            elif "error" in signal_type.lower() or "failed" in signal_type.lower():
                self.console_tab.append_formatted_text(message + "\n", "error")
            elif "connected" in signal_type.lower() or "completed" in signal_type.lower():
                self.console_tab.append_formatted_text(message + "\n", "success")
            else:
                self.console_tab.append_formatted_text(message + "\n", "info")
    
    async def connect_to_ollama(self, url: str = "http://localhost:11434") -> bool:
        """Connect to Ollama service."""
        return await self.llm_manager.connect_provider("ollama", url)
    
    async def disconnect_from_llm(self):
        """Disconnect from current LLM provider."""
        await self.llm_manager.disconnect_provider()
    
    async def send_query(self, prompt: str, model: str = "phi", **kwargs) -> str:
        """Send query to LLM and return final response."""
        full_response = ""
        
        async for response in self.llm_manager.send_query(prompt, model, **kwargs):
            if response.error:
                return f"Error: {response.error}"
            
            if response.content:
                full_response += response.content
            
            if response.is_complete:
                break
        
        return full_response
    
    async def send_streaming_query(self, prompt: str, model: str = "phi", **kwargs):
        """Send query to LLM with streaming response (signals handle UI updates)."""
        self.current_query_id = str(uuid.uuid4())
        self.response_buffer = ""
        
        async for response in self.llm_manager.send_query(prompt, model, **kwargs):
            if response.content:
                self.response_buffer += response.content
            
            if response.is_complete:
                self.current_query_id = None
                break
    
    def get_available_models(self) -> List[str]:
        """Get available models from active provider."""
        provider = self.llm_manager.get_active_provider()
        if provider:
            return provider.available_models
        return []
    
    def is_connected(self) -> bool:
        """Check if connected to LLM."""
        provider = self.llm_manager.get_active_provider()
        return provider.connected if provider else False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        provider = self.llm_manager.get_active_provider()
        if provider:
            return {
                'connected': provider.connected,
                'provider': provider.provider_id,
                'url': provider.connection_url,
                'model': provider.current_model,
                'available_models': len(provider.available_models),
                'stats': provider.get_stats()
            }
        return {
            'connected': False,
            'provider': None,
            'url': None,
            'model': None,
            'available_models': 0,
            'stats': {}
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get LLM usage metrics."""
        return self.metrics_observer.get_metrics()



# =============================================================================
# ENHANCED UNIFIED CONSOLE TAB WITH LLM INTEGRATION
# =============================================================================
# llm_provider_signals.py
class EnhancedUnifiedConsoleTab:
    """Enhanced version of UnifiedConsoleTab with LLM integration."""
    
    def __init__(self, signal_bridge, settings, signal_line: Optional[SignalLine] = None):
        # Initialize base console (assume this exists from your original code)
        self.signal_bridge = signal_bridge
        self.settings = settings
        self.signal_line = signal_line
        
        # LLM Integration
        self.llm_integration = LLMConsoleIntegration(self, signal_line)
        
        # Enhanced state
        self.llm_initialized = False
        
    async def initialize_llm_system(self, signal_line: SignalLine):
        """Initialize the LLM system with signal line."""
        self.signal_line = signal_line
        await self.llm_integration.initialize(signal_line)
        self.llm_initialized = True
        
        # Update console state
        self.llm_connected = False
        self.llm_url = self.settings.llm_settings.get('default_url', 'http://localhost:11434')
        self.llm_model = self.settings.llm_settings.get('default_model', 'phi')
        
        logger.info("Enhanced Unified Console Tab initialized with LLM system")
    
    async def connect_to_llm(self, url: str):
        """Connect to LLM service using signal system."""
        if not self.llm_initialized:
            self.append_formatted_text("❌ LLM system not initialized\n", "error")
            return
        
        success = await self.llm_integration.connect_to_ollama(url)
        if success:
            self.llm_connected = True
            self.llm_url = url
            
            # Load available models
            models = self.llm_integration.get_available_models()
            if models and not self.llm_model in models:
                self.llm_model = models[0]  # Use first available model
        else:
            self.llm_connected = False
    
    async def disconnect_from_llm(self):
        """Disconnect from LLM service."""
        if self.llm_initialized:
            await self.llm_integration.disconnect_from_llm()
        self.llm_connected = False
    
    async def execute_llm_query(self, query: str):
        """Execute LLM query with signal-based streaming."""
        if not self.llm_connected:
            self.append_formatted_text("❌ Not connected to LLM. Use /connect <url>\n", "error")
            return
        
        if not self.llm_initialized:
            self.append_formatted_text("❌ LLM system not initialized\n", "error")
            return
        
        # Send streaming query (UI updates handled by signals)
        await self.llm_integration.send_streaming_query(query, self.llm_model)
    
    def execute_special_command(self, command: str):
        """Enhanced special command execution with LLM commands."""
        parts = command[1:].split()
        cmd = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        # LLM-specific commands
        if cmd == "llm_connect" or cmd == "connect":
            url = args[0] if args else self.llm_url
            asyncio.create_task(self.connect_to_llm(url))
        
        elif cmd == "llm_disconnect" or cmd == "disconnect":
            asyncio.create_task(self.disconnect_from_llm())
        
        elif cmd == "llm_models" or cmd == "models":
            models = self.llm_integration.get_available_models()
            if models:
                self.append_formatted_text("Available models:\n", "info")
                for model in models:
                    marker = " (current)" if model == self.llm_model else ""
                    self.append_formatted_text(f"  - {model}{marker}\n", "output")
            else:
                self.append_formatted_text("No models available\n", "info")
        
        elif cmd == "llm_model" or cmd == "model":
            if args:
                self.llm_model = args[0]
                self.append_formatted_text(f"Model set to: {self.llm_model}\n", "success")
            else:
                self.append_formatted_text(f"Current model: {self.llm_model}\n", "info")
        
        elif cmd == "llm_status":
            status = self.llm_integration.get_connection_status()
            self.show_llm_status(status)
        
        elif cmd == "llm_metrics":
            metrics = self.llm_integration.get_metrics()
            self.show_llm_metrics(metrics)
        
        elif cmd == "llm_test":
            test_prompt = " ".join(args) if args else "Hello, how are you?"
            asyncio.create_task(self.execute_llm_query(test_prompt))
        
        else:
            # Fall back to original special commands
            # (call original method if it exists)
            pass
    
    def show_llm_status(self, status: Dict[str, Any]):
        """Show detailed LLM status."""
        connected = status.get('connected', False)
        provider = status.get('provider', 'Unknown')
        url = status.get('url', 'Unknown')
        model = status.get('model', 'Unknown')
        
        status_text = f"""
LLM Status:
  Connected:    {connected}
  Provider:     {provider}
  URL:          {url}  
  Model:        {model}
  Available:    {status.get('available_models', 0)} models
  
"""
        self.append_formatted_text(status_text, "info")
        
        # Show stats if available
        stats = status.get('stats', {})
        if stats:
            stats_text = f"""Statistics:
  Queries Sent:     {stats.get('queries_sent', 0)}
  Queries Completed: {stats.get('queries_completed', 0)}
  Success Rate:     {stats.get('success_rate', 0):.1%}
  Avg Time:         {stats.get('avg_processing_time', 0):.2f}s

"""
            self.append_formatted_text(stats_text, "info")
    
    def show_llm_metrics(self, metrics: Dict[str, Any]):
        """Show LLM usage metrics."""
        metrics_text = f"""
LLM Metrics:
  Total Queries:    {metrics.get('total_queries', 0)}
  Successful:       {metrics.get('successful_queries', 0)}
  Failed:           {metrics.get('failed_queries', 0)}
  Success Rate:     {metrics.get('success_rate', 0):.1%}
  
  Processing Time:  {metrics.get('total_processing_time', 0):.2f}s total
  Average Time:     {metrics.get('avg_processing_time', 0):.2f}s
  
  Tokens:           {metrics.get('total_tokens', 0)}
  Chunks Received:  {metrics.get('chunks_received', 0)}

"""
        self.append_formatted_text(metrics_text, "info")
    
    def show_help(self):
        """Enhanced help with LLM commands."""
        help_text = """
Enhanced Unified Console Help:

LLM Commands:
  /connect <url>     - Connect to LLM service (default: localhost:11434)
  /disconnect        - Disconnect from LLM
  /models            - List available models
  /model <name>      - Set current model
  /llm_status        - Show detailed LLM status and stats
  /llm_metrics       - Show LLM usage metrics
  /llm_test [prompt] - Test LLM with prompt

Standard Commands:
  /help              - Show this help
  /clear             - Clear console
  /cd <dir>          - Change directory
  /pwd               - Show current directory
  /status            - Show system status
  /history           - Show command history
  /settings          - Open settings dialog

Input Routing:
  Regular text       → LLM query (if connected)
  $ command          → Force system command
  ! command          → Force system command  
  cmd: command       → Force system command
  /command           → Special console command

Navigation:
  ↑/↓ arrows        - Navigate command history
  Home              - Go to start of input line

Examples:
  /connect http://localhost:11434  → Connect to local Ollama
  /models                         → List available models
  /model phi                      → Switch to phi model
  what is python?                 → LLM query
  $ ls -la                        → System command

"""
        self.append_formatted_text(help_text, "help")
    
    # Placeholder for methods that would exist in original implementation
    def append_formatted_text(self, text: str, text_type: str = "output"):
        """Append formatted text to console (placeholder)."""
        # This would call the original implementation
        print(f"[{text_type}] {text}", end="")




class DebugWindow(QMainWindow):
    """Debug window to monitor signals in real-time"""
    
    def __init__(self, signal_emitter: SimpleSignalEmitter):
        super().__init__()
        self.signal_emitter = signal_emitter
        self.setup_ui()
        
        # Add observer to monitor signals
        self.signal_emitter.add_observer(self.on_signal_received)
        
        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(1000)  # Every second
    
    def setup_ui(self):
        """Setup debug UI"""
        self.setWindowTitle("Signal Monitor")
        self.setGeometry(1300, 100, 800, 600)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #d4d4d4; }
            QTextEdit { 
                background-color: #0d1117; color: #c9d1d9; 
                border: 1px solid #30363d; font-family: 'Consolas', monospace;
                font-size: 12px;
            }
            QLabel { color: #f0f6fc; font-weight: bold; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Stats display
        self.stats_label = QLabel("Signal Statistics")
        layout.addWidget(self.stats_label)
        
        # Signal log
        self.signal_log = QTextEdit()
        self.signal_log.setReadOnly(True)
        layout.addWidget(self.signal_log)
        
        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_btn)
    
    def on_signal_received(self, signal: SimpleSignal):
        """Handle new signal"""
        timestamp = signal.timestamp.strftime('%H:%M:%S.%f')[:-3]
        
        # Color code by signal type
        if 'DOCUMENT' in signal.signal_type:
            color = '#58a6ff'  # Blue
        elif 'PLUGIN' in signal.signal_type:
            color = '#a5f3fc'  # Cyan
        elif 'CURSOR' in signal.signal_type:
            color = '#6b7280'  # Gray
        else:
            color = '#f0f6fc'  # White
        
        # Add to log
        log_entry = f"""
<span style="color: {color};">
[{timestamp}] <b>{signal.signal_type}</b> from {signal.source_id}
</span>"""
        
        if signal.data:
            data_preview = {k: str(v)[:30] + '...' if len(str(v)) > 30 else v 
                          for k, v in signal.data.items()}
            log_entry += f"""<br><span style="color: #7c3aed; margin-left: 20px;">
Data: {data_preview}
</span>"""
        
        self.signal_log.append(log_entry)
        
        # Auto-scroll to bottom
        scrollbar = self.signal_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Limit log size
        if self.signal_log.document().blockCount() > 200:
            cursor = self.signal_log.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
    
    def refresh_display(self):
        """Refresh statistics display"""
        stats = self.signal_emitter.get_stats()
        self.stats_label.setText(
            f"📊 Signals: {stats['signals_emitted']} | "
            f"Observers: {stats['observers']} | "
            f"Recent: {len(stats['recent_signals'])}"
        )
    
    def clear_log(self):
        """Clear the signal log"""
        self.signal_log.clear()


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AQUA")
    app.setApplicationVersion("0.0")
    app.setOrganizationName("CloudyCadet")
    
    # Create and show main window
    window = AquaWordProcessor()
    window.show()
    
    print("AQUA 🚀 Enhanced Word Processor with Unified Console started!")

    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())