#!/usr/bin/env python3
"""
Prompt Stage

A command prompt as a pipeline stage:
- Receives input from user (keyboard)
- Receives input from LLM (streaming)
- Parses commands
- Emits signals for execution
- Outputs to a Document

Uses the Document class for display, not raw QTextEdit.
"""

from __future__ import annotations
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QKeyEvent, QTextCursor, QFont, QColor

from document import Document, SpanStyle, Styles


# =============================================================================
# COMMAND TYPES
# =============================================================================

class CommandType(Enum):
    """Types of parsed commands."""
    SPECIAL = auto()      # /command
    SYSTEM = auto()       # $ or ! prefix
    LLM_QUERY = auto()    # Default text → LLM
    INTERNAL = auto()     # Internal pipeline command


@dataclass
class ParsedCommand:
    """A parsed command ready for execution."""
    command_type: CommandType
    command: str
    args: Tuple[str, ...]
    kwargs: Dict[str, Any]
    raw_input: str
    source: str = "user"  # "user", "llm", "system"
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# PROMPT STYLES
# =============================================================================

class PromptStyles:
    """Styles specific to the prompt."""
    PROMPT = SpanStyle(foreground="#89ddff", bold=True)
    USER_INPUT = SpanStyle(foreground="#c0caf5")
    COMMAND = SpanStyle(foreground="#9ece6a")
    LLM_THINKING = SpanStyle(foreground="#7aa2f7", italic=True)
    LLM_RESPONSE = SpanStyle(foreground="#c0caf5")
    ERROR = SpanStyle(foreground="#ff6b6b")
    SUCCESS = SpanStyle(foreground="#9ece6a")
    INFO = SpanStyle(foreground="#7aa2f7")
    SYSTEM = SpanStyle(foreground="#565f89", italic=True)


# =============================================================================
# PROMPT STAGE
# =============================================================================

class PromptStage(Document):
    """
    Command prompt as a pipeline stage.
    
    Signals:
        command_parsed(ParsedCommand): Emitted when a command is parsed
        llm_query(str): Emitted when text should go to LLM
        system_command(str): Emitted for shell commands
        special_command(str, tuple): Emitted for /commands
    
    The prompt maintains an input position and handles:
    - Command history (up/down arrows)
    - Command parsing and routing
    - LLM response streaming display
    - Output from command execution
    """
    
    # Signals
    command_parsed = Signal(object)  # ParsedCommand
    llm_query = Signal(str)
    system_command = Signal(str)
    special_command = Signal(str, tuple)  # command, args
    
    def __init__(self, parent=None):
        super().__init__(parent, read_only=False)
        
        # Prompt configuration
        self.prompt_symbol = "> "
        self.special_prefix = "/"
        self.system_prefixes = ["$", "!", "cmd:"]
        self.default_to_llm = True
        
        # State
        self._input_start_pos = 0
        self._history: List[str] = []
        self._history_index = 0
        self._waiting_for_llm = False
        self._llm_connected = False
        self._current_directory = Path.cwd()
        
        # Active LLM stream
        self._llm_stream_id: Optional[str] = None
        
        # Setup
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup prompt appearance and behavior."""
        # Override document style
        self.setStyleSheet("""
            QTextEdit {
                background-color: #0c0c0c;
                color: #cccccc;
                border: none;
                padding: 8px;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }
        """)
        
        font = QFont("Consolas", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        
        # Show initial prompt
        self._show_prompt()
    
    # -------------------------------------------------------------------------
    # PROMPT DISPLAY
    # -------------------------------------------------------------------------
    
    def _show_prompt(self):
        """Show the command prompt."""
        self.append_text(self.prompt_symbol, PromptStyles.PROMPT)
        self._input_start_pos = self.textCursor().position()
    
    def _new_line(self):
        """Move to a new line."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n")
        self.setTextCursor(cursor)
    
    def _get_current_input(self) -> str:
        """Get text from input start to end."""
        cursor = self.textCursor()
        cursor.setPosition(self._input_start_pos)
        cursor.movePosition(
            QTextCursor.MoveOperation.End,
            QTextCursor.MoveMode.KeepAnchor
        )
        return cursor.selectedText()
    
    def _clear_current_input(self):
        """Clear current input line."""
        cursor = self.textCursor()
        cursor.setPosition(self._input_start_pos)
        cursor.movePosition(
            QTextCursor.MoveOperation.End,
            QTextCursor.MoveMode.KeepAnchor
        )
        cursor.removeSelectedText()
    
    # -------------------------------------------------------------------------
    # INPUT HANDLING
    # -------------------------------------------------------------------------
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key presses with prompt-aware logic."""
        cursor = self.textCursor()
        current_pos = cursor.position()
        
        # Don't allow editing before input start
        if current_pos < self._input_start_pos:
            if event.key() not in (Qt.Key.Key_End, Qt.Key.Key_Down, Qt.Key.Key_Up):
                cursor.setPosition(self._input_start_pos)
                self.setTextCursor(cursor)
                if event.key() not in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                    return
        
        # Handle special keys
        if event.key() == Qt.Key.Key_Return and not event.modifiers():
            self._execute_current_input()
            return
        
        elif event.key() == Qt.Key.Key_Up:
            self._navigate_history(-1)
            return
        
        elif event.key() == Qt.Key.Key_Down:
            self._navigate_history(1)
            return
        
        elif event.key() == Qt.Key.Key_Backspace:
            if current_pos <= self._input_start_pos:
                return  # Don't delete prompt
        
        elif event.key() == Qt.Key.Key_Home:
            cursor.setPosition(self._input_start_pos)
            self.setTextCursor(cursor)
            return
        
        elif event.key() == Qt.Key.Key_Left:
            if current_pos <= self._input_start_pos:
                return
        
        # Default handling
        super().keyPressEvent(event)
    
    def _execute_current_input(self):
        """Execute the current input line."""
        input_text = self._get_current_input().strip()
        self._new_line()
        
        if not input_text:
            self._show_prompt()
            return
        
        # Add to history
        if not self._history or self._history[-1] != input_text:
            self._history.append(input_text)
        self._history_index = len(self._history)
        
        # Parse and route
        parsed = self._parse_input(input_text)
        self.command_parsed.emit(parsed)
        
        # Route based on type
        if parsed.command_type == CommandType.SPECIAL:
            self._handle_special_command(parsed)
        elif parsed.command_type == CommandType.SYSTEM:
            self._handle_system_command(parsed)
        elif parsed.command_type == CommandType.LLM_QUERY:
            self._handle_llm_query(parsed)
        else:
            self._show_prompt()
    
    def _parse_input(self, text: str) -> ParsedCommand:
        """Parse input into a command."""
        text = text.strip()
        
        # Check for special command (/command)
        if text.startswith(self.special_prefix):
            parts = text[len(self.special_prefix):].split()
            command = parts[0] if parts else ""
            args = tuple(parts[1:]) if len(parts) > 1 else ()
            return ParsedCommand(
                command_type=CommandType.SPECIAL,
                command=command,
                args=args,
                kwargs={},
                raw_input=text
            )
        
        # Check for system command ($ or !)
        for prefix in self.system_prefixes:
            if text.startswith(prefix):
                command = text[len(prefix):].strip()
                return ParsedCommand(
                    command_type=CommandType.SYSTEM,
                    command=command,
                    args=(),
                    kwargs={},
                    raw_input=text
                )
        
        # Default: LLM query or plain text
        if self.default_to_llm and self._llm_connected:
            return ParsedCommand(
                command_type=CommandType.LLM_QUERY,
                command=text,
                args=(),
                kwargs={},
                raw_input=text
            )
        
        # Not connected to LLM, treat as unknown
        return ParsedCommand(
            command_type=CommandType.INTERNAL,
            command=text,
            args=(),
            kwargs={},
            raw_input=text
        )
    
    # -------------------------------------------------------------------------
    # COMMAND HANDLERS
    # -------------------------------------------------------------------------
    
    def _handle_special_command(self, cmd: ParsedCommand):
        """Handle /commands internally or emit signal."""
        command = cmd.command.lower()
        args = cmd.args
        
        # Built-in commands
        if command == "help":
            self._show_help()
        elif command == "clear":
            self.clear_document()
            self._show_prompt()
            return
        elif command == "history":
            self._show_history()
        elif command == "pwd":
            self.append_line(str(self._current_directory), PromptStyles.INFO)
        elif command == "cd":
            self._change_directory(args[0] if args else str(Path.home()))
        else:
            # Emit for external handling
            self.special_command.emit(command, args)
        
        self._show_prompt()
    
    def _handle_system_command(self, cmd: ParsedCommand):
        """Handle system shell commands."""
        self.append_line(f"$ {cmd.command}", PromptStyles.COMMAND)
        self.system_command.emit(cmd.command)
        # Note: actual execution should be done by signal handler
        # which then calls output_result()
    
    def _handle_llm_query(self, cmd: ParsedCommand):
        """Handle LLM queries."""
        if not self._llm_connected:
            self.append_line("Not connected to LLM. Use /connect", PromptStyles.ERROR)
            self._show_prompt()
            return
        
        self._waiting_for_llm = True
        self.append_text("🤖 ", PromptStyles.LLM_THINKING)
        
        # Start stream for response
        self._llm_stream_id = self.stream_start(PromptStyles.LLM_RESPONSE)
        
        # Emit query
        self.llm_query.emit(cmd.command)
    
    # -------------------------------------------------------------------------
    # OUTPUT METHODS (for external use)
    # -------------------------------------------------------------------------
    
    def output_result(self, text: str, style: SpanStyle = None):
        """Output result text (from command execution, etc.)."""
        self.append_line(text, style or PromptStyles.INFO)
        if not self._waiting_for_llm:
            self._show_prompt()
    
    def output_error(self, text: str):
        """Output error text."""
        self.append_line(text, PromptStyles.ERROR)
        if not self._waiting_for_llm:
            self._show_prompt()
    
    def output_success(self, text: str):
        """Output success text."""
        self.append_line(text, PromptStyles.SUCCESS)
        if not self._waiting_for_llm:
            self._show_prompt()
    
    # -------------------------------------------------------------------------
    # LLM STREAMING INTERFACE
    # -------------------------------------------------------------------------
    
    def set_llm_connected(self, connected: bool, message: str = ""):
        """Set LLM connection status."""
        self._llm_connected = connected
        status = "Connected" if connected else "Disconnected"
        style = PromptStyles.SUCCESS if connected else PromptStyles.ERROR
        self.append_line(f"[{status}] {message}", style)
        if not self._waiting_for_llm:
            self._show_prompt()
    
    def receive_llm_chunk(self, chunk: str):
        """Receive streaming chunk from LLM."""
        if self._llm_stream_id:
            self.stream_chunk(chunk, self._llm_stream_id)
    
    def receive_llm_complete(self, full_response: str = ""):
        """LLM response complete."""
        if self._llm_stream_id:
            self.stream_complete(self._llm_stream_id)
            self._llm_stream_id = None
        
        self._new_line()
        self._waiting_for_llm = False
        self._show_prompt()
    
    def receive_llm_error(self, error: str):
        """LLM error received."""
        if self._llm_stream_id:
            self.stream_complete(self._llm_stream_id)
            self._llm_stream_id = None
        
        self.append_line(f"Error: {error}", PromptStyles.ERROR)
        self._waiting_for_llm = False
        self._show_prompt()
    
    # -------------------------------------------------------------------------
    # HISTORY
    # -------------------------------------------------------------------------
    
    def _navigate_history(self, delta: int):
        """Navigate command history."""
        if not self._history:
            return
        
        self._clear_current_input()
        
        self._history_index = max(0, min(len(self._history), self._history_index + delta))
        
        if self._history_index < len(self._history):
            history_text = self._history[self._history_index]
            cursor = self.textCursor()
            cursor.setPosition(self._input_start_pos)
            cursor.insertText(history_text)
    
    def _show_history(self):
        """Show command history."""
        if not self._history:
            self.append_line("No command history", PromptStyles.INFO)
            return
        
        self.append_line("Command History:", PromptStyles.INFO)
        for i, cmd in enumerate(self._history[-20:], 1):  # Last 20
            self.append_line(f"  {i}. {cmd}", PromptStyles.SYSTEM)
    
    # -------------------------------------------------------------------------
    # BUILT-IN COMMANDS
    # -------------------------------------------------------------------------
    
    def _show_help(self):
        """Show help text."""
        help_text = """
Prompt Commands:
  /help          - Show this help
  /clear         - Clear console
  /history       - Show command history
  /pwd           - Show current directory
  /cd <dir>      - Change directory
  /connect       - Connect to LLM
  /disconnect    - Disconnect from LLM
  /model <name>  - Set LLM model

Input Routing:
  Regular text   → LLM query (if connected)
  $ command      → System shell command
  ! command      → System shell command
  /command       → Special command

Navigation:
  ↑/↓ arrows     - Navigate history
  Home           - Go to input start
"""
        for line in help_text.strip().split('\n'):
            self.append_line(line, PromptStyles.INFO)
    
    def _change_directory(self, path: str):
        """Change working directory."""
        try:
            new_dir = Path(path).resolve()
            if new_dir.exists() and new_dir.is_dir():
                self._current_directory = new_dir
                self.append_line(f"Changed to: {new_dir}", PromptStyles.SUCCESS)
            else:
                self.append_line(f"Directory not found: {path}", PromptStyles.ERROR)
        except Exception as e:
            self.append_line(f"Error: {e}", PromptStyles.ERROR)


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Prompt Stage Test")
    window.resize(800, 400)
    
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    
    prompt = PromptStage()
    layout.addWidget(prompt)
    
    # Connect signals for testing
    def on_special(cmd, args):
        print(f"Special command: /{cmd} {args}")
        if cmd == "test":
            prompt.output_success("Test command executed!")
        elif cmd == "connect":
            prompt.set_llm_connected(True, "Simulated LLM connection")
    
    def on_system(cmd):
        print(f"System command: $ {cmd}")
        prompt.output_result(f"Would execute: {cmd}")
    
    def on_llm_query(query):
        print(f"LLM query: {query}")
        # Simulate streaming response
        import threading
        def simulate_response():
            import time
            chunks = ["Hello", "! ", "I'm ", "a ", "simulated ", "LLM ", "response."]
            for chunk in chunks:
                time.sleep(0.1)
                # Must use invokeMethod or similar for thread safety
                QTimer.singleShot(0, lambda c=chunk: prompt.receive_llm_chunk(c))
            time.sleep(0.1)
            QTimer.singleShot(0, prompt.receive_llm_complete)
        
        threading.Thread(target=simulate_response, daemon=True).start()
    
    prompt.special_command.connect(on_special)
    prompt.system_command.connect(on_system)
    prompt.llm_query.connect(on_llm_query)
    
    window.setCentralWidget(central)
    window.show()
    
    # Initial message
    prompt.append_line("Welcome to Prompt Stage", PromptStyles.INFO)
    prompt.append_line("Type /help for commands, /connect to simulate LLM", PromptStyles.SYSTEM)
    
    sys.exit(app.exec())
