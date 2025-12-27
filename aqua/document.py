#!/usr/bin/env python3
"""
Document Class

General purpose rich text display widget with:
- Styled spans (color, font, background, borders)
- Streaming text support
- Interactive regions (click handlers)
- Animation hooks
- Signal emission for clicks and events

This replaces direct QTextEdit usage throughout the application.
"""

from __future__ import annotations
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from PySide6.QtWidgets import QTextEdit, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QEvent
from PySide6.QtGui import (
    QFont, QColor, QTextCursor, QTextCharFormat, QTextDocument,
    QMouseEvent, QPainter, QBrush, QPen
)


# =============================================================================
# SPAN STYLING
# =============================================================================

@dataclass
class SpanStyle:
    """Style definition for a text span."""
    foreground: str = "#d4d4d4"
    background: str = ""
    font_family: str = ""
    font_size: int = 0  # 0 means use default
    bold: bool = False
    italic: bool = False
    underline: bool = False
    
    # For interactive spans
    hover_foreground: str = ""
    hover_background: str = ""
    cursor: str = "pointing"  # "pointing", "text", "default"
    
    def to_char_format(self) -> QTextCharFormat:
        """Convert to Qt text format."""
        fmt = QTextCharFormat()
        
        if self.foreground:
            fmt.setForeground(QColor(self.foreground))
        if self.background:
            fmt.setBackground(QColor(self.background))
        if self.font_family:
            font = QFont(self.font_family)
            if self.font_size > 0:
                font.setPointSize(self.font_size)
            fmt.setFont(font)
        elif self.font_size > 0:
            font = fmt.font()
            font.setPointSize(self.font_size)
            fmt.setFont(font)
        
        if self.bold:
            fmt.setFontWeight(QFont.Weight.Bold)
        if self.italic:
            fmt.setFontItalic(True)
        if self.underline:
            fmt.setFontUnderline(True)
        
        return fmt


# Preset styles
class Styles:
    """Common preset styles."""
    DEFAULT = SpanStyle()
    ERROR = SpanStyle(foreground="#ff6b6b")
    SUCCESS = SpanStyle(foreground="#9ece6a")
    INFO = SpanStyle(foreground="#7aa2f7")
    WARNING = SpanStyle(foreground="#e0af68")
    MUTED = SpanStyle(foreground="#6b7280")
    
    # Interactive
    LINK = SpanStyle(
        foreground="#7aa2f7", 
        underline=True,
        hover_foreground="#bb9af7",
        cursor="pointing"
    )
    COMMAND = SpanStyle(
        foreground="#9ece6a",
        hover_background="#1a1a2e",
        cursor="pointing"
    )
    ENTITY = SpanStyle(
        foreground="#bb9af7",
        bold=True,
        hover_foreground="#f7768e",
        cursor="pointing"
    )
    
    # Streaming
    STREAMING = SpanStyle(foreground="#c0caf5")
    LLM_RESPONSE = SpanStyle(foreground="#c0caf5")
    USER_INPUT = SpanStyle(foreground="#7aa2f7", bold=True)
    SYSTEM = SpanStyle(foreground="#565f89", italic=True)


# =============================================================================
# SPAN DEFINITION
# =============================================================================

@dataclass
class Span:
    """A styled, optionally interactive text region."""
    span_id: str
    start: int  # Character position in document
    end: int
    text: str
    style: SpanStyle
    
    # Interactivity
    interactive: bool = False
    on_click: Optional[Callable[[str, dict], None]] = None  # callback(span_id, data)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # State
    hovered: bool = False
    
    def contains(self, position: int) -> bool:
        """Check if position is within this span."""
        return self.start <= position < self.end
    
    def length(self) -> int:
        return self.end - self.start


# =============================================================================
# STREAM DEFINITION
# =============================================================================

@dataclass 
class TextStream:
    """An active text stream (e.g., LLM response)."""
    stream_id: str
    start_position: int
    style: SpanStyle
    chunks: List[str] = field(default_factory=list)
    complete: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_text(self) -> str:
        return "".join(self.chunks)
    
    def get_length(self) -> int:
        return sum(len(c) for c in self.chunks)


# =============================================================================
# DOCUMENT CLASS
# =============================================================================

class Document(QTextEdit):
    """
    General purpose rich text display with spans, streaming, and interactivity.
    
    Signals:
        span_clicked(span_id, data): Emitted when an interactive span is clicked
        span_hovered(span_id, data): Emitted when mouse enters a span
        stream_started(stream_id): Emitted when a new stream begins
        stream_complete(stream_id): Emitted when a stream finishes
        text_changed_debounced(): Emitted after text changes settle (debounced)
    """
    
    # Signals
    span_clicked = Signal(str, dict)      # span_id, span.data
    span_hovered = Signal(str, dict)      # span_id, span.data
    stream_started = Signal(str)          # stream_id
    stream_complete_signal = Signal(str)  # stream_id
    text_changed_debounced = Signal()
    
    def __init__(self, parent=None, read_only: bool = False):
        super().__init__(parent)
        
        # Configuration
        self._read_only = read_only
        self.setReadOnly(read_only)
        
        # Spans registry
        self.spans: Dict[str, Span] = {}
        self._span_positions: List[Tuple[int, int, str]] = []  # Sorted list for lookup
        
        # Streams
        self.streams: Dict[str, TextStream] = {}
        self._active_stream: Optional[str] = None
        
        # Hover state
        self._hovered_span_id: Optional[str] = None
        
        # Debounce timer for text changes
        self._change_timer = QTimer()
        self._change_timer.setSingleShot(True)
        self._change_timer.timeout.connect(self.text_changed_debounced.emit)
        
        # Setup
        self._setup_document()
        self._setup_events()
    
    def _setup_document(self):
        """Setup document appearance."""
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #d4d4d4;
                border: none;
                padding: 10px;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }
        """)
        
        font = QFont("Consolas", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        
        # Enable mouse tracking for hover detection
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
    
    def _setup_events(self):
        """Setup event connections."""
        self.textChanged.connect(self._on_text_changed)
    
    # -------------------------------------------------------------------------
    # TEXT OPERATIONS
    # -------------------------------------------------------------------------
    
    def append_text(self, text: str, style: SpanStyle = None) -> str:
        """
        Append text with optional style. Returns span_id.
        """
        style = style or Styles.DEFAULT
        
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        start_pos = cursor.position()
        
        # Apply style
        fmt = style.to_char_format()
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        
        end_pos = cursor.position()
        
        # Create span
        span_id = f"span_{uuid.uuid4().hex[:8]}"
        span = Span(
            span_id=span_id,
            start=start_pos,
            end=end_pos,
            text=text,
            style=style
        )
        self.spans[span_id] = span
        self._update_span_positions()
        
        # Scroll to end
        self.moveCursor(QTextCursor.MoveOperation.End)
        
        return span_id
    
    def append_styled(
        self, 
        text: str, 
        style: SpanStyle = None,
        interactive: bool = False,
        on_click: Callable[[str, dict], None] = None,
        data: Dict[str, Any] = None
    ) -> str:
        """
        Append text with style and optional interactivity.
        Returns span_id.
        """
        style = style or Styles.DEFAULT
        
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        start_pos = cursor.position()
        
        # Apply style
        fmt = style.to_char_format()
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        
        end_pos = cursor.position()
        
        # Create span
        span_id = f"span_{uuid.uuid4().hex[:8]}"
        span = Span(
            span_id=span_id,
            start=start_pos,
            end=end_pos,
            text=text,
            style=style,
            interactive=interactive or (on_click is not None),
            on_click=on_click,
            data=data or {}
        )
        self.spans[span_id] = span
        self._update_span_positions()
        
        self.moveCursor(QTextCursor.MoveOperation.End)
        
        return span_id
    
    def append_line(self, text: str = "", style: SpanStyle = None) -> str:
        """Append text followed by newline."""
        return self.append_text(text + "\n", style)
    
    def append_interactive(
        self,
        text: str,
        on_click: Callable[[str, dict], None],
        data: Dict[str, Any] = None,
        style: SpanStyle = None
    ) -> str:
        """Shorthand for appending interactive text."""
        style = style or Styles.LINK
        return self.append_styled(text, style, interactive=True, on_click=on_click, data=data)
    
    def insert_at(self, position: int, text: str, style: SpanStyle = None) -> str:
        """Insert text at specific position."""
        style = style or Styles.DEFAULT
        
        cursor = self.textCursor()
        cursor.setPosition(position)
        
        start_pos = cursor.position()
        
        fmt = style.to_char_format()
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        
        end_pos = cursor.position()
        
        # Create span
        span_id = f"span_{uuid.uuid4().hex[:8]}"
        span = Span(
            span_id=span_id,
            start=start_pos,
            end=end_pos,
            text=text,
            style=style
        )
        self.spans[span_id] = span
        
        # Shift all spans after this position
        offset = len(text)
        for sid, s in self.spans.items():
            if sid != span_id and s.start >= position:
                s.start += offset
                s.end += offset
        
        self._update_span_positions()
        
        return span_id
    
    # -------------------------------------------------------------------------
    # STREAMING OPERATIONS
    # -------------------------------------------------------------------------
    
    def stream_start(self, style: SpanStyle = None, stream_id: str = None) -> str:
        """
        Start a new text stream. Returns stream_id.
        Subsequent stream_chunk calls append to this stream.
        """
        style = style or Styles.STREAMING
        stream_id = stream_id or f"stream_{uuid.uuid4().hex[:8]}"
        
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        start_pos = cursor.position()
        
        stream = TextStream(
            stream_id=stream_id,
            start_position=start_pos,
            style=style
        )
        self.streams[stream_id] = stream
        self._active_stream = stream_id
        
        self.stream_started.emit(stream_id)
        
        return stream_id
    
    def stream_chunk(self, text: str, stream_id: str = None):
        """
        Append chunk to stream. Uses active stream if stream_id not specified.
        """
        stream_id = stream_id or self._active_stream
        if not stream_id or stream_id not in self.streams:
            # No active stream, just append normally
            self.append_text(text, Styles.STREAMING)
            return
        
        stream = self.streams[stream_id]
        if stream.complete:
            return
        
        stream.chunks.append(text)
        
        # Append to document
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        fmt = stream.style.to_char_format()
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        
        self.moveCursor(QTextCursor.MoveOperation.End)
    
    def stream_complete(self, stream_id: str = None) -> Optional[str]:
        """
        Mark stream as complete. Creates a span for the full streamed text.
        Returns span_id of the completed stream.
        """
        stream_id = stream_id or self._active_stream
        if not stream_id or stream_id not in self.streams:
            return None
        
        stream = self.streams[stream_id]
        stream.complete = True
        
        # Create span for the full streamed text
        full_text = stream.get_text()
        end_position = stream.start_position + len(full_text)
        
        span_id = f"span_{stream_id}"
        span = Span(
            span_id=span_id,
            start=stream.start_position,
            end=end_position,
            text=full_text,
            style=stream.style
        )
        self.spans[span_id] = span
        self._update_span_positions()
        
        if self._active_stream == stream_id:
            self._active_stream = None
        
        self.stream_complete_signal.emit(stream_id)
        
        return span_id
    
    # -------------------------------------------------------------------------
    # SPAN OPERATIONS
    # -------------------------------------------------------------------------
    
    def get_span_at(self, position: int) -> Optional[Span]:
        """Get span at document position."""
        for start, end, span_id in self._span_positions:
            if start <= position < end:
                return self.spans.get(span_id)
        return None
    
    def get_interactive_span_at(self, position: int) -> Optional[Span]:
        """Get interactive span at position."""
        span = self.get_span_at(position)
        if span and span.interactive:
            return span
        return None
    
    def remove_span(self, span_id: str):
        """Remove span from registry (doesn't remove text)."""
        if span_id in self.spans:
            del self.spans[span_id]
            self._update_span_positions()
    
    def clear_spans(self):
        """Clear all span registrations."""
        self.spans.clear()
        self._span_positions.clear()
    
    def _update_span_positions(self):
        """Rebuild sorted span position list for fast lookup."""
        self._span_positions = [
            (s.start, s.end, s.span_id)
            for s in self.spans.values()
        ]
        self._span_positions.sort(key=lambda x: x[0])
    
    # -------------------------------------------------------------------------
    # MOUSE EVENTS
    # -------------------------------------------------------------------------
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for interactive spans."""
        if event.button() == Qt.MouseButton.LeftButton:
            cursor = self.cursorForPosition(event.pos())
            position = cursor.position()
            
            span = self.get_interactive_span_at(position)
            if span:
                # Emit signal
                self.span_clicked.emit(span.span_id, span.data)
                
                # Call callback if set
                if span.on_click:
                    span.on_click(span.span_id, span.data)
                
                event.accept()
                return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for hover effects."""
        cursor = self.cursorForPosition(event.pos())
        position = cursor.position()
        
        span = self.get_interactive_span_at(position)
        
        if span:
            if self._hovered_span_id != span.span_id:
                # New span hovered
                self._leave_span(self._hovered_span_id)
                self._enter_span(span)
        else:
            if self._hovered_span_id:
                self._leave_span(self._hovered_span_id)
        
        super().mouseMoveEvent(event)
    
    def _enter_span(self, span: Span):
        """Handle entering a span hover."""
        span.hovered = True
        self._hovered_span_id = span.span_id
        
        # Change cursor
        if span.style.cursor == "pointing":
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Emit signal
        self.span_hovered.emit(span.span_id, span.data)
        
        # Apply hover style if defined
        if span.style.hover_foreground or span.style.hover_background:
            self._apply_hover_style(span)
    
    def _leave_span(self, span_id: Optional[str]):
        """Handle leaving a span hover."""
        if not span_id:
            return
        
        span = self.spans.get(span_id)
        if span:
            span.hovered = False
            self._restore_span_style(span)
        
        self._hovered_span_id = None
        self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
    
    def _apply_hover_style(self, span: Span):
        """Apply hover style to span."""
        cursor = self.textCursor()
        cursor.setPosition(span.start)
        cursor.setPosition(span.end, QTextCursor.MoveMode.KeepAnchor)
        
        fmt = span.style.to_char_format()
        if span.style.hover_foreground:
            fmt.setForeground(QColor(span.style.hover_foreground))
        if span.style.hover_background:
            fmt.setBackground(QColor(span.style.hover_background))
        
        cursor.setCharFormat(fmt)
    
    def _restore_span_style(self, span: Span):
        """Restore normal style after hover."""
        cursor = self.textCursor()
        cursor.setPosition(span.start)
        cursor.setPosition(span.end, QTextCursor.MoveMode.KeepAnchor)
        
        fmt = span.style.to_char_format()
        cursor.setCharFormat(fmt)
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def clear_document(self):
        """Clear all text and spans."""
        self.clear()
        self.spans.clear()
        self.streams.clear()
        self._span_positions.clear()
        self._active_stream = None
        self._hovered_span_id = None
    
    def get_text(self) -> str:
        """Get plain text content."""
        return self.toPlainText()
    
    def set_text(self, text: str, style: SpanStyle = None):
        """Set document text (replaces all content)."""
        self.clear_document()
        if text:
            self.append_text(text, style)
    
    def _on_text_changed(self):
        """Handle text changes with debounce."""
        self._change_timer.start(300)  # 300ms debounce
    
    # -------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # -------------------------------------------------------------------------
    
    def append_error(self, text: str) -> str:
        return self.append_line(text, Styles.ERROR)
    
    def append_success(self, text: str) -> str:
        return self.append_line(text, Styles.SUCCESS)
    
    def append_info(self, text: str) -> str:
        return self.append_line(text, Styles.INFO)
    
    def append_warning(self, text: str) -> str:
        return self.append_line(text, Styles.WARNING)
    
    def append_system(self, text: str) -> str:
        return self.append_line(text, Styles.SYSTEM)


# =============================================================================
# EDITABLE DOCUMENT (for text editing use cases)
# =============================================================================

class EditableDocument(Document):
    """
    Document variant that allows editing.
    Tracks modifications and provides undo/redo.
    """
    
    modified = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent, read_only=False)
        self._is_modified = False
        self.textChanged.connect(self._mark_modified)
    
    def _mark_modified(self):
        if not self._is_modified:
            self._is_modified = True
            self.modified.emit()
    
    def is_modified(self) -> bool:
        return self._is_modified
    
    def set_modified(self, value: bool):
        self._is_modified = value


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
    
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Document Test")
    window.resize(800, 600)
    
    central = QWidget()
    layout = QVBoxLayout(central)
    
    doc = Document()
    layout.addWidget(doc)
    
    # Test buttons
    def test_styled():
        doc.append_line("This is normal text")
        doc.append_line("This is an error", Styles.ERROR)
        doc.append_line("This is success", Styles.SUCCESS)
        doc.append_line("This is info", Styles.INFO)
    
    def test_interactive():
        def on_click(span_id, data):
            print(f"Clicked: {span_id}, data: {data}")
            doc.append_info(f"You clicked: {data.get('label', 'unknown')}")
        
        doc.append_text("Click here: ")
        doc.append_interactive("Action 1", on_click, {"label": "Action 1", "command": "/do_thing"})
        doc.append_text(" or ")
        doc.append_interactive("Action 2", on_click, {"label": "Action 2", "command": "/other"})
        doc.append_line()
    
    def test_streaming():
        stream_id = doc.stream_start(Styles.LLM_RESPONSE)
        
        chunks = ["Hello", ", ", "this", " ", "is", " ", "streaming", " ", "text", "!"]
        
        def add_chunk(idx=0):
            if idx < len(chunks):
                doc.stream_chunk(chunks[idx], stream_id)
                QTimer.singleShot(100, lambda: add_chunk(idx + 1))
            else:
                doc.stream_complete(stream_id)
                doc.append_line()
                doc.append_success("Stream complete!")
        
        add_chunk()
    
    btn1 = QPushButton("Test Styled Text")
    btn1.clicked.connect(test_styled)
    layout.addWidget(btn1)
    
    btn2 = QPushButton("Test Interactive")
    btn2.clicked.connect(test_interactive)
    layout.addWidget(btn2)
    
    btn3 = QPushButton("Test Streaming")
    btn3.clicked.connect(test_streaming)
    layout.addWidget(btn3)
    
    btn4 = QPushButton("Clear")
    btn4.clicked.connect(doc.clear_document)
    layout.addWidget(btn4)
    
    window.setCentralWidget(central)
    window.show()
    
    sys.exit(app.exec())
