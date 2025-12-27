#!/usr/bin/env python3
"""
Simplified Word Processor - No Background Workers
Fixes:
1. Removed unnecessary EventQueue and workers
2. Simple direct signal emission 
3. Optional signal system (can run without it)
4. No infinite loops or background threads
5. Clean startup and shutdown
"""

import sys
import os
import json
import time
import uuid
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QToolBar, QPushButton, QLineEdit, QLabel, QSplitter,
    QStatusBar, QFileDialog, QMessageBox, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QDockWidget, QComboBox, QSpinBox, QCheckBox,
    QSlider, QProgressBar, QListWidget, QListWidgetItem, QGroupBox,
    QScrollArea, QTextBrowser, QFrame, QGridLayout, QSpacerItem,
    QSizePolicy
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, QThread, QObject, QPropertyAnimation, 
    QEasingCurve, QRect, QSize, QPoint, QMimeData, QUrl
)
from PySide6.QtGui import (
    QFont, QTextCharFormat, QColor, QTextCursor, QAction,
    QKeySequence, QPalette, QSyntaxHighlighter, QTextDocument,
    QTextBlockFormat, QTextListFormat, QBrush, QPen, QPixmap,
    QIcon, QFontMetrics, QTextOption, QDragEnterEvent, QDropEvent
)


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


@dataclass
class DocumentState:
    """Document state"""
    document_id: str
    file_path: Optional[str] = None
    title: str = "Untitled"
    content: str = ""
    modified: bool = False
    word_count: int = 0
    char_count: int = 0
    line_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# SIMPLIFIED TEXT EDITOR
# =============================================================================

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
# MAIN WINDOW
# =============================================================================

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


# =============================================================================
# DEBUGGING AND MONITORING
# =============================================================================

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
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Simple Word Processor")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("Simple Systems")
    
    # Create main window
    window = SimpleWordProcessor()
    window.show()
    
    # Create debug window (optional)
    debug_window = DebugWindow(window.signal_emitter)
    debug_window.show()
    
    print("🚀 Simple Word Processor started!")
    print("📡 Signal system active (no background workers)")
    print("🐛 Debug window available for signal monitoring")
    print("💡 Try: Create documents, type text, execute plugins")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()