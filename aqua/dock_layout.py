#!/usr/bin/env python3
"""
Dock Layout System

Provides:
- TabDockArea: A tabbed container that can hold any widget type
- DockLayout: Manages multiple TabDockAreas in a flexible layout
- Drag-drop tabs between areas
- Tab type registry for extensibility

No QSplitter sliders - clean programmatic layout with optional resize handles.
"""

from __future__ import annotations
import uuid
from typing import Dict, List, Optional, Callable, Any, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTabBar,
    QSizePolicy, QFrame, QStackedWidget, QApplication, QMainWindow,
    QLabel, QPushButton, QMenu
)
from PySide6.QtCore import (
    Qt, Signal, QPoint, QMimeData, QByteArray, QDataStream, QIODevice,
    QSize, QEvent, QTimer
)
from PySide6.QtGui import (
    QDrag, QPixmap, QPainter, QColor, QCursor, QMouseEvent, QDragEnterEvent,
    QDropEvent, QAction
)


# =============================================================================
# POSITION ENUM
# =============================================================================

class DockPosition(Enum):
    """Position in the dock layout."""
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()
    CENTER = auto()


# =============================================================================
# TAB INFO
# =============================================================================

@dataclass
class TabInfo:
    """Information about a tab."""
    tab_id: str
    tab_type: str
    title: str
    widget: QWidget
    closable: bool = True
    data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TAB TYPE REGISTRY
# =============================================================================

class TabTypeRegistry:
    """
    Registry for tab types.
    Register a factory function for each tab type to enable creation.
    """
    
    _instance: Optional[TabTypeRegistry] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factories: Dict[str, Callable[..., QWidget]] = {}
            cls._instance._icons: Dict[str, str] = {}
        return cls._instance
    
    @classmethod
    def register(cls, tab_type: str, factory: Callable[..., QWidget], icon: str = ""):
        """Register a tab type with its factory function."""
        instance = cls()
        instance._factories[tab_type] = factory
        if icon:
            instance._icons[tab_type] = icon
    
    @classmethod
    def create(cls, tab_type: str, **kwargs) -> Optional[QWidget]:
        """Create a widget of the given tab type."""
        instance = cls()
        factory = instance._factories.get(tab_type)
        if factory:
            return factory(**kwargs)
        return None
    
    @classmethod
    def get_types(cls) -> List[str]:
        """Get list of registered tab types."""
        return list(cls()._factories.keys())
    
    @classmethod
    def get_icon(cls, tab_type: str) -> str:
        """Get icon for tab type."""
        return cls()._icons.get(tab_type, "")


# =============================================================================
# DRAGGABLE TAB BAR
# =============================================================================

class DraggableTabBar(QTabBar):
    """
    Tab bar that supports drag-drop of tabs between TabDockAreas.
    """
    
    tab_drag_started = Signal(int, QPoint)  # index, global_pos
    tab_dropped_external = Signal(str, QPoint)  # tab_id, global_pos
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_start_pos: Optional[QPoint] = None
        self._dragging = False
        self.setAcceptDrops(True)
        self.setMovable(True)  # Allow reordering within same bar
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        
        if self._drag_start_pos is None:
            super().mouseMoveEvent(event)
            return
        
        # Check if we've moved enough to start drag
        distance = (event.pos() - self._drag_start_pos).manhattanLength()
        if distance < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return
        
        # Start drag
        index = self.tabAt(self._drag_start_pos)
        if index < 0:
            super().mouseMoveEvent(event)
            return
        
        self._dragging = True
        self.tab_drag_started.emit(index, event.globalPosition().toPoint())
        
        # Create drag object
        drag = QDrag(self)
        mime_data = QMimeData()
        
        # Store tab info in mime data
        parent_area = self.parent()
        if hasattr(parent_area, 'get_tab_info'):
            tab_info = parent_area.get_tab_info(index)
            if tab_info:
                data = QByteArray()
                stream = QDataStream(data, QIODevice.OpenModeFlag.WriteOnly)
                stream.writeQString(tab_info.tab_id)
                stream.writeQString(str(id(parent_area)))  # Source area ID
                mime_data.setData("application/x-dock-tab", data)
        
        drag.setMimeData(mime_data)
        
        # Create drag pixmap
        pixmap = QPixmap(self.tabRect(index).size())
        pixmap.fill(QColor(60, 60, 60, 200))
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, self.tabText(index))
        painter.end()
        drag.setPixmap(pixmap)
        
        # Execute drag
        result = drag.exec(Qt.DropAction.MoveAction)
        
        self._dragging = False
        self._drag_start_pos = None
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_start_pos = None
        self._dragging = False
        super().mouseReleaseEvent(event)


# =============================================================================
# TAB DOCK AREA
# =============================================================================

class TabDockArea(QWidget):
    """
    A tabbed container that can hold any widget type.
    Supports drag-drop of tabs to/from other TabDockAreas.
    """
    
    # Signals
    tab_added = Signal(str)      # tab_id
    tab_removed = Signal(str)    # tab_id
    tab_moved = Signal(str, str) # tab_id, to_area_id
    area_empty = Signal()        # Emitted when last tab is removed
    
    def __init__(
        self, 
        area_id: str = None,
        position: DockPosition = DockPosition.CENTER,
        parent=None
    ):
        super().__init__(parent)
        
        self.area_id = area_id or f"area_{uuid.uuid4().hex[:8]}"
        self.position = position
        
        # Tab storage
        self.tabs: Dict[str, TabInfo] = {}
        
        # Setup UI
        self._setup_ui()
        self._setup_drag_drop()
    
    def _setup_ui(self):
        """Setup the tab widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget with custom tab bar
        self.tab_widget = QTabWidget()
        self.tab_bar = DraggableTabBar(self)
        self.tab_widget.setTabBar(self.tab_bar)
        
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(False)  # We handle movement ourselves
        self.tab_widget.setDocumentMode(True)
        
        # Connect signals
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tab_bar.tab_drag_started.connect(self._on_tab_drag_started)
        
        layout.addWidget(self.tab_widget)
        
        # Style
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d30;
                color: #d4d4d4;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #007acc;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #3e3e42;
            }
        """)
    
    def _setup_drag_drop(self):
        """Setup drag-drop acceptance."""
        self.setAcceptDrops(True)
    
    # -------------------------------------------------------------------------
    # TAB MANAGEMENT
    # -------------------------------------------------------------------------
    
    def add_tab(
        self,
        widget: QWidget,
        title: str,
        tab_type: str = "generic",
        tab_id: str = None,
        closable: bool = True,
        data: Dict[str, Any] = None
    ) -> str:
        """
        Add a tab to this area.
        Returns tab_id.
        """
        tab_id = tab_id or f"tab_{uuid.uuid4().hex[:8]}"
        
        # Create tab info
        tab_info = TabInfo(
            tab_id=tab_id,
            tab_type=tab_type,
            title=title,
            widget=widget,
            closable=closable,
            data=data or {}
        )
        
        self.tabs[tab_id] = tab_info
        
        # Add to tab widget
        index = self.tab_widget.addTab(widget, title)
        
        # Store tab_id in widget for lookup
        widget.setProperty("tab_id", tab_id)
        
        # Set closable
        if not closable:
            self.tab_widget.tabBar().setTabButton(
                index, 
                QTabBar.ButtonPosition.RightSide, 
                None
            )
        
        self.tab_added.emit(tab_id)
        
        # Make it current
        self.tab_widget.setCurrentIndex(index)
        
        return tab_id
    
    def remove_tab(self, tab_id: str, emit_signal: bool = True) -> Optional[QWidget]:
        """
        Remove a tab by ID.
        Returns the widget (doesn't destroy it).
        """
        if tab_id not in self.tabs:
            return None
        
        tab_info = self.tabs[tab_id]
        widget = tab_info.widget
        
        # Find index
        index = self._get_tab_index(tab_id)
        if index >= 0:
            self.tab_widget.removeTab(index)
        
        del self.tabs[tab_id]
        
        if emit_signal:
            self.tab_removed.emit(tab_id)
        
        # Check if empty
        if len(self.tabs) == 0:
            self.area_empty.emit()
        
        return widget
    
    def get_tab_info(self, index: int) -> Optional[TabInfo]:
        """Get tab info by index."""
        if index < 0 or index >= self.tab_widget.count():
            return None
        
        widget = self.tab_widget.widget(index)
        tab_id = widget.property("tab_id")
        return self.tabs.get(tab_id)
    
    def get_tab_by_id(self, tab_id: str) -> Optional[TabInfo]:
        """Get tab info by ID."""
        return self.tabs.get(tab_id)
    
    def _get_tab_index(self, tab_id: str) -> int:
        """Get tab index by ID."""
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if widget.property("tab_id") == tab_id:
                return i
        return -1
    
    def set_current_tab(self, tab_id: str):
        """Set current tab by ID."""
        index = self._get_tab_index(tab_id)
        if index >= 0:
            self.tab_widget.setCurrentIndex(index)
    
    def current_tab_id(self) -> Optional[str]:
        """Get current tab ID."""
        index = self.tab_widget.currentIndex()
        if index >= 0:
            widget = self.tab_widget.widget(index)
            return widget.property("tab_id")
        return None
    
    def tab_count(self) -> int:
        """Get number of tabs."""
        return len(self.tabs)
    
    # -------------------------------------------------------------------------
    # DRAG AND DROP
    # -------------------------------------------------------------------------
    
    def _on_tab_drag_started(self, index: int, global_pos: QPoint):
        """Handle start of tab drag."""
        pass  # Drag is handled by DraggableTabBar
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept tab drops."""
        if event.mimeData().hasFormat("application/x-dock-tab"):
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle tab drop from another area."""
        if not event.mimeData().hasFormat("application/x-dock-tab"):
            event.ignore()
            return
        
        # Decode mime data
        data = event.mimeData().data("application/x-dock-tab")
        stream = QDataStream(data, QIODevice.OpenModeFlag.ReadOnly)
        tab_id = stream.readQString()
        source_area_id = stream.readQString()
        
        # Don't accept drops from self
        if source_area_id == str(id(self)):
            event.ignore()
            return
        
        # Find source area and move tab
        # This needs to go through the DockLayout
        dock_layout = self._find_dock_layout()
        if dock_layout:
            dock_layout.move_tab(tab_id, source_area_id, self.area_id)
        
        event.acceptProposedAction()
    
    def _find_dock_layout(self) -> Optional[DockLayout]:
        """Find parent DockLayout."""
        parent = self.parent()
        while parent:
            if isinstance(parent, DockLayout):
                return parent
            parent = parent.parent()
        return None
    
    # -------------------------------------------------------------------------
    # CLOSE HANDLING
    # -------------------------------------------------------------------------
    
    def _on_tab_close_requested(self, index: int):
        """Handle tab close request."""
        tab_info = self.get_tab_info(index)
        if tab_info and tab_info.closable:
            self.remove_tab(tab_info.tab_id)


# =============================================================================
# DOCK LAYOUT
# =============================================================================

class DockLayout(QWidget):
    """
    Manages multiple TabDockAreas in a flexible layout.
    
    Layout structure:
    ┌─────────────────────────────────────┐
    │              TOP                     │
    ├──────┬─────────────────┬────────────┤
    │      │                 │            │
    │ LEFT │     CENTER      │   RIGHT    │
    │      │                 │            │
    ├──────┴─────────────────┴────────────┤
    │             BOTTOM                   │
    └─────────────────────────────────────┘
    """
    
    # Signals
    layout_changed = Signal()
    tab_moved = Signal(str, str, str)  # tab_id, from_area, to_area
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Areas by position
        self.areas: Dict[DockPosition, TabDockArea] = {}
        self._area_by_id: Dict[str, TabDockArea] = {}
        
        # Setup layout
        self._setup_layout()
    
    def _setup_layout(self):
        """Setup the dock layout structure."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        
        # Top area
        self._top_container = QWidget()
        self._top_layout = QVBoxLayout(self._top_container)
        self._top_layout.setContentsMargins(0, 0, 0, 0)
        self._top_container.hide()
        main_layout.addWidget(self._top_container)
        
        # Middle row: Left | Center | Right
        middle_widget = QWidget()
        self._middle_layout = QHBoxLayout(middle_widget)
        self._middle_layout.setContentsMargins(0, 0, 0, 0)
        self._middle_layout.setSpacing(2)
        
        # Left area
        self._left_container = QWidget()
        self._left_layout = QVBoxLayout(self._left_container)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_container.hide()
        self._middle_layout.addWidget(self._left_container)
        
        # Center area (always visible, takes remaining space)
        self._center_container = QWidget()
        self._center_layout = QVBoxLayout(self._center_container)
        self._center_layout.setContentsMargins(0, 0, 0, 0)
        self._center_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self._middle_layout.addWidget(self._center_container, 1)
        
        # Right area
        self._right_container = QWidget()
        self._right_layout = QVBoxLayout(self._right_container)
        self._right_layout.setContentsMargins(0, 0, 0, 0)
        self._right_container.hide()
        self._middle_layout.addWidget(self._right_container)
        
        main_layout.addWidget(middle_widget, 1)
        
        # Bottom area
        self._bottom_container = QWidget()
        self._bottom_layout = QVBoxLayout(self._bottom_container)
        self._bottom_layout.setContentsMargins(0, 0, 0, 0)
        self._bottom_container.hide()
        main_layout.addWidget(self._bottom_container)
        
        # Style
        self.setStyleSheet("""
            DockLayout {
                background: #1e1e1e;
            }
        """)
    
    # -------------------------------------------------------------------------
    # AREA MANAGEMENT
    # -------------------------------------------------------------------------
    
    def get_or_create_area(self, position: DockPosition) -> TabDockArea:
        """Get existing area or create new one at position."""
        if position in self.areas:
            return self.areas[position]
        
        # Create new area
        area = TabDockArea(position=position)
        area.area_empty.connect(lambda: self._on_area_empty(position))
        
        self.areas[position] = area
        self._area_by_id[area.area_id] = area
        
        # Add to layout
        container, layout = self._get_container_for_position(position)
        layout.addWidget(area)
        container.show()
        
        self.layout_changed.emit()
        
        return area
    
    def get_area(self, position: DockPosition) -> Optional[TabDockArea]:
        """Get area at position (may be None)."""
        return self.areas.get(position)
    
    def get_area_by_id(self, area_id: str) -> Optional[TabDockArea]:
        """Get area by ID."""
        # Check by actual ID
        if area_id in self._area_by_id:
            return self._area_by_id[area_id]
        
        # Check by Python object ID (for drag-drop)
        for area in self.areas.values():
            if str(id(area)) == area_id:
                return area
        
        return None
    
    def _get_container_for_position(self, position: DockPosition) -> Tuple[QWidget, QVBoxLayout]:
        """Get container and layout for position."""
        mapping = {
            DockPosition.TOP: (self._top_container, self._top_layout),
            DockPosition.BOTTOM: (self._bottom_container, self._bottom_layout),
            DockPosition.LEFT: (self._left_container, self._left_layout),
            DockPosition.RIGHT: (self._right_container, self._right_layout),
            DockPosition.CENTER: (self._center_container, self._center_layout),
        }
        return mapping[position]
    
    def _on_area_empty(self, position: DockPosition):
        """Handle area becoming empty."""
        if position == DockPosition.CENTER:
            # Don't hide center
            return
        
        area = self.areas.get(position)
        if area and area.tab_count() == 0:
            container, _ = self._get_container_for_position(position)
            container.hide()
            self.layout_changed.emit()
    
    # -------------------------------------------------------------------------
    # TAB OPERATIONS
    # -------------------------------------------------------------------------
    
    def add_tab(
        self,
        widget: QWidget,
        title: str,
        position: DockPosition = DockPosition.CENTER,
        tab_type: str = "generic",
        **kwargs
    ) -> str:
        """Add a tab to the specified position."""
        area = self.get_or_create_area(position)
        return area.add_tab(widget, title, tab_type, **kwargs)
    
    def move_tab(self, tab_id: str, from_area_id: str, to_area_id: str):
        """Move tab from one area to another."""
        from_area = self.get_area_by_id(from_area_id)
        to_area = self.get_area_by_id(to_area_id)
        
        if not from_area or not to_area:
            return
        
        tab_info = from_area.get_tab_by_id(tab_id)
        if not tab_info:
            return
        
        # Remove from source (don't destroy widget)
        widget = from_area.remove_tab(tab_id, emit_signal=False)
        if not widget:
            return
        
        # Add to target
        to_area.add_tab(
            widget,
            tab_info.title,
            tab_info.tab_type,
            tab_id=tab_info.tab_id,
            closable=tab_info.closable,
            data=tab_info.data
        )
        
        self.tab_moved.emit(tab_id, from_area_id, to_area_id)
    
    def create_tab_from_type(
        self,
        tab_type: str,
        title: str = None,
        position: DockPosition = DockPosition.CENTER,
        **kwargs
    ) -> Optional[str]:
        """Create a new tab from registered type."""
        widget = TabTypeRegistry.create(tab_type, **kwargs)
        if not widget:
            return None
        
        title = title or tab_type.title()
        return self.add_tab(widget, title, position, tab_type)
    
    # -------------------------------------------------------------------------
    # LAYOUT PRESETS
    # -------------------------------------------------------------------------
    
    def set_sizes(
        self,
        left_width: int = 0,
        right_width: int = 0,
        top_height: int = 0,
        bottom_height: int = 0
    ):
        """Set sizes for dock areas."""
        if left_width > 0:
            self._left_container.setFixedWidth(left_width)
        else:
            self._left_container.setMaximumWidth(16777215)  # Reset
        
        if right_width > 0:
            self._right_container.setFixedWidth(right_width)
        else:
            self._right_container.setMaximumWidth(16777215)
        
        if top_height > 0:
            self._top_container.setFixedHeight(top_height)
        else:
            self._top_container.setMaximumHeight(16777215)
        
        if bottom_height > 0:
            self._bottom_container.setFixedHeight(bottom_height)
        else:
            self._bottom_container.setMaximumHeight(16777215)


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton
    
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyleSheet("""
        QMainWindow { background: #1e1e1e; }
        QWidget { color: #d4d4d4; }
    """)
    
    window = QMainWindow()
    window.setWindowTitle("Dock Layout Test")
    window.resize(1200, 800)
    
    # Create dock layout
    dock_layout = DockLayout()
    window.setCentralWidget(dock_layout)
    
    # Register some tab types
    TabTypeRegistry.register("editor", lambda: QTextEdit())
    TabTypeRegistry.register("button", lambda: QPushButton("Click me"))
    
    # Add tabs to different positions
    editor1 = QTextEdit()
    editor1.setPlainText("This is the main document area.\n\nDrag tabs between areas!")
    dock_layout.add_tab(editor1, "Document 1", DockPosition.CENTER, "editor")
    
    editor2 = QTextEdit()
    editor2.setPlainText("Second document")
    dock_layout.add_tab(editor2, "Document 2", DockPosition.CENTER, "editor")
    
    # Right panel
    assistant = QTextEdit()
    assistant.setPlainText("Assistant panel\n\nThis could be chat, help, etc.")
    assistant.setReadOnly(True)
    dock_layout.add_tab(assistant, "Assistant", DockPosition.RIGHT, "assistant")
    
    # Bottom panel
    console = QTextEdit()
    console.setPlainText("> Ready.\n> Type commands here...")
    dock_layout.add_tab(console, "Console", DockPosition.BOTTOM, "console")
    
    prompt = QTextEdit()
    prompt.setPlainText("> ")
    dock_layout.add_tab(prompt, "Prompt", DockPosition.BOTTOM, "prompt")
    
    # Set initial sizes
    dock_layout.set_sizes(right_width=300, bottom_height=200)
    
    window.show()
    
    print("Drag tabs between the different areas!")
    print(f"Registered tab types: {TabTypeRegistry.get_types()}")
    
    sys.exit(app.exec())
