"""
Pine Dock Layout
================

Dockable panel system.

Features:
- Drag-drop tabs
- Configurable layouts
- State persistence

SOURCE: aqua/dock_layout.py
Copy the full implementation from that file.
"""

try:
    from PySide6.QtWidgets import QMainWindow, QDockWidget

    class DockLayout(QMainWindow):
        """
        Dock layout manager.

        TODO: Copy full implementation from aqua/dock_layout.py
        """

        def __init__(self, parent=None):
            super().__init__(parent)

        def add_dock(self, name: str, widget, area=None) -> QDockWidget:
            """Add a dockable widget."""
            dock = QDockWidget(name, self)
            dock.setWidget(widget)
            self.addDockWidget(area or 1, dock)  # Qt.LeftDockWidgetArea
            return dock

except ImportError:
    DockLayout = None
