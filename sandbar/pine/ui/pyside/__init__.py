"""
Pine PySide6 UI Components
==========================

Desktop application components built with PySide6.

Components:
- StoryWorldEngine: OpenGL story world with entities
- ChatInterface: LLM-connected chat
- Console: Unified console application
- DockLayout: Dockable panel system

Source Files (to integrate):
- story_world_pyside6.py (52KB) -> story_world.py
- proc_streamer_v1_6.py (36KB) -> chat_interface.py
- unified_console_tab.py (44KB) -> console.py
- aqua/dock_layout.py -> dock_layout.py
"""

# Only import if PySide6 is available
try:
    from PySide6.QtWidgets import QApplication
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

if PYSIDE_AVAILABLE:
    from .story_world import StoryWorldEngine
    from .chat_interface import ChatInterface
    from .console import UnifiedConsole
    from .dock_layout import DockLayout

    __all__ = [
        'StoryWorldEngine',
        'ChatInterface',
        'UnifiedConsole',
        'DockLayout',
    ]
else:
    __all__ = []
