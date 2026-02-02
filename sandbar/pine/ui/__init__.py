"""
Pine UI - Presentation Layer
============================

User interface components.

Submodules:
- pyside: PySide6 desktop applications
- web: Web-based interfaces

Source Files (to integrate):
- story_world_pyside6.py -> pyside/story_world.py
- proc_streamer_v1_6.py -> pyside/chat_interface.py
- unified_console_tab.py -> pyside/console.py
- aqua/ -> pyside/aqua.py
- room.html -> web/room.py
"""

# UI is optional - only import if PySide6 available
try:
    from .pyside import *
    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False

__all__ = ['HAS_PYSIDE']
