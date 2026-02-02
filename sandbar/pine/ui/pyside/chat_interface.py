"""
Pine Chat Interface
===================

LLM-connected chat interface.

Features:
- Streaming LLM output
- Layered text styles
- Theme system
- Settings persistence

SOURCE: proc_streamer_v1_6.py (36KB)
Copy the full implementation from that file.
"""

try:
    from PySide6.QtWidgets import QMainWindow, QWidget

    class ChatInterface(QMainWindow):
        """
        LLM chat interface.

        TODO: Copy full implementation from proc_streamer_v1_6.py
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Pine Chat")
            self.resize(800, 600)

        def set_provider(self, provider: str, model: str) -> None:
            """Set LLM provider and model."""
            pass

        async def send_message(self, message: str) -> str:
            """Send message and get response."""
            return ""

except ImportError:
    ChatInterface = None
