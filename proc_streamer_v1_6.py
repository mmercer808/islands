#!/usr/bin/env python3
"""
proc-streamer — v1.6

What’s new in v1.6
  • **Reliable Ollama connect**: health‑check on /api/tags, clear error messages, model presence check.
  • **Default URL = http://127.0.0.1:11434** (avoids IPv6/localhost edge cases).
  • **Auto‑connect on launch** (can be turned off by commenting one line).
  • Chat dock is ensured visible + raises on start; Assistant shows streaming output.
  • Keeps: layered styles (two buffers + blend + per‑layer opacity), themes, console cmds.

Run:
  python proc_streamer.py
"""
from __future__ import annotations
import sys, os, json, uuid, threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

APP_NAME = "proc-streamer"
VERSION = "1.6"
SETTINGS_FILE = Path("settings.json")

# ----------------------------------------------------------------------------
# Optional deps / integrations
# ----------------------------------------------------------------------------
try:
    import requests  # HTTP client for LLM + GitHub
except Exception:
    requests = None

# Optional terminal tab (ignored if missing)
try:
    from llm_terminal_complete import LLMTerminal
    HAVE_LLM_TERMINAL = True
except Exception:
    LLMTerminal = None
    HAVE_LLM_TERMINAL = False

# ----------------------------------------------------------------------------
# Qt imports
# ----------------------------------------------------------------------------
from PySide6.QtCore import Qt, Signal, QObject, QSize, QPoint
from PySide6.QtGui import (
    QAction, QFont, QTextCursor, QColor, QPainter, QPixmap, QPdfWriter, QPageSize
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QSplitter, QFileDialog, QStatusBar, QDockWidget, QLabel, QPushButton,
    QToolBar, QMessageBox, QLineEdit, QComboBox, QDoubleSpinBox, QTabWidget
)

# =============================================================================
# Theme system
# =============================================================================
@dataclass
class Theme:
    name: str = "dark"
    font_family: str = "Consolas"
    font_size: int = 14
    accent: str = "#00bcd4"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def dark() -> "Theme":
        return Theme(name="dark", font_family="Consolas", font_size=14, accent="#00bcd4")

    @staticmethod
    def light() -> "Theme":
        return Theme(name="light", font_family="Consolas", font_size=14, accent="#2962ff")


class ThemeManager:
    def __init__(self):
        self.theme: Theme = Theme.dark()
        self._data = {}
        self.load()

    def load(self):
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                self._data = data
                theme = data.get("theme", {})
                if theme:
                    self.theme = Theme(**theme)
            except Exception:
                self.theme = Theme.dark()

    def save(self):
        data = self._data or {}
        data["theme"] = self.theme.to_dict()
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))

    def apply_to_app(self, app: QApplication):
        t = self.theme
        base_bg = "#1e1e1e" if t.name == "dark" else "#f7f7f8"
        base_fg = "#e6e6e6" if t.name == "dark" else "#1b1b1f"
        panel_bg = "#0f1115" if t.name == "dark" else "#ffffff"
        tab_bg = "#2b2f36" if t.name == "dark" else "#e9eefb"
        border = "#3e3e42" if t.name == "dark" else "#cfd7ff"

        app.setStyleSheet(f"""
            * {{ font-family: {t.font_family}; font-size: {t.font_size}px; }}
            QMainWindow {{ background:{base_bg}; color:{base_fg}; }}
            QTextEdit, QLineEdit, QListWidget {{ background:{panel_bg}; color:{base_fg}; border:1px solid {border}; }}
            QDockWidget::title {{ background:{tab_bg}; color:{base_fg}; padding:6px; }}
            QPushButton {{ background:{tab_bg}; color:{base_fg}; border:1px solid {border}; padding:6px 10px; border-radius:8px; }}
            QPushButton:hover {{ border-color:{t.accent}; }}
            QStatusBar {{ background:{t.accent}; color:white; }}
        """)

# =============================================================================
# Settings
# =============================================================================
class GlobalSettings:
    def __init__(self):
        raw = {}
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text())
            except Exception:
                raw = {}
        self.llm = raw.get("llm", {
            "provider": "ollama",
            "url": "http://127.0.0.1:11434",   # avoid ::1/IPv6 mismatch
            "model": "llama3:latest",  # Fixed: use full model name with tag
            "api_key": "",
            "timeout": 60,
            "stream": True,
        })
        self.console = raw.get("console", {
            "history_size": 200,
            "prompt_symbol": "> ",
            "command_prefixes": ["$", "!", "cmd:"],
            "special_prefix": "/",
            "default_to_llm": True
        })
        self.ui = raw.get("ui", {
            "layer_enabled": True,
            "layer_blend": "Overlay",
            "layer_a_opacity": 0.85,
            "layer_b_opacity": 0.35,
            "chat_floating": False,
        })

    def persist(self, tm: ThemeManager):
        data = {
            "theme": tm.theme.to_dict(),
            "llm": self.llm,
            "console": self.console,
            "ui": self.ui,
        }
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))

# =============================================================================
# Snapshot model
# =============================================================================
@dataclass
class Snapshot:
    id: str
    created_at: str
    label: str
    content: str

class DocSession:
    def __init__(self):
        self.doc_id = str(uuid.uuid4())
        self.title = "Untitled"
        self.path: Optional[Path] = None
        self._text: str = ""
        self.snapshots: List[Snapshot] = []

    def set_text(self, text: str):
        self._text = text

    def text(self) -> str:
        return self._text

    def snapshot(self, label: str) -> str:
        sid = str(uuid.uuid4())
        snap = Snapshot(sid, datetime.now().isoformat(timespec="seconds"), label, self._text)
        self.snapshots.append(snap)
        return sid

# =============================================================================
# AssistChannel — streaming chat
# =============================================================================
class AssistChannel(QObject):
    chunk = Signal(str)
    complete = Signal(str)
    error = Signal(str)
    status = Signal(bool, str)

    def __init__(self, settings: GlobalSettings):
        super().__init__()
        self.s = settings
        self._connected = False
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    # ----- public API
    def connect(self, provider=None, url=None, model=None, api_key=None):
        if provider: self.s.llm["provider"] = provider
        if url: self.s.llm["url"] = url
        if model: self.s.llm["model"] = model
        if api_key: self.s.llm["api_key"] = api_key
        ok, msg = self._healthcheck()
        self._connected = ok
        self.status.emit(ok, msg)

    def disconnect(self):
        self._connected = False
        self.status.emit(False, "Disconnected")

    def query(self, prompt: str):
        if not self._connected:
            self.error.emit("Not connected")
            return
        if requests is None:
            self.error.emit("`requests` not installed")
            return
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run_query, args=(prompt,), daemon=True)
        self._thread.start()

    # ----- internals
    def _healthcheck(self) -> Tuple[bool, str]:
        if requests is None:
            return False, "Install `requests` to connect"
        provider = self.s.llm["provider"].lower()
        if provider == "ollama":
            try:
                url = self.s.llm["url"].rstrip("/") + "/api/tags"
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json() or {}
                models = [m.get("name") for m in data.get("models", [])]
                m = self.s.llm.get("model")
                if m and models:
                    # Check for exact match first
                    if m not in models:
                        # Check for partial match (e.g., "llama3" matches "llama3:latest")
                        partial_match = any(model.startswith(m + ":") or model == m for model in models)
                        if not partial_match:
                            return False, f"Ollama OK, but model '{m}' not found. Installed: {', '.join(models) or 'none'}. Try: `ollama pull {m}`."
                return True, f"Connected to Ollama @ {self.s.llm['url']} ({len(models)} models)"
            except Exception as e:
                return False, f"Ollama connect failed: {e}"
        else:
            try:
                url = self.s.llm["url"].rstrip("/") + "/v1/models"
                headers = {"Authorization": f"Bearer {self.s.llm['api_key']}"} if self.s.llm["api_key"] else {}
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                return True, f"Connected to OpenAI‑compatible @ {self.s.llm['url']}"
            except Exception as e:
                return False, f"OpenAI‑compat connect failed: {e}"

    def _run_query(self, prompt: str):
        provider = self.s.llm["provider"].lower()
        try:
            if provider == "ollama":
                self._query_ollama(prompt)
            else:
                self._query_openai_compat(prompt)
        except Exception as e:
            self.error.emit(str(e))

    def _query_ollama(self, prompt: str):
        url = self.s.llm["url"].rstrip("/") + "/api/generate"
        model = self.s.llm["model"]
        payload = {"model": model, "prompt": prompt, "stream": True}
        with requests.post(url, json=payload, stream=True, timeout=self.s.llm["timeout"]) as r:
            r.raise_for_status()
            full = []
            for line in r.iter_lines(decode_unicode=True):
                if self._stop_flag: break
                if not line: continue
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        tok = obj["response"]; full.append(tok); self.chunk.emit(tok)
                    if obj.get("done"): break
                except Exception:
                    full.append(line); self.chunk.emit(line)
            self.complete.emit("".join(full))

    def _query_openai_compat(self, prompt: str):
        url = self.s.llm["url"].rstrip("/") + "/v1/chat/completions"
        model = self.s.llm["model"]
        headers = {"Authorization": f"Bearer {self.s.llm['api_key']}"} if self.s.llm["api_key"] else {}
        data = {"model": model, "stream": True, "messages": [{"role": "user", "content": prompt}]}
        with requests.post(url, headers=headers, json=data, stream=True, timeout=self.s.llm["timeout"]) as r:
            r.raise_for_status()
            full = []
            for line in r.iter_lines(decode_unicode=True):
                if self._stop_flag: break
                if not line: continue
                if line.startswith("data: "):
                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]": break
                    try:
                        obj = json.loads(payload)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            full.append(delta); self.chunk.emit(delta)
                    except Exception:
                        pass
            self.complete.emit("".join(full))

# =============================================================================
# Layered styling engine (mixin)
# =============================================================================
COMPOSITION_MAP = {
    "Normal": QPainter.CompositionMode_SourceOver,
    "Multiply": QPainter.CompositionMode_Multiply,
    "Screen": QPainter.CompositionMode_Screen,
    "Overlay": QPainter.CompositionMode_Overlay,
    "Darken": QPainter.CompositionMode_Darken,
    "Lighten": QPainter.CompositionMode_Lighten,
    "Plus": QPainter.CompositionMode_Plus,
    "Difference": QPainter.CompositionMode_Difference,
}

class LayeredStyleMixin:
    def init_layered(self):
        self._layer_enabled: bool = True
        self._layer_blend: str = "Overlay"
        self._layer_a_opacity: float = 0.85
        self._layer_b_opacity: float = 0.35
        self._layer_style_a: str = ""
        self._layer_style_b: str = ""

    def set_layer_enabled(self, v: bool): self._layer_enabled = bool(v); self.update()
    def set_layer_blend(self, name: str):
        if name in COMPOSITION_MAP: self._layer_blend = name; self.update()
    def set_layer_opacity_a(self, val: float): self._layer_a_opacity = max(0.0, min(1.0, float(val))); self.update()
    def set_layer_opacity_b(self, val: float): self._layer_b_opacity = max(0.0, min(1.0, float(val))); self.update()
    def set_layer_styles(self, style_a: str, style_b: str): self._layer_style_a, self._layer_style_b = style_a, style_b; self.update()

    def _render_to_pixmap_with_style(self, style: str) -> QPixmap:
        orig = self.styleSheet()
        try:
            if style: self.setStyleSheet(style)
            pm = QPixmap(self.size()); pm.fill(Qt.transparent)
            painter = QPainter(pm)
            self.render(painter, targetOffset=QPoint(0, 0), sourceRegion=None)
            painter.end()
            return pm
        finally:
            self.setStyleSheet(orig)

    def paintEvent(self, ev):  # type: ignore[override]
        if not getattr(self, "_layer_enabled", False):
            return super().paintEvent(ev)
        if self.width() <= 0 or self.height() <= 0:
            return super().paintEvent(ev)
        pm_a = self._render_to_pixmap_with_style(self._layer_style_a)
        pm_b = self._render_to_pixmap_with_style(self._layer_style_b)
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setOpacity(self._layer_a_opacity); painter.drawPixmap(0, 0, pm_a)
        painter.setOpacity(self._layer_b_opacity)
        painter.setCompositionMode(COMPOSITION_MAP.get(self._layer_blend, QPainter.CompositionMode_SourceOver))
        painter.drawPixmap(0, 0, pm_b)
        painter.end()

# =============================================================================
# Console widget
# =============================================================================
class UnifiedConsoleTab(QTextEdit, LayeredStyleMixin):
    command_executed = Signal(str, dict)
    llm_query_sent = Signal(str)

    def __init__(self, settings: GlobalSettings, assist: AssistChannel):
        QTextEdit.__init__(self); LayeredStyleMixin.init_layered(self)
        self.settings = settings; self.assist = assist
        self.history: List[str] = []; self.hidx = 0
        self.prompt_text = self.settings.console["prompt_symbol"]; self.input_start = 0
        self.llm_connected = False
        self._connect_signals(); self._setup_console(); self._welcome(); self._prompt()
        self.set_layer_styles(
            "QTextEdit { background:#0b0c0f; color:#cfd8dc; border:none; selection-background-color:#274060; }",
            "QTextEdit { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0f1221, stop:1 #0a0d18); color:#e3f2fd; border:1px solid #223; }",
        )

    def _connect_signals(self):
        self.assist.status.connect(self._on_status)
        self.assist.chunk.connect(self._on_chunk)
        self.assist.complete.connect(self._on_complete)
        self.assist.error.connect(self._on_error)

    def _setup_console(self):
        font = QFont("Consolas", 14); font.setFixedPitch(True); self.setFont(font); self.setAcceptRichText(True)

    def _welcome(self):
        msg = (
            f"Unified Console\nLLM: {self.settings.llm['model']} @ {self.settings.llm['url']} | Connected: {self.llm_connected}\n\n"
            "Commands:\n  /help\n  /connect  /disconnect\n  /model M\n  /summary  /rewrite  /outline\n  /theme <dark|light>\n  /chat <message>\n  /snap\n  /debug\n  /clear\n\n"
            f"🔍 Debug: Model='{self.settings.llm['model']}', URL='{self.settings.llm['url']}'\n"
        )
        self._append(msg, "info")

    def _append(self, text: str, kind="output"):
        self.setTextColor(QColor({
            "error": "#ff6b6b", "info": "#7aa2f7", "success": "#9ece6a"
        }.get(kind, "#cfd8dc")))
        self.moveCursor(QTextCursor.End); self.insertPlainText(text + ("\n" if not text.endswith("\n") else "")); self.moveCursor(QTextCursor.End)

    def _prompt(self):
        self.setTextColor(QColor("#89ddff")); self.moveCursor(QTextCursor.End); self.insertPlainText(self.prompt_text); self.input_start = self.textCursor().position()

    def keyPressEvent(self, e):
        cur = self.textCursor()
        if cur.position() < self.input_start and e.key() not in (Qt.Key_End, Qt.Key_Down):
            cur.setPosition(self.input_start); self.setTextCursor(cur)
            if e.key() not in (Qt.Key_Up, Qt.Key_Down):
                return
        if e.key() == Qt.Key_Return and not e.modifiers():
            self._exec_current()
        elif e.key() == Qt.Key_Up:
            self._hist(-1)
        elif e.key() == Qt.Key_Down:
            self._hist(1)
        elif e.key() == Qt.Key_Backspace and cur.position() <= self.input_start:
            return
        elif e.key() == Qt.Key_Home:
            cur.setPosition(self.input_start); self.setTextCursor(cur)
        else:
            super().keyPressEvent(e)

    def _get_current_input(self) -> str:
        cur = self.textCursor(); cur.setPosition(self.input_start); cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor); return cur.selectedText()

    def _new_line(self):
        self.moveCursor(QTextCursor.End); self.insertPlainText("\n"); self.moveCursor(QTextCursor.End)

    def _hist(self, delta):
        if not self.history: return
        self.hidx = max(0, min(len(self.history), self.hidx + delta))
        cur = self.textCursor(); cur.setPosition(self.input_start); self.setTextCursor(cur)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor); cur.removeSelectedText()
        if self.hidx < len(self.history): self.insertPlainText(self.history[self.hidx])

    def _exec_current(self):
        text = self._get_current_input().strip(); self._new_line()
        if text and (not self.history or self.history[-1] != text): self.history.append(text); self.hidx = len(self.history)
        self._route(text)
        if not self.settings.console["default_to_llm"] or not self.llm_connected: self._prompt()

    def _is_special(self, t): return t.startswith(self.settings.console["special_prefix"])
    def _is_system(self, t):  return any(t.startswith(p) for p in self.settings.console["command_prefixes"])

    def _route(self, text: str):
        if not text: self._prompt(); return
        if self._is_special(text): self._run_special(text[1:].strip()); return
        elif self._is_system(text): self._run_shell(text.split(" ",1)[1] if " " in text else ""); return
        elif self.llm_connected and self.settings.console["default_to_llm"]: self._send_llm(text); return
        else: self._append("No LLM connected. Use /connect", "error")

    def _run_special(self, spec: str):
        parts = spec.split(); cmd = parts[0] if parts else ""; args = parts[1:]
        mw = self.parent().parent().parent()
        if cmd == "clear": self.clear(); self._prompt(); return
        if cmd == "help": self._welcome(); self._prompt(); return
        if cmd == "connect": self.assist.connect(); return
        if cmd == "disconnect": self.assist.disconnect(); self._prompt(); return
        if cmd == "model":
            if args: self.assist.connect(model=args[0])
            else: self._append("Usage: /model <name>", "info")
            self._prompt(); return
        if cmd in ("summary","rewrite","outline"):
            self.llm_query_sent.emit(cmd); mw.run_editor_tool(cmd); return
        if cmd == "theme" and args:
            mode = args[0].lower()
            if mode in ("dark","light"):
                mw.theme_mgr.theme = Theme.dark() if mode == "dark" else Theme.light(); mw.theme_mgr.save(); mw.apply_theme(); self._append(f"Theme set to {mode}", "success")
            else: self._append("Usage: /theme <dark|light>", "info")
            self._prompt(); return
        if cmd == "snap": mw._snapshot(); self._prompt(); return
        if cmd == "chat":
            msg = " ".join(args).strip()
            if msg: mw.assistant_append_user(msg)
            else: self._append("Usage: /chat <message>", "info")
            self._prompt(); return
        if cmd == "debug":
            self._append(f"🔍 Debug Info:", "info")
            self._append(f"  Model: '{self.settings.llm['model']}'", "info")
            self._append(f"  URL: '{self.settings.llm['url']}'", "info")
            self._append(f"  Provider: '{self.settings.llm['provider']}'", "info")
            self._append(f"  Connected: {self.llm_connected}", "info")
            self._append(f"  Requests available: {requests is not None}", "info")
            self._prompt(); return
        self._append(f"Unknown command: /{cmd}", "error"); self._prompt()

    def _run_shell(self, command: str):
        import subprocess
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            if res.stdout: self._append(res.stdout, "output")
            if res.stderr: self._append(res.stderr, "error")
            self.command_executed.emit(command, {"rc": res.returncode})
        except Exception as e:
            self._append(str(e), "error")

    def _send_llm(self, text: str):
        self._append(f"{self.settings.llm['model']}: ", "info"); self.llm_query_sent.emit(text); self.assist.query(text)

    def _on_status(self, ok: bool, msg: str):
        self.llm_connected = ok; self._append(("[OK] " if ok else "[X] ") + msg, "success" if ok else "error")

    def _on_chunk(self, tok: str):
        self.setTextColor(QColor("#c0caf5")); self.insertPlainText(tok); self.moveCursor(QTextCursor.End)

    def _on_complete(self, text: str):
        self.insertPlainText("\n"); self._prompt()

    def _on_error(self, msg: str):
        self._append(msg, "error"); self._prompt()

# =============================================================================
# Editor & Assistant widgets (layered)
# =============================================================================
class SimpleTextEditor(QTextEdit, LayeredStyleMixin):
    def __init__(self):
        QTextEdit.__init__(self); LayeredStyleMixin.init_layered(self)
        style_a = (
            "QTextEdit {"
            " background: #1e1e1e; color:#d4d4d4; border:1px solid #3e3e42; padding:15px;"
            " selection-background-color:#264f78; selection-color:#ffffff;"
            " font-family:'Consolas','Courier New',monospace; font-size:14px; }"
        )
        style_b = (
            "QTextEdit {"
            " background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #23262e, stop:1 #1a1d25);"
            " color:#e8f0ff; border:1px solid #475; padding:15px; }"
        )
        self.set_layer_styles(style_a, style_b)
        font = QFont("Consolas", 14); font.setStyleHint(QFont.Monospace); self.setFont(font)

class AssistantPanel(QWidget, LayeredStyleMixin):
    request_llm = Signal(str)
    apply_suggestion = Signal(str)

    def __init__(self):
        QWidget.__init__(self); LayeredStyleMixin.init_layered(self)
        lay = QVBoxLayout(self)
        self.header = QLabel("Assistant"); lay.addWidget(self.header)
        self.view = QTextEdit(); self.view.setReadOnly(True); lay.addWidget(self.view, 1)
        row = QHBoxLayout(); self.input = QLineEdit(); self.input.setPlaceholderText("Ask… (Enter to send)"); self.send = QPushButton("Send")
        row.addWidget(self.input, 1); row.addWidget(self.send); lay.addLayout(row)
        self.apply_btn = QPushButton("Insert into editor"); lay.addWidget(self.apply_btn)
        self.send.clicked.connect(self._send); self.input.returnPressed.connect(self._send); self.apply_btn.clicked.connect(self._apply)
        self.set_layer_styles(
            "QWidget { background:#0e0e10; } QTextEdit { background:#0e0e10; color:#dcdcdc; border:1px solid #333; }",
            "QWidget { background:#0b0f14; } QTextEdit { background:#0b0f14; color:#e5f1ff; border:1px solid #1f3142; }",
        )

    def _send(self):
        t = self.input.text().strip()
        if t:
            self.view.append(f"<span style='color:#7aa2f7'>You:</span> {t}")
            self.request_llm.emit(t); self.input.clear()

    def _apply(self):
        text = self.view.toPlainText().splitlines()[-1] if self.view.toPlainText() else ""
        if text: self.apply_suggestion.emit(text)

    def append_chunk(self, s: str):
        self.view.setTextColor(QColor("#c0caf5")); self.view.moveCursor(QTextCursor.End); self.view.insertPlainText(s); self.view.moveCursor(QTextCursor.End)
    def append_complete(self): self.view.insertPlainText("\n")
    def append_error(self, msg: str): self.view.append(f"<span style='color:#ff6b6b'>Error:</span> {msg}")

# =============================================================================
# Main Window
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = GlobalSettings(); self.theme_mgr = ThemeManager(); self.session = DocSession()
        self.setWindowTitle(f"{APP_NAME} {VERSION}"); self.resize(1180, 760)

        # central editor
        self.editor = SimpleTextEditor()

        # console + channel
        self.assist = AssistChannel(self.settings)
        self.console = UnifiedConsoleTab(self.settings, self.assist)

        # assistant panel
        self.assistant = AssistantPanel()
        self.assistant.request_llm.connect(self.assist.query)
        self.assist.chunk.connect(self.assistant.append_chunk)
        self.assist.complete.connect(lambda _: self.assistant.append_complete())
        self.assist.error.connect(self.assistant.append_error)
        self.assistant.apply_suggestion.connect(self._insert_into_editor)

        # layout: editor over console
        v = QSplitter(Qt.Vertical); v.addWidget(self.editor); v.addWidget(self.console); v.setSizes([520, 220])
        central = QWidget(); cl = QVBoxLayout(central); cl.addWidget(v, 1); self.setCentralWidget(central)

        # dock: Assistant (+ optional terminal)
        self.chat_tabs = QTabWidget(); self.chat_tabs.addTab(self.assistant, "Assistant")
        if HAVE_LLM_TERMINAL:
            try:
                term = LLMTerminal(None, None, parent=self)
                self.chat_tabs.addTab(term, "LLM Terminal")
            except Exception:
                pass
        self.chat_dock = QDockWidget("Chat", self); self.chat_dock.setWidget(self.chat_tabs)
        self.chat_dock.setMinimumWidth(320)
        self.chat_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.chat_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.chat_dock)
        self.chat_dock.show(); self.chat_dock.raise_(); self.chat_tabs.setCurrentWidget(self.assistant)

        self._make_toolbar(); self._make_statusbar(); self.apply_theme()

        # Auto-connect to Ollama (comment out if you prefer manual)
        self.assist.connect()

    # toolbar & status ---------------------------------------------------------
    def _make_toolbar(self):
        tb = QToolBar("Main", self); tb.setIconSize(QSize(18, 18)); self.addToolBar(tb)
        act_new = QAction("New", self); act_new.triggered.connect(self._new_doc)
        act_open = QAction("Open", self); act_open.triggered.connect(self._open_doc)
        act_save = QAction("Save", self); act_save.triggered.connect(self._save_doc)
        act_export = QAction("Export…", self); act_export.triggered.connect(self._export_doc)
        act_snap = QAction("Snapshot", self); act_snap.triggered.connect(self._snapshot)
        act_connect = QAction("Connect", self); act_connect.triggered.connect(lambda: self.assist.connect())
        act_disc = QAction("Disconnect", self); act_disc.triggered.connect(self.assist.disconnect)
        tb.addActions([act_new, act_open, act_save, act_export, act_snap]); tb.addSeparator(); tb.addActions([act_connect, act_disc]); tb.addSeparator()

        self.theme_toggle = QAction("Theme", self); self.theme_toggle.triggered.connect(self.toggle_theme); tb.addAction(self.theme_toggle)
        tb.addWidget(QLabel("  Accent:")); self.accent_combo = QComboBox(); self.accent_combo.addItems(["#00bcd4", "#ff4081", "#2962ff", "#00e676", "#ffa000"])
        self.accent_combo.currentTextChanged.connect(self.change_accent); tb.addWidget(self.accent_combo); tb.addSeparator()

        tb.addWidget(QLabel("Layered:")); self.layer_toggle = QAction("On/Off", self); self.layer_toggle.setCheckable(True)
        self.layer_toggle.setChecked(True)
        self.layer_toggle.toggled.connect(self._toggle_layered); tb.addAction(self.layer_toggle)
        tb.addWidget(QLabel(" Blend:")); self.blend_combo = QComboBox(); self.blend_combo.addItems(list(COMPOSITION_MAP.keys()))
        self.blend_combo.setCurrentText("Overlay"); self.blend_combo.currentTextChanged.connect(self._blend_changed); tb.addWidget(self.blend_combo)
        tb.addWidget(QLabel(" A:")); self.alpha_a = QDoubleSpinBox(); self.alpha_a.setRange(0.0,1.0); self.alpha_a.setSingleStep(0.05); self.alpha_a.setValue(0.85); self.alpha_a.valueChanged.connect(self._alpha_changed); tb.addWidget(self.alpha_a)
        tb.addWidget(QLabel(" B:")); self.alpha_b = QDoubleSpinBox(); self.alpha_b.setRange(0.0,1.0); self.alpha_b.setSingleStep(0.05); self.alpha_b.setValue(0.35); self.alpha_b.valueChanged.connect(self._alpha_changed); tb.addWidget(self.alpha_b)

    def _make_statusbar(self):
        sb = QStatusBar(); self.setStatusBar(sb)
        self.sb_msg = QLabel("Ready"); self.sb_wc = QLabel("Words: 0")
        sb.addPermanentWidget(self.sb_wc); sb.addWidget(self.sb_msg)
        self.editor.textChanged.connect(self._update_counts); self._update_counts()

    # theme -------------------------------------------------------------------
    def apply_theme(self):
        self.theme_mgr.apply_to_app(QApplication.instance())
        f = QFont(self.theme_mgr.theme.font_family, self.theme_mgr.theme.font_size); f.setStyleHint(QFont.Monospace)
        self.editor.setFont(f); self.console.setFont(f); self.assistant.view.setFont(f)
        self.settings.persist(self.theme_mgr)

    def toggle_theme(self):
        self.theme_mgr.theme = Theme.light() if self.theme_mgr.theme.name == "dark" else Theme.dark(); self.theme_mgr.save(); self.apply_theme(); self.console._append(f"Theme set to {self.theme_mgr.theme.name}", "success")

    def change_accent(self, color: str):
        t = self.theme_mgr.theme; t.accent = color; self.theme_mgr.theme = t; self.theme_mgr.save(); self.apply_theme()

    # chat dock & layered UI --------------------------------------------------
    def _toggle_layered(self, on: bool):
        for w in (self.editor, self.console, self.assistant):
            if isinstance(w, LayeredStyleMixin): w.set_layer_enabled(on)

    def _blend_changed(self, name: str):
        for w in (self.editor, self.console, self.assistant):
            if isinstance(w, LayeredStyleMixin): w.set_layer_blend(name)

    def _alpha_changed(self, _):
        a = float(self.alpha_a.value()); b = float(self.alpha_b.value())
        for w in (self.editor, self.console, self.assistant):
            if isinstance(w, LayeredStyleMixin): w.set_layer_opacity_a(a); w.set_layer_opacity_b(b)

    # file ops ----------------------------------------------------------------
    def _new_doc(self):
        self.editor.clear(); self.session = DocSession(); self.sb_msg.setText("New document")

    def _open_doc(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Text (*.txt);;Markdown (*.md);;All (*.*)")
        if not path: return
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        self.editor.setPlainText(txt); self.session.path = Path(path); self.session.title = Path(path).name
        self.sb_msg.setText(f"Opened {self.session.title}")

    def _save_doc(self):
        if not self.session.path:
            path, _ = QFileDialog.getSaveFileName(self, "Save As", self.session.title + ".txt", "Text (*.txt);;Markdown (*.md)")
            if not path: return
            self.session.path = Path(path); self.session.title = Path(path).name
        Path(self.session.path).write_text(self.editor.toPlainText(), encoding="utf-8")
        self.sb_msg.setText(f"Saved {self.session.title}")

    def _export_doc(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", self.session.title, "DOCX (*.docx);;PDF (*.pdf)")
        if not path: return
        if path.lower().endswith(".docx"):
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument()
                for line in self.editor.toPlainText().splitlines(): doc.add_paragraph(line)
                doc.save(path); self.sb_msg.setText(f"Exported DOCX: {Path(path).name}")
            except ImportError:
                QMessageBox.warning(self, "Missing dependency", "Install python-docx to export DOCX")
        elif path.lower().endswith(".pdf"):
            writer = QPdfWriter(path); writer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))
            doc = self.editor.document(); doc.print_(writer); self.sb_msg.setText(f"Exported PDF: {Path(path).name}")
        else:
            Path(path).write_text(self.editor.toPlainText(), encoding="utf-8"); self.sb_msg.setText(f"Exported: {Path(path).name}")

    # snapshots ---------------------------------------------------------------
    def _snapshot_to_disk(self) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S"); Path("snapshots").mkdir(exist_ok=True)
        name = f"snapshot-{ts}.txt"; (Path("snapshots")/name).write_text(self.editor.toPlainText(), encoding="utf-8"); return name

    def _snapshot(self):
        self.session.set_text(self.editor.toPlainText()); sid = self.session.snapshot("manual"); fname = self._snapshot_to_disk(); short = sid.split("-")[0]
        self.sb_msg.setText(f"Snapshot: {short}  •  saved ↳ snapshots/{fname}")

    # assistant helpers -------------------------------------------------------
    def assistant_append_user(self, msg: str):
        self.assistant.view.append(f"<span style='color:#7aa2f7'>You:</span> {msg}"); self.assist.query(msg)

    def _insert_into_editor(self, text: str):
        cur = self.editor.textCursor();
        if cur.hasSelection(): cur.removeSelectedText()
        cur.insertText(text); self.editor.setTextCursor(cur); self._update_counts()

    def _update_counts(self):
        words = len(self.editor.toPlainText().split()); self.sb_wc.setText(f"Words: {words}")

# =============================================================================
# Entry
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(f"{APP_NAME} {VERSION}")
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
