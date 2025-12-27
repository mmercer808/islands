#!/usr/bin/env python3
"""
Proc Streamer (Legacy UI) - novelist-friendly single-file app

- Keeps the *old* interface & layout you preferred:
  * Central text editor with the classic dark theme & monospace font
  * Bottom console (Unified Console Tab) for commands and LLM talk
  * Right dock "Assistant" panel for streamed responses + quick actions
  * Minimal toolbar (New/Open/Save/Export/Snapshot/Assistant actions)
- Adds a tiny AssistChannel that can stream from:
  * Ollama (default, http://localhost:11434, /api/generate)
  * OpenAI-compatible (LM Studio etc., /v1/chat/completions, stream=true)
- Document: single doc model with snapshots
- Core authoring ops exposed via console commands: /summary /rewrite /outline
- Minimal inline "balloon" prototype: change suggestions appear as inline anchors; clicking them opens assistant panel with Accept / Reject

Requires: PySide6 (UI). Optional: requests (for HTTP), python-docx (for DOCX export).

Run:
  python proc_streamer_legacy_ui.py
"""

import sys, os, json, time, uuid, threading, queue
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

# --- Optional dependencies
try:
    import requests
except Exception:
    requests = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

# --- Qt imports
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QAction, QFont, QTextCursor, QKeySequence, QColor, QTextCharFormat, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QSplitter, QFileDialog, QStatusBar, QDockWidget, QLabel, QPushButton,
    QToolBar, QListWidget, QListWidgetItem, QMessageBox
)

# =============================================================================
# Global settings (mirrors your Unified Console Tab pattern)
# =============================================================================

class GlobalSettings:
    def __init__(self):
        self.llm = {
            "provider": "ollama",                    # 'ollama' or 'openai'
            "url": "http://localhost:11434",        # ollama default
            "model": "llama3",                       # adjust to your local tags
            "api_key": "",                           # for OpenAI-compatible
            "timeout": 60,
            "stream": True,
        }
        self.console = {
            "history_size": 200,
            "prompt_symbol": "> ",
            "command_prefixes": ["$", "!", "cmd:"],
            "special_prefix": "/",
            "default_to_llm": True
        }
        self.ui = {
            "theme": "dark",
            "font_family": "Consolas",
            "font_size": 14
        }

# =============================================================================
# Simple signals & snapshot model (from your Simple Word Processor spirit)
# =============================================================================

class SimpleSignalEmitter(QObject):
    signal_emitted = Signal(str, str, dict)  # signal_type, source, data
    def __init__(self):
        super().__init__()
        self.count = 0
    def emit(self, t, src, data=None):
        self.count += 1
        self.signal_emitted.emit(t, src, data or {})

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
# AssistChannel - minimal, threadsafe streaming client for Ollama & OpenAI-like
# =============================================================================

class AssistChannel(QObject):
    chunk = Signal(str)        # streamed text chunk
    complete = Signal(str)     # full text
    error = Signal(str)        # error message
    status = Signal(bool, str) # connected?, message

    def __init__(self, settings: GlobalSettings):
        super().__init__()
        self.s = settings
        self._connected = False
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    # --- public API
    def connect(self, provider=None, url=None, model=None, api_key=None):
        if provider: self.s.llm["provider"] = provider
        if url: self.s.llm["url"] = url
        if model: self.s.llm["model"] = model
        if api_key: self.s.llm["api_key"] = api_key
        self._connected = True
        self.status.emit(True, f"Connected to {self.s.llm['provider']} @ {self.s.llm['url']} as {self.s.llm['model']}")

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
        # run in background thread (non-blocking UI)
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run_query, args=(prompt,), daemon=True)
        self._thread.start()

    # --- internals
    def _run_query(self, prompt: str):
        provider = self.s.llm["provider"]
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
                        tok = obj["response"]
                        full.append(tok)
                        self.chunk.emit(tok)
                    if obj.get("done"):
                        break
                except Exception:
                    # Some ollama builds send stray lines; treat as plain text
                    full.append(line)
                    self.chunk.emit(line)
            self.complete.emit("".join(full))

    def _query_openai_compat(self, prompt: str):
        url = self.s.llm["url"].rstrip("/") + "/v1/chat/completions"
        model = self.s.llm["model"]
        headers = {"Authorization": f"Bearer {self.s.llm['api_key']}"} if self.s.llm["api_key"] else {}
        data = {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}]
        }
        with requests.post(url, headers=headers, json=data, stream=True, timeout=self.s.llm["timeout"]) as r:
            r.raise_for_status()
            full = []
            for line in r.iter_lines(decode_unicode=True):
                if self._stop_flag: break
                if not line: continue
                if line.startswith("data: "):
                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            full.append(delta)
                            self.chunk.emit(delta)
                    except Exception:
                        pass
            self.complete.emit("".join(full))

# =============================================================================
# Console (keeps your old feel)
# =============================================================================

class UnifiedConsoleTab(QTextEdit):
    command_executed = Signal(str, dict)
    llm_query_sent = Signal(str)

    def __init__(self, settings: GlobalSettings, assist: AssistChannel):
        super().__init__()
        self.settings = settings
        self.assist = assist
        self.history: List[str] = []
        self.hidx = 0
        self.prompt_text = self.settings.console["prompt_symbol"]
        self.input_start = 0
        self.llm_connected = False
        self._connect_signals()
        self._setup_console()
        self._welcome()
        self._prompt()

    def _connect_signals(self):
        self.assist.status.connect(self._on_status)
        self.assist.chunk.connect(self._on_chunk)
        self.assist.complete.connect(self._on_complete)
        self.assist.error.connect(self._on_error)

    def _setup_console(self):
        font = QFont(self.settings.ui["font_family"], self.settings.ui["font_size"])
        font.setFixedPitch(True)
        self.setFont(font)
        self.setAcceptRichText(True)
        # dark terminal look
        self.setStyleSheet("""
            QTextEdit { background:#0c0c0c; color:#cccccc; border:none; selection-background-color:#264f78; }
        """)

    def _welcome(self):
        msg = f"""Unified Console - legacy style
LLM: {self.settings.llm['model']} @ {self.settings.llm['url']} | Connected: {self.llm_connected}

Commands:
  /help        Show help
  /connect     Connect to LLM
  /disconnect  Disconnect LLM
  /model M     Switch model
  /summary     Summarize editor (uses selection if any)
  /rewrite     Rewrite selection (or entire doc)
  /outline     Outline the doc
  /clear       Clear console

Routing:
  Regular text → LLM (when connected)
  $ or ! or cmd: → System command (shell)
"""
        self._append(msg, "info")

    # --- core I/O helpers
    def _append(self, text: str, kind="output"):
        if kind == "error":
            self.setTextColor(QColor("#ff6b6b"))
        elif kind == "info":
            self.setTextColor(QColor("#7aa2f7"))
        elif kind == "success":
            self.setTextColor(QColor("#9ece6a"))
        else:
            self.setTextColor(QColor("#cccccc"))
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(text + ("\n" if not text.endswith("\n") else ""))
        self.moveCursor(QTextCursor.End)

    def _prompt(self):
        self.setTextColor(QColor("#89ddff"))
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(self.prompt_text)
        self.input_start = self.textCursor().position()

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
        cur = self.textCursor()
        cur.setPosition(self.input_start)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        return cur.selectedText()

    def _new_line(self):
        self.moveCursor(QTextCursor.End)
        self.insertPlainText("\n")
        self.moveCursor(QTextCursor.End)

    def _hist(self, delta):
        if not self.history: return
        self.hidx = max(0, min(len(self.history), self.hidx + delta))
        # clear current input
        cur = self.textCursor()
        cur.setPosition(self.input_start); self.setTextCursor(cur)
        cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        if self.hidx < len(self.history):
            self.insertPlainText(self.history[self.hidx])

    def _exec_current(self):
        text = self._get_current_input().strip()
        self._new_line()
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text); self.hidx = len(self.history)
        self._route(text)
        if not self.settings.console["default_to_llm"] or not self.llm_connected:
            self._prompt()  # when LLM streaming, prompt gets added on completion

    # --- routing
    def _is_special(self, t): return t.startswith(self.settings.console["special_prefix"])
    def _is_system(self, t):  return any(t.startswith(p) for p in self.settings.console["command_prefixes"])

    def _route(self, text: str):
        if not text:
            self._prompt(); return

        if self._is_special(text):
            self._run_special(text[1:].strip())
        elif self._is_system(text):
            self._run_shell(text.split(" ",1)[1] if " " in text else "")
        elif self.llm_connected and self.settings.console["default_to_llm"]:
            self._send_llm(text)
        else:
            self._append("No LLM connected. Use /connect", "error")

    # --- special commands
    def _run_special(self, spec: str):
        parts = spec.split()
        cmd = parts[0] if parts else ""
        args = parts[1:]
        if cmd == "clear":
            self.clear(); self._prompt(); return
        if cmd == "help":
            self._welcome(); self._prompt(); return
        if cmd == "connect":
            self.assist.connect()  # use defaults from settings
            return
        if cmd == "disconnect":
            self.assist.disconnect(); self._prompt(); return
        if cmd == "model":
            if args:
                self.assist.connect(model=args[0])
            else:
                self._append("Usage: /model <name>", "info")
            self._prompt(); return
        if cmd in ("summary","rewrite","outline"):
            self.llm_query_sent.emit(cmd)
            # emit a placeholder; MainWindow will collect text + build prompt
            self.parent().parent().parent().run_editor_tool(cmd)
            return

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
        self._append(f"{self.settings.llm['model']}: ", "info")
        self.llm_query_sent.emit(text)
        self.assist.query(text)

    # --- AssistChannel callbacks
    def _on_status(self, ok: bool, msg: str):
        self.llm_connected = ok
        self._append(("[OK] " if ok else "[X] ") + msg, "success" if ok else "error")

    def _on_chunk(self, tok: str):
        self.setTextColor(QColor("#c0caf5"))
        self.insertPlainText(tok)
        self.moveCursor(QTextCursor.End)

    def _on_complete(self, text: str):
        self.insertPlainText("\n")
        self._prompt()  # new prompt when streaming completes

    def _on_error(self, msg: str):
        self._append(msg, "error")
        self._prompt()

# =============================================================================
# Editor widget (preserves your old aesthetic & behavior)
# =============================================================================

class SimpleTextEditor(QTextEdit):
    selection_changed = Signal(int, int)

    def __init__(self):
        super().__init__()
        self._setup()

    def _setup(self):
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; color: #d4d4d4;
                border: 1px solid #3e3e42; padding: 15px;
                selection-background-color: #264f78; selection-color: #ffffff;
                font-family: 'Consolas','Courier New', monospace; font-size: 14px;
            }
            QTextEdit:focus { border-color: #007acc; }
        """)
        font = QFont("Consolas", 14)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.textChanged.connect(self._on_change)
        self.selectionChanged.connect(self._on_sel)

    def _on_change(self):
        # debounced-ish snapshot hinting could happen here
        pass

    def _on_sel(self):
        c = self.textCursor()
        self.selection_changed.emit(c.selectionStart(), c.selectionEnd())

# =============================================================================
# Assistant Dock (right side) - streamed output + quick actions
# =============================================================================

class AssistantPanel(QWidget):
    request_llm = Signal(str)      # prompt string
    apply_suggestion = Signal(str) # content to insert

    def __init__(self):
        super().__init__()
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.setStyleSheet("QTextEdit { background:#0e0e0e; color:#dcdcdc; border:1px solid #333; }")
        self.btn_apply = QPushButton("Insert into editor")
        self.btn_apply.clicked.connect(self._apply_clicked)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Assistant"))
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.input = QTextEdit(); self.input.setFixedHeight(80)
        row.addWidget(self.input, 1)
        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self._send_clicked)
        row.addWidget(self.btn_send, 0)
        lay.addLayout(row)
        lay.addWidget(self.btn_apply)

    def _send_clicked(self):
        prompt = self.input.toPlainText().strip()
        if prompt:
            self.view.append(f"<span style='color:#7aa2f7'>You:</span> {prompt}")
            self.request_llm.emit(prompt)

    def _apply_clicked(self):
        text = self._last_plain()
        if text:
            self.apply_suggestion.emit(text)

    def append_chunk(self, text: str):
        self.view.setTextColor(QColor("#c0caf5"))
        self.view.moveCursor(QTextCursor.End)
        self.view.insertPlainText(text)
        self.view.moveCursor(QTextCursor.End)

    def append_complete(self):
        self.view.insertPlainText("\n")

    def append_error(self, msg: str):
        self.view.append(f"<span style='color:#ff6b6b'>Error:</span> {msg}")

    def _last_plain(self) -> str:
        return self.view.toPlainText().splitlines()[-1] if self.view.toPlainText() else ""

# =============================================================================
# Main Window (legacy layout)
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = GlobalSettings()
        self.session = DocSession()
        self.signals = SimpleSignalEmitter()

        self.setWindowTitle("Proc Streamer (Legacy UI)")
        self.resize(1100, 720)

        # central editor
        self.editor = SimpleTextEditor()

        # bottom console
        self.assist = AssistChannel(self.settings)
        self.console = UnifiedConsoleTab(self.settings, self.assist)

        # right dock assistant
        self.assistant = AssistantPanel()

        # wire assistant panel to AssistChannel
        self.assistant.request_llm.connect(self.assist.query)
        self.assist.chunk.connect(self.assistant.append_chunk)
        self.assist.complete.connect(lambda _: self.assistant.append_complete())
        self.assist.error.connect(self.assistant.append_error)
        self.assistant.apply_suggestion.connect(self._insert_into_editor)

        # assemble splitter (editor over console, legacy vibe)
        v = QSplitter(Qt.Vertical)
        v.addWidget(self.editor)
        v.addWidget(self.console)
        v.setSizes([500, 220])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(v, 1)
        self.setCentralWidget(central)

        # dock
        dock = QDockWidget("Assistant", self)
        dock.setWidget(self.assistant)
        dock.setMinimumWidth(320)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self._make_toolbar()
        self._make_statusbar()

    # ------------ toolbar & status
    def _make_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        act_new = QAction("New", self); act_new.triggered.connect(self._new_doc)
        act_open = QAction("Open", self); act_open.triggered.connect(self._open_doc)
        act_save = QAction("Save", self); act_save.triggered.connect(self._save_doc)
        act_export = QAction("Export…", self); act_export.triggered.connect(self._export_doc)
        act_snap = QAction("Snapshot", self); act_snap.triggered.connect(self._snapshot)
        act_connect = QAction("Connect", self); act_connect.triggered.connect(lambda: self.assist.connect())
        act_disc = QAction("Disconnect", self); act_disc.triggered.connect(self.assist.disconnect)

        tb.addActions([act_new, act_open, act_save, act_export, act_snap])
        tb.addSeparator()
        tb.addActions([act_connect, act_disc])

    def _make_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.sb_msg = QLabel("Ready")
        self.sb_wc = QLabel("Words: 0")
        sb.addPermanentWidget(self.sb_wc)
        sb.addWidget(self.sb_msg)
        # update word count live
        self.editor.textChanged.connect(self._update_counts)
        self._update_counts()

    # ------------ file ops
    def _new_doc(self):
        self.editor.clear()
        self.session = DocSession()
        self.sb_msg.setText("New document")

    def _open_doc(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Text (*.txt);;Markdown (*.md);;All (*.*)")
        if not path: return
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        self.editor.setPlainText(txt)
        self.session.path = Path(path); self.session.title = Path(path).name
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
            if DocxDocument is None:
                QMessageBox.warning(self, "Missing dependency", "Install python-docx to export DOCX")
                return
            doc = DocxDocument()
            for line in self.editor.toPlainText().splitlines():
                doc.add_paragraph(line)
            doc.save(path)
            self.sb_msg.setText(f"Exported DOCX: {Path(path).name}")
        elif path.lower().endswith(".pdf"):
            # simple Qt PDF export
            from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter
            writer = QPdfWriter(path)
            writer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))
            doc = self.editor.document()
            doc.print_(writer)
            self.sb_msg.setText(f"Exported PDF: {Path(path).name}")
        else:
            Path(path).write_text(self.editor.toPlainText(), encoding="utf-8")
            self.sb_msg.setText(f"Exported: {Path(path).name}")

    def _snapshot(self):
        self.session.set_text(self.editor.toPlainText())
        sid = self.session.snapshot("manual")
        short = sid.split("-")[0]
        self.sb_msg.setText(f"Snapshot: {short}")

    def _insert_into_editor(self, text: str):
        cur = self.editor.textCursor()
        if cur.hasSelection():
            cur.removeSelectedText()
        cur.insertText(text)
        self.editor.setTextCursor(cur)
        self._update_counts()

    def _update_counts(self):
        words = len(self.editor.toPlainText().split())
        self.sb_wc.setText(f"Words: {words}")

    # ------------ LLM helper (used by /summary /rewrite /outline shortcuts)
    def run_editor_tool(self, tool: str):
        c = self.editor.textCursor()
        text = c.selectedText() if c.hasSelection() else self.editor.toPlainText()

        if tool == "summary":
            prompt = f"Summarize the following text for a novelist's working notes:\n\n{text[:12000]}"
        elif tool == "rewrite":
            prompt = f"Rewrite the following passage with clearer prose while preserving voice and meaning:\n\n{text[:12000]}"
        elif tool == "outline":
            prompt = f"Produce a hierarchical outline of the following text, using #, ##, - bullets as needed:\n\n{text[:12000]}"
        else:
            return

        # stream via assistant channel so both console & panel get updates
        self.assistant.view.append(f"<span style='color:#7aa2f7'>Tool:</span> {tool}")
        self.assist.query(prompt)

# =============================================================================
# Main
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Proc Streamer (Legacy UI)")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
