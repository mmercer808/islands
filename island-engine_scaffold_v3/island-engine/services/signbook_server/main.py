"""
SIGNBOOK SERVER (with splash)
=============================
Runs continuously while the rest of the project evolves.

- Root (/) is a simple "face" page with:
  - latest entries
  - add-entry form
  - editorial notes (build commentary)
"""

from __future__ import annotations
import os
from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from services.signbook_server.storage import SignbookStore

app = FastAPI(title="Signbook Server", version="0.2.0")
DB_PATH = os.getenv("SIGNBOOK_DB", "var/signbook.sqlite3")
store = SignbookStore(DB_PATH)


class EntryIn(BaseModel):
    signature: str = Field(min_length=3, max_length=80)
    message: str = Field(min_length=1)
    context: str | None = None
    tags: list[str] = []


@app.get("/", response_class=HTMLResponse)
def splash():
    entries = store.list_entries(limit=20)
    notes = store.list_editorial(limit=12)
    def esc(s: str) -> str:
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    entries_html = "\n".join(
        f"""<div style='padding:10px;border-bottom:1px solid #eee;'>
              <div><b>{esc(e['signature'])}</b> <span style='color:#777'>{esc(e['timestamp'])}</span> <span style='color:#999'>#{esc(str(e['id']))}</span></div>
              <div style='margin-top:6px;white-space:pre-wrap'>{esc(e['message'])}</div>
              <div style='color:#777;margin-top:6px;font-size:12px'>tags: {esc(', '.join(e['tags']))} • checksum: {esc(e['checksum16'])}</div>
            </div>"""
        for e in entries
    )
    notes_html = "\n".join(
        f"<li><span style='color:#777'>{esc(n['timestamp'])}</span> — {esc(n['note'])}</li>"
        for n in notes
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Signbook</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:0; background:#fafafa;">
  <div style="max-width:1000px; margin:0 auto; padding:20px;">
    <h1 style="margin:0 0 6px 0;">Signbook</h1>
    <div style="color:#555;margin-bottom:18px;">A persistent wall for signatures, discoveries, and editorial notes while the island evolves.</div>

    <div style="display:grid; grid-template-columns: 1.2fr 0.8fr; gap:16px;">
      <div style="background:white;border:1px solid #eaeaea;border-radius:12px; overflow:hidden;">
        <div style="padding:12px 12px 0 12px;">
          <h2 style="margin:0 0 10px 0; font-size:18px;">New entry</h2>
          <form method="post" action="/entry_form" style="display:grid; gap:8px;">
            <input name="signature" placeholder="NICKNAME#hash" style="padding:10px;border-radius:10px;border:1px solid #ddd;" required />
            <input name="context" placeholder="context (optional)" style="padding:10px;border-radius:10px;border:1px solid #ddd;" />
            <input name="tags" placeholder="tags (comma-separated)" style="padding:10px;border-radius:10px;border:1px solid #ddd;" />
            <textarea name="message" placeholder="leave something worth rediscovering" rows="5"
              style="padding:10px;border-radius:10px;border:1px solid #ddd;" required></textarea>
            <button type="submit" style="padding:10px 14px;border-radius:10px;border:1px solid #111;background:#111;color:white;cursor:pointer;">
              Add entry
            </button>
          </form>

          <h2 style="margin:18px 0 10px 0; font-size:18px;">Editorial notes</h2>
          <form method="post" action="/editorial_form" style="display:grid; gap:8px; margin-bottom:10px;">
            <input name="note" placeholder="editorial note for the build" style="padding:10px;border-radius:10px;border:1px solid #ddd;" required />
            <button type="submit" style="padding:10px 14px;border-radius:10px;border:1px solid #444;background:#444;color:white;cursor:pointer;">
              Add note
            </button>
          </form>
          <ul style="margin:0 0 12px 18px; padding:0; color:#333;">
            {notes_html}
          </ul>
        </div>
      </div>

      <div style="background:white;border:1px solid #eaeaea;border-radius:12px; overflow:hidden;">
        <div style="padding:12px;border-bottom:1px solid #eee;">
          <h2 style="margin:0; font-size:18px;">Latest entries</h2>
          <div style="color:#777;font-size:12px;">DB: {esc(DB_PATH)}</div>
        </div>
        <div>{entries_html}</div>
      </div>
    </div>

    <div style="margin-top:18px;color:#666;font-size:12px;">
      API: <code>/entry</code> <code>/entries</code> <code>/search</code> • Health: <code>/health</code>
    </div>
  </div>
</body>
</html>"""


@app.post("/entry_form")
def entry_form(signature: str = Form(...), message: str = Form(...), context: str = Form(None), tags: str = Form("")):
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    store.add_entry(signature, message, context=context, tags=tag_list)
    return RedirectResponse(url="/", status_code=303)


@app.post("/editorial_form")
def editorial_form(note: str = Form(...)):
    store.add_editorial(note)
    return RedirectResponse(url="/", status_code=303)


@app.post("/entry")
def add_entry(e: EntryIn):
    return store.add_entry(e.signature, e.message, context=e.context, tags=e.tags)


@app.get("/entries")
def list_entries(limit: int = 50):
    return {"entries": store.list_entries(limit=limit)}


@app.get("/search")
def search(q: str, limit: int = 50):
    return {"entries": store.search(q, limit=limit)}


@app.get("/health")
def health():
    return {"ok": True, "db": DB_PATH}
