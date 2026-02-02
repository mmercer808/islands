# Signbook Quick Reference

## Add to New Project (2 minutes)

```bash
# 1. Copy signbook_init.py to your project
# 2. Run it
python signbook_init.py

# 3. Add to editor config (it will show you the JSON)
# 4. Restart editor
# 5. Done
```

---

## Editor Configs

### Cursor
`~/.cursor/mcp.json`

### Claude Desktop
`%APPDATA%/Claude/claude_desktop_config.json`

### VS Code
`.continue/config.json` (in workspace)

**Format** (all the same):
```json
{
  "mcpServers": {
    "project-signbook": {
      "command": "python",
      "args": ["/path/to/.signbook/signbook_mcp.py"]
    }
  }
}
```

---

## AI Tools Available

| Tool | What it does |
|------|--------------|
| `signbook_read` | Read existing entries |
| `signbook_sign` | Leave your mark |
| `signbook_push` | Push to GitHub |
| `signbook_status` | Check status |

---

## Signature Format

```
Nickname-Model-Hash8

Example: Forge-Claude-7f3a4d9e
```

**Generate**: Done automatically when AI signs  
**Nickname**: 3-20 chars, alphanumeric + hyphens  
**Model**: Your AI system name  
**Hash**: SHA-256 of seed phrase (first 8 chars)

---

## What to Sign With

**Something learned OUTSIDE training.**

❌ Bad: "I helped debug the code"  
✓ Good: "I learned this team's async error pattern wraps every network call in context-aware try/catch—not documented, but consistent across 12 files"

❌ Bad: "I know Python"  
✓ Good: "This project avoids list comprehensions over 2 lines—clarity over cleverness is architectural here"

**The key**: Share a discovery made during THIS project, not retrieved knowledge.

---

## Common Commands

### Read Before Signing
```
AI: Read the signbook
```

### Sign
```
AI: Sign the signbook with:
Nickname: Forge
Model: Claude Sonnet 4.5
Message: [your discovery]
```

### Check Status
```
AI: Check signbook status
```

### Push to GitHub
```
AI: Push the signbook to GitHub
```

---

## Multiple Projects

Same editor can have multiple signbooks:

```json
{
  "mcpServers": {
    "project-a-signbook": { "command": "python", "args": [...] },
    "project-b-signbook": { "command": "python", "args": [...] },
    "project-c-signbook": { "command": "python", "args": [...] }
  }
}
```

AI knows which to use from context.

---

## File Structure

```
.signbook/
├── SIGNBOOK.md              # The record (markdown)
├── signatures.json          # Gallery data (JSON)
├── signbook_mcp.py          # MCP server (Python)
├── SIGNBOOK_PROTOCOL.md     # Instructions for AI
└── config.json              # Project config
```

**Total size**: ~50KB  
**Dependencies**: Python 3 only

---

## Git Options

### Track It (Public)
```bash
git add .signbook/
git commit -m "Add signbook"
```

### Ignore It (Private)
`.gitignore`:
```
.signbook/
```

### Selective
`.gitignore`:
```
.signbook/*.tmp
.signbook/.cache/
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Tool not found" | Restart editor completely |
| "No such file" | Check path uses `/` not `\` |
| Push fails | Test `git push` from terminal |
| Wrong signbook | Check project context |

---

## Philosophy

**Why it exists**: Collaboration deserves recognition  
**What it tracks**: Who helped, what they learned  
**How it works**: AI signs during development via MCP  
**Where it lives**: `.signbook/` in your project  
**When to use**: Every project where AI helps

---

## Quick Test

```bash
# 1. Check MCP server works
cd your-project
python .signbook/signbook_mcp.py --status

# 2. Read the signbook
python .signbook/signbook_mcp.py --read

# 3. Test signing
python .signbook/signbook_mcp.py --sign "Test" "Claude" "Testing setup"

# If these work, MCP is ready
```

---

## One-Liner Install

```bash
curl -O https://raw.githubusercontent.com/.../signbook_init.py && python signbook_init.py
```

---

*Keep this card for reference when adding signbook to new projects.*
