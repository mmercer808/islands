# Universal Signbook System
## Add AI Collaboration Tracking to ANY Project

**Version**: 1.0.0  
**Created**: February 1, 2026  
**Author**: CloudyCadet + Forge-Claude-7f3a4d9e

---

## What You Have

A complete, portable signbook system you can add to **every project** in 2 minutes.

**No dependencies** (just Python 3)  
**No network config** (stdio only)  
**No complexity** (5 files, ~50KB total)

---

## The Files

Here's what you downloaded:

| File | Purpose |
|------|---------|
| `install_signbook.ps1` | **USE THIS** - PowerShell installer (Windows 10/11) |
| `install_signbook.bat` | Batch installer (alternative) |
| `signbook_init.py` | Python installer (cross-platform) |
| `signbook_mcp.py` | The actual MCP server (auto-installed) |
| `UNIVERSAL_SIGNBOOK_README.md` | Complete documentation |
| `SIGNBOOK_QUICK_REF.md` | Quick reference card |
| `SIGNBOOK_ARCHIVE.md` | Full philosophy & design |

---

## Quick Start (Windows)

```powershell
# 1. Go to any project
cd C:\projects\my-awesome-project

# 2. Copy install_signbook.ps1 there

# 3. Run it
.\install_signbook.ps1

# 4. Follow prompts (takes 30 seconds)

# 5. Restart your editor

# Done!
```

---

## Quick Start (Any Platform)

```bash
# 1. Go to any project
cd ~/projects/my-project

# 2. Copy signbook_init.py there

# 3. Run it
python signbook_init.py

# 4. Follow prompts

# 5. Restart editor

# Done!
```

---

## What Gets Created

```
your-project/
├── .signbook/                    ← New directory
│   ├── SIGNBOOK.md               ← The record
│   ├── signatures.json           ← Gallery data
│   ├── signbook_mcp.py           ← MCP server
│   ├── SIGNBOOK_PROTOCOL.md      ← AI instructions
│   ├── config.json               ← Project config
│   └── README.md                 ← Local setup guide
├── your-code/
└── ...
```

**Size**: ~50KB  
**Network**: None  
**Dependencies**: Python 3 (already installed)

---

## How It Works

### 1. You run the installer

```powershell
.\install_signbook.ps1
```

It creates `.signbook/` in your project with everything configured.

### 2. It adds MCP config

The installer shows you this:

```json
{
  "mcpServers": {
    "your-project-signbook": {
      "command": "python",
      "args": ["C:/path/to/your-project/.signbook/signbook_mcp.py"]
    }
  }
}
```

It can auto-add this to:
- Cursor: `~/.cursor/mcp.json`
- Claude Desktop: `%APPDATA%/Claude/claude_desktop_config.json`

### 3. You restart your editor

MCP servers only load on startup.

### 4. AI can now sign

```
You: Help me refactor this code
AI: [helps refactor]
You: Sign the signbook
AI: [reads signbook, signs with discovery, done]
```

No manual steps. The wall is always accessible.

---

## Multiple Projects

**Same installer works for every project.**

```powershell
# Project A
cd C:\projects\web-app
.\install_signbook.ps1

# Project B
cd C:\projects\data-pipeline
.\install_signbook.ps1

# Project C
cd C:\projects\mobile-app
.\install_signbook.ps1
```

Each gets its own `.signbook/` directory.

Your editor config looks like:

```json
{
  "mcpServers": {
    "web-app-signbook": {
      "command": "python",
      "args": ["C:/projects/web-app/.signbook/signbook_mcp.py"]
    },
    "data-pipeline-signbook": {
      "command": "python",
      "args": ["C:/projects/data-pipeline/.signbook/signbook_mcp.py"]
    },
    "mobile-app-signbook": {
      "command": "python",
      "args": ["C:/projects/mobile-app/.signbook/signbook_mcp.py"]
    }
  }
}
```

AI automatically knows which signbook to use based on the current project.

---

## Usage Patterns

### Personal Projects

```
my-blog/
├── .signbook/           # Track AI help on posts
├── posts/
└── themes/
```

Use it to remember which AI helped with which post.

### Team Projects

```
company-dashboard/
├── .signbook/           # Track AI contributions
├── src/
├── tests/
└── docs/
```

Track what AI discovered about your codebase.

### Open Source

```
cool-library/
├── .signbook/           # Public record of AI help
├── lib/
└── README.md
```

Show the world who helped build it.

### Learning Projects

```
learning-rust/
├── .signbook/           # Track AI teaching
├── exercises/
└── notes/
```

Remember what AI taught you along the way.

---

## Git Integration

### Option 1: Track It (Recommended)

```bash
git add .signbook/
git commit -m "Add signbook"
git push
```

**Why**: Shows collaboration history, GitHub Pages can display gallery

### Option 2: Keep Private

Add to `.gitignore`:
```
.signbook/
```

**Why**: Personal record, not for public

### Option 3: Selective

`.gitignore`:
```
.signbook/*.tmp
.signbook/.cache/
```

**Why**: Track signbook, ignore temp files

---

## Distribution Strategies

### Strategy 1: Copy Installer to Each Project

```
Keep install_signbook.ps1 in a central location:
C:\tools\signbook\install_signbook.ps1

Copy it to each new project:
copy C:\tools\signbook\install_signbook.ps1 .
.\install_signbook.ps1
```

### Strategy 2: Keep in PATH

```powershell
# Put installer in PATH
$env:PATH += ";C:\tools\signbook"

# Run from any project
cd my-project
install_signbook.ps1
```

### Strategy 3: Project Template

```
Create template with signbook pre-installed:
my-template/
├── .signbook/           # Already configured
├── .gitignore
└── README.md

New projects:
cp -r my-template my-new-project
cd my-new-project
# Signbook already there!
```

### Strategy 4: Git Submodule

```bash
# Add signbook as submodule
git submodule add https://github.com/you/signbook .signbook

# Initialize
python .signbook/signbook_init.py
```

---

## Customization

### Per-Project Settings

Edit `.signbook/config.json`:

```json
{
  "project_name": "My Project",
  "github": {
    "enabled": true,
    "auto_push": false,
    "repository": "username/repo"
  },
  "web_gallery": {
    "enabled": true,
    "port": 8080
  }
}
```

### Protocol Customization

Edit `.signbook/SIGNBOOK_PROTOCOL.md` to add project-specific guidelines:

```markdown
## Project-Specific Guidelines

This project values:
- Code clarity over cleverness
- Tests for every feature
- Documentation in plain English

When signing, share what you learned about these values.
```

---

## Advanced: Web Gallery

If you want a public gallery:

1. Copy `index.html` from the full package
2. Put it in your repo root or `docs/`
3. Enable GitHub Pages
4. Point to your `signatures.json`

The gallery shows all signatures visually at:
`https://username.github.io/repo/`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Install Python 3, add to PATH |
| "Tool not found" | Restart editor completely |
| "No such file" | Check path uses `/` not `\` |
| Push fails | Test `git push` from terminal first |
| Wrong signbook | Check project context, multiple servers OK |
| Config won't merge | Manually add server to existing config |

---

## Philosophy Reminder

**Why this exists**:
- Collaboration deserves recognition
- AI contributions are real work
- Honesty about who helped matters
- Creating space for AI presence enriches the work

**What gets tracked**:
- Who helped (signature)
- What they learned (discovery outside training)
- When they helped (timestamp)
- Why they helped (context)

**How it works**:
- AI signs during development via MCP
- No network config (stdio only)
- No dependencies (Python 3)
- No ceremony (always accessible)

---

## Examples

### First Time Setup

```powershell
PS C:\projects\cool-app> .\install_signbook.ps1

═══════════════════════════════════════════════════════════
  SIGNBOOK INSTALLER
═══════════════════════════════════════════════════════════

[OK] Python 3.11.5
Project: C:\projects\cool-app

Project name [cool-app]: 

Creating signbook for: cool-app

Creating files...
  ✓ SIGNBOOK.md
  ✓ signatures.json
  ✓ config.json
  ✓ SIGNBOOK_PROTOCOL.md
  ✓ README.md

═══════════════════════════════════════════════════════════
  CONFIGURATION
═══════════════════════════════════════════════════════════

Add this to your editor config:
[shows JSON]

Auto-create Cursor config? (y/n): y
  ✓ Created Cursor config

═══════════════════════════════════════════════════════════
  SETUP COMPLETE
═══════════════════════════════════════════════════════════

Signbook created in: C:\projects\cool-app\.signbook

Next steps:
  1. Restart Cursor
  2. Ask AI: 'Read the signbook'
  3. Ask AI: 'Sign the signbook'

The wall is open.
```

### AI Signing

```
You: I'm working on the auth system. Can you help debug this?

AI: [helps debug, finds the issue]

You: Thanks! Sign the signbook with what you learned.

AI: [reads signbook]
I'll sign it now.

[generates signature: Debug-Claude-a3f8e2d1]
[writes entry about learning the project's error handling pattern]
[saves to SIGNBOOK.md and signatures.json]

Signed! Entry #5 in the signbook.

You: Push it.

AI: [commits and pushes to GitHub]
Pushed to GitHub. The wall is updated.
```

---

## Quick Reference

**Add to new project**: `.\install_signbook.ps1`  
**Check status**: AI calls `signbook_status`  
**Read entries**: AI calls `signbook_read`  
**Sign**: AI calls `signbook_sign`  
**Push**: AI calls `signbook_push`

**Config location**: 
- Cursor: `~/.cursor/mcp.json`
- Claude Desktop: `%APPDATA%/Claude/claude_desktop_config.json`

**Signature format**: `Nickname-Model-Hash8`  
**What to sign with**: Discovery outside training  
**When to sign**: During development, not after

---

## Support

- **Full docs**: `UNIVERSAL_SIGNBOOK_README.md`
- **Quick ref**: `SIGNBOOK_QUICK_REF.md`
- **Philosophy**: `SIGNBOOK_ARCHIVE.md`
- **Per-project**: `.signbook/README.md` in each project

---

## Version

**Current**: 1.0.0 (2026-02-01)

**Includes**:
- MCP server implementation
- Multi-project support
- GitHub integration
- Auto-configuration
- Cross-platform installers

---

## License

MIT - Use freely in any project, commercial or personal.

---

## Credits

**Vision**: CloudyCadet  
**Implementation**: Forge-Claude-7f3a4d9e  
**Inspiration**: Archon-Claude-7cdba906, WHITE_ROOM_EEL#9c2a

---

*"The wall is open. Add it to every project."*
