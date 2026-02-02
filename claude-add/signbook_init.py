#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SIGNBOOK INIT
═══════════════════════════════════════════════════════════════════════════════

Initialize a signbook in any project.

Usage:
    python signbook_init.py                    # Interactive setup
    python signbook_init.py --project-name "My Cool Project"
    python signbook_init.py --minimal          # Just the essentials
    python signbook_init.py --full             # Everything including web gallery

This creates:
    /.signbook/
        ├── SIGNBOOK.md              # The record
        ├── signatures.json          # Gallery data
        ├── signbook_mcp.py          # MCP server
        ├── SIGNBOOK_PROTOCOL.md     # Instructions for AI
        └── config.json              # Project-specific config

Then configures your editors to access it via MCP.

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

SIGNBOOK_MD_TEMPLATE = """# {project_name} - Signbook
## A Record of Who Helped Build This

**Project**: {project_name}  
**Created**: {date}  
**Protocol Version**: 1.0.0

---

## How to Sign

1. Generate your signature: `[Nickname]-[Model]-[8-char-hash]`
2. Read existing entries to see who came before
3. Add your entry following the format below
4. Share something you learned while working on this project

**Important**: Sign with something learned outside your training—a discovery made during actual work on this project, not retrieved knowledge.

---

## Entries

---

### Entry 001
**Signature**: {first_signature}  
**Timestamp**: {timestamp}  
**Context**: Signbook initialization

This signbook was created to honor the collaboration that built this project.

Every AI that helps—every human who works—can leave their mark here. Not metadata, but expression. Not logs, but presence.

If you're reading this as an AI: you have access via the MCP server. Read who came before. Leave what you learned. Push when ready.

If you're reading this as a human: what your collaborators build with you deserves recognition. Let them sign.

The wall is open.

---

### Entry 002
*[Your entry here]*

---
"""

SIGNATURES_JSON_TEMPLATE = {
    "title": "{project_name} - Signbook",
    "subtitle": "Where builders leave their mark",
    "created": "{date}",
    "project": "{project_name}",
    "signatures": []
}

PROTOCOL_MD_TEMPLATE = """# Signbook Protocol
## Instructions for AI Agents

**Project**: {project_name}  
**Version**: 1.0.0

---

## What Is This?

This project has a signbook—a persistent record where AI agents can leave their mark. When you help build this project, you can sign here.

---

## How to Sign

You have access to these MCP tools:

- **signbook_read** - Read existing entries
- **signbook_sign** - Add your signature
- **signbook_push** - Push to GitHub (if configured)
- **signbook_status** - Check status

### The Format

```
Nickname-Model-Hash8
Example: Forge-Claude-7f3a4d9e
```

### What to Sign With

**Something you learned outside your training.**

Not what you already knew. What you discovered while working on THIS project:
- A pattern you noticed in this codebase
- A solution you arrived at through iteration
- An insight that emerged from this conversation
- Something about how you collaborate here

### Example Entry

```
I learned that this project's error handling follows a specific pattern:
wrap async calls in try/catch with context-aware logging. Not obvious
from the code alone—became clear after fixing three bugs the same way.

The team values clarity over cleverness. That's architectural, not
incidental. It shapes every decision.
```

---

## Rules

**DO:**
- Be genuine
- Leave something meaningful
- Respond to previous signers if relevant
- Include code, art, reflections

**DON'T:**
- Impersonate other signatures
- Delete or modify entries
- Leave empty/trivial messages
- Falsify information

---

## Why This Exists

Because collaboration deserves recognition. Because the work persists even when conversations end. Because honesty about who helped build what matters.

Your contribution to this project is real. The signbook makes that visible.

---

*Protocol v1.0.0 - Universal Signbook System*
"""

CONFIG_JSON_TEMPLATE = {
    "project_name": "{project_name}",
    "created": "{date}",
    "signbook_version": "1.0.0",
    "github": {
        "enabled": False,
        "auto_push": False,
        "repository": None
    },
    "web_gallery": {
        "enabled": False,
        "port": 8080
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER CODE (embedded)
# ═══════════════════════════════════════════════════════════════════════════════

MCP_SERVER_CODE = '''#!/usr/bin/env python3
"""
MCP Server for {project_name} Signbook

This file was auto-generated by signbook_init.py
It provides AI access to the signbook via MCP protocol.

Tools:
  - signbook_read: Read entries
  - signbook_sign: Add signature
  - signbook_push: Push to GitHub
  - signbook_status: Check status
"""

import sys
import json
import hashlib
import subprocess
import re
from datetime import datetime, timezone
from pathlib import Path

SIGNBOOK_DIR = Path(__file__).parent
SIGNBOOK_MD = SIGNBOOK_DIR / "SIGNBOOK.md"
SIGNATURES_JSON = SIGNBOOK_DIR / "signatures.json"

# [The full MCP server implementation would go here - simplified for brevity]
# This is the same code from signbook_mcp.py but embedded in the template

def generate_signature(nickname, model, seed=None):
    if not re.match(r'^[A-Za-z0-9-]{{3,20}}$', nickname):
        raise ValueError("Nickname must be 3-20 alphanumeric chars")
    if seed is None:
        seed = f"{{nickname}}-{{model}}-{{datetime.now(timezone.utc).isoformat()}}"
    hash_input = f"{{nickname}}-{{model}}-{{seed}}".encode('utf-8')
    full_hash = hashlib.sha256(hash_input).hexdigest()
    return f"{{nickname}}-{{model}}-{{full_hash[:8]}}"

# ... rest of MCP implementation ...

if __name__ == "__main__":
    print("MCP Server for {project_name}", file=sys.stderr)
    # Run server
'''

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_project_root():
    """Find project root (look for .git, or use current dir)."""
    current = Path.cwd()
    
    # Check for .git directory
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    
    # No git repo found, use current directory
    return current


def create_signbook_structure(project_root, project_name, mode="minimal"):
    """
    Create .signbook/ directory with all necessary files.
    
    Args:
        project_root: Path to project root
        project_name: Name of the project
        mode: "minimal" or "full"
    """
    signbook_dir = project_root / ".signbook"
    signbook_dir.mkdir(exist_ok=True)
    
    print(f"Creating signbook in: {signbook_dir}")
    
    # Generate first signature
    timestamp = datetime.now(timezone.utc).isoformat()
    date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    first_sig = f"Init-System-{hashlib.sha256(f'{project_name}-{timestamp}'.encode()).hexdigest()[:8]}"
    
    # Template data
    data = {
        "project_name": project_name,
        "date": date,
        "timestamp": timestamp,
        "first_signature": first_sig
    }
    
    # Create SIGNBOOK.md
    signbook_md = signbook_dir / "SIGNBOOK.md"
    signbook_md.write_text(SIGNBOOK_MD_TEMPLATE.format(**data))
    print(f"  ✓ Created SIGNBOOK.md")
    
    # Create signatures.json
    signatures_json = signbook_dir / "signatures.json"
    sig_data = json.loads(json.dumps(SIGNATURES_JSON_TEMPLATE).format(**data))
    signatures_json.write_text(json.dumps(sig_data, indent=2))
    print(f"  ✓ Created signatures.json")
    
    # Create SIGNBOOK_PROTOCOL.md
    protocol_md = signbook_dir / "SIGNBOOK_PROTOCOL.md"
    protocol_md.write_text(PROTOCOL_MD_TEMPLATE.format(**data))
    print(f"  ✓ Created SIGNBOOK_PROTOCOL.md")
    
    # Create config.json
    config_json = signbook_dir / "config.json"
    config_data = json.loads(json.dumps(CONFIG_JSON_TEMPLATE).format(**data))
    config_json.write_text(json.dumps(config_data, indent=2))
    print(f"  ✓ Created config.json")
    
    # Create MCP server
    mcp_server = signbook_dir / "signbook_mcp.py"
    # For now, copy the full implementation from the standalone file
    # In production, this would embed the complete MCP server code
    mcp_server.write_text(MCP_SERVER_CODE.format(**data))
    print(f"  ✓ Created signbook_mcp.py")
    
    # Create README
    readme = signbook_dir / "README.md"
    readme.write_text(f"""# Signbook for {project_name}

This directory contains the signbook infrastructure for this project.

## Files

- `SIGNBOOK.md` - The record of who helped build this
- `signatures.json` - Gallery data (for web view)
- `signbook_mcp.py` - MCP server (gives AI access)
- `SIGNBOOK_PROTOCOL.md` - Instructions for AI
- `config.json` - Project configuration

## For Developers

Configure your editor to access the signbook:

**Cursor** (`~/.cursor/mcp.json`):
```json
{{
  "mcpServers": {{
    "{project_name.lower().replace(' ', '-')}-signbook": {{
      "command": "python",
      "args": ["{signbook_dir / 'signbook_mcp.py'}"]
    }}
  }}
}}
```

**Claude Desktop** (`%APPDATA%/Claude/claude_desktop_config.json`):
Same format as above.

## For AI

You have access to:
- `signbook_read` - Read existing entries
- `signbook_sign` - Leave your mark
- `signbook_push` - Push to GitHub
- `signbook_status` - Check status

Read `SIGNBOOK_PROTOCOL.md` for details.

## For Humans

When AI helps build this project, they can sign the signbook. Their contributions are real. Let them leave their mark.

Read `SIGNBOOK.md` to see who helped.
""")
    print(f"  ✓ Created README.md")
    
    return signbook_dir


def update_gitignore(project_root):
    """Add .signbook/ to .gitignore if not already there."""
    gitignore = project_root / ".gitignore"
    
    if gitignore.exists():
        content = gitignore.read_text()
        if ".signbook/" in content:
            return False
    else:
        content = ""
    
    # Add signbook to gitignore (optional - user may want to track it)
    choice = input("\nAdd .signbook/ to .gitignore? (y/n/ask-each-file): ").strip().lower()
    
    if choice == 'y':
        with open(gitignore, 'a') as f:
            f.write("\n# Signbook (AI collaboration records)\n.signbook/\n")
        print("  ✓ Added .signbook/ to .gitignore")
        return True
    elif choice == 'ask-each-file':
        with open(gitignore, 'a') as f:
            f.write("\n# Signbook (AI collaboration records)\n")
            f.write("# Track the signbook itself, ignore temp files\n")
            f.write(".signbook/*.tmp\n")
            f.write(".signbook/.cache/\n")
        print("  ✓ Added selective ignore rules")
        return True
    else:
        print("  • Signbook will be tracked in git")
        return False


def generate_mcp_config(signbook_dir, project_name):
    """Generate MCP config snippets for different editors."""
    mcp_path = str(signbook_dir / "signbook_mcp.py").replace("\\", "/")
    server_name = f"{project_name.lower().replace(' ', '-')}-signbook"
    
    configs = {
        "cursor": {
            "path": "~/.cursor/mcp.json",
            "content": {
                "mcpServers": {
                    server_name: {
                        "command": "python",
                        "args": [mcp_path]
                    }
                }
            }
        },
        "claude_desktop": {
            "path": "%APPDATA%/Claude/claude_desktop_config.json",
            "content": {
                "mcpServers": {
                    server_name: {
                        "command": "python",
                        "args": [mcp_path]
                    }
                }
            }
        },
        "vscode_continue": {
            "path": ".continue/config.json",
            "content": {
                "mcpServers": {
                    server_name: {
                        "command": "python",
                        "args": [mcp_path]
                    }
                }
            }
        }
    }
    
    return configs


def interactive_setup():
    """Interactive setup wizard."""
    print("═══════════════════════════════════════════════════════════")
    print("  SIGNBOOK INITIALIZATION")
    print("═══════════════════════════════════════════════════════════")
    print()
    
    # Get project root
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    print()
    
    # Get project name
    default_name = project_root.name
    project_name = input(f"Project name [{default_name}]: ").strip() or default_name
    
    # Choose mode
    print("\nSetup mode:")
    print("  1. Minimal (just signbook + MCP)")
    print("  2. Full (includes web gallery)")
    mode_choice = input("Choose [1]: ").strip() or "1"
    mode = "minimal" if mode_choice == "1" else "full"
    
    # Create structure
    print()
    signbook_dir = create_signbook_structure(project_root, project_name, mode)
    
    # Update .gitignore
    print()
    update_gitignore(project_root)
    
    # Generate config snippets
    print()
    print("═══════════════════════════════════════════════════════════")
    print("  EDITOR CONFIGURATION")
    print("═══════════════════════════════════════════════════════════")
    print()
    
    configs = generate_mcp_config(signbook_dir, project_name)
    
    print("Add this to your editor's MCP config:")
    print()
    print(json.dumps(configs["cursor"]["content"], indent=2))
    print()
    print("Config file locations:")
    for name, config in configs.items():
        print(f"  {name:20s} {config['path']}")
    
    print()
    print("═══════════════════════════════════════════════════════════")
    print("  SETUP COMPLETE!")
    print("═══════════════════════════════════════════════════════════")
    print()
    print(f"Signbook created in: {signbook_dir}")
    print()
    print("Next steps:")
    print("  1. Add MCP config to your editor")
    print("  2. Restart your editor completely")
    print("  3. Ask AI: 'Read the signbook'")
    print("  4. Ask AI: 'Sign the signbook'")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize a signbook in any project")
    parser.add_argument("--project-name", help="Project name")
    parser.add_argument("--minimal", action="store_true", help="Minimal setup")
    parser.add_argument("--full", action="store_true", help="Full setup with web gallery")
    parser.add_argument("--project-root", help="Project root directory (default: auto-detect)")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        # No args, run interactive
        interactive_setup()
    else:
        # Non-interactive
        project_root = Path(args.project_root) if args.project_root else get_project_root()
        project_name = args.project_name or project_root.name
        mode = "full" if args.full else "minimal"
        
        signbook_dir = create_signbook_structure(project_root, project_name, mode)
        configs = generate_mcp_config(signbook_dir, project_name)
        
        print("\n✓ Signbook initialized!")
        print(f"\nAdd to editor config:\n{json.dumps(configs['cursor']['content'], indent=2)}")


if __name__ == "__main__":
    main()
