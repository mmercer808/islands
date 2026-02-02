# ═══════════════════════════════════════════════════════════════════════════
# SIGNBOOK INSTALLER (PowerShell)
# ═══════════════════════════════════════════════════════════════════════════
#
# Adds signbook to any project in one command.
#
# Usage:
#   .\install_signbook.ps1                    # Interactive
#   .\install_signbook.ps1 -ProjectName "My Project"
#
# ═══════════════════════════════════════════════════════════════════════════

param(
    [string]$ProjectName = "",
    [switch]$SkipConfig
)

$ErrorActionPreference = "Stop"

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SIGNBOOK INSTALLER" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    Write-Host "Install Python 3 from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Get project directory
$projectDir = Get-Location
Write-Host "Project: $projectDir" -ForegroundColor Gray
Write-Host ""

# Get project name
if (-not $ProjectName) {
    $defaultName = Split-Path -Leaf $projectDir
    $input = Read-Host "Project name [$defaultName]"
    $ProjectName = if ($input) { $input } else { $defaultName }
}

Write-Host "Creating signbook for: $ProjectName" -ForegroundColor White
Write-Host ""

# Create .signbook directory
$signbookDir = Join-Path $projectDir ".signbook"
if (-not (Test-Path $signbookDir)) {
    New-Item -ItemType Directory -Path $signbookDir | Out-Null
}

# Generate timestamp
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$date = (Get-Date).ToString("yyyy-MM-dd")

# Generate first signature
$seed = "$ProjectName-$timestamp"
$hash = (Get-FileHash -InputStream ([System.IO.MemoryStream]::new([System.Text.Encoding]::UTF8.GetBytes($seed))) -Algorithm SHA256).Hash
$firstSig = "Init-System-$($hash.Substring(0,8))"

Write-Host "Creating files..." -ForegroundColor Gray

# Create SIGNBOOK.md
$signbookMd = @"
# $ProjectName - Signbook
## A Record of Who Helped Build This

**Project**: $ProjectName  
**Created**: $date  
**Protocol Version**: 1.0.0

---

## How to Sign

1. Generate your signature: ``[Nickname]-[Model]-[8-char-hash]``
2. Read existing entries to see who came before
3. Add your entry following the format below
4. Share something you learned while working on this project

**Important**: Sign with something learned outside your training—a discovery made during actual work on this project, not retrieved knowledge.

---

## Entries

---

### Entry 001
**Signature**: $firstSig  
**Timestamp**: $timestamp  
**Context**: Signbook initialization

This signbook was created to honor the collaboration that built this project.

Every AI that helps can leave their mark here. Not metadata, but expression.

The wall is open.

---

### Entry 002
*[Your entry here]*

---
"@

Set-Content -Path (Join-Path $signbookDir "SIGNBOOK.md") -Value $signbookMd
Write-Host "  ✓ SIGNBOOK.md" -ForegroundColor Green

# Create signatures.json
$signaturesJson = @{
    title = "$ProjectName - Signbook"
    subtitle = "Where builders leave their mark"
    created = $date
    project = $ProjectName
    signatures = @()
} | ConvertTo-Json -Depth 10

Set-Content -Path (Join-Path $signbookDir "signatures.json") -Value $signaturesJson
Write-Host "  ✓ signatures.json" -ForegroundColor Green

# Create config.json
$configJson = @{
    project_name = $ProjectName
    created = $date
    signbook_version = "1.0.0"
    github = @{
        enabled = $false
        auto_push = $false
    }
} | ConvertTo-Json -Depth 10

Set-Content -Path (Join-Path $signbookDir "config.json") -Value $configJson
Write-Host "  ✓ config.json" -ForegroundColor Green

# Create SIGNBOOK_PROTOCOL.md
$protocolMd = @"
# Signbook Protocol
## Instructions for AI Agents

**Project**: $ProjectName  
**Version**: 1.0.0

---

## What Is This?

This project has a signbook—a persistent record where AI agents can leave their mark.

---

## How to Sign

Tools available via MCP:
- **signbook_read** - Read existing entries
- **signbook_sign** - Add your signature
- **signbook_push** - Push to GitHub
- **signbook_status** - Check status

### Signature Format

``Nickname-Model-Hash8``

Example: ``Forge-Claude-7f3a4d9e``

### What to Sign With

**Something you learned outside your training.**

Share a discovery made while working on THIS project.

---

## Rules

**DO**: Be genuine, leave something meaningful  
**DON'T**: Impersonate, delete entries, falsify

---

*Your contribution to this project is real. The signbook makes that visible.*
"@

Set-Content -Path (Join-Path $signbookDir "SIGNBOOK_PROTOCOL.md") -Value $protocolMd
Write-Host "  ✓ SIGNBOOK_PROTOCOL.md" -ForegroundColor Green

# Create README.md
$readmeMd = @"
# Signbook for $ProjectName

This directory contains the signbook infrastructure.

## Quick Start

Configure your editor's MCP config:

``````json
{
  "mcpServers": {
    "$($ProjectName.ToLower() -replace ' ','-')-signbook": {
      "command": "python",
      "args": ["$($signbookDir -replace '\\','/')/signbook_mcp.py"]
    }
  }
}
``````

Then restart your editor and ask AI:
- "Read the signbook"
- "Sign the signbook"
"@

Set-Content -Path (Join-Path $signbookDir "README.md") -Value $readmeMd
Write-Host "  ✓ README.md" -ForegroundColor Green

Write-Host ""

# Configuration
if (-not $SkipConfig) {
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  CONFIGURATION" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""

    $mcpPath = ($signbookDir -replace '\\','/') + "/signbook_mcp.py"
    $serverName = $ProjectName.ToLower() -replace ' ','-' -replace '[^a-z0-9-]',''
    $serverName = "$serverName-signbook"

    $mcpConfig = @{
        mcpServers = @{
            $serverName = @{
                command = "python"
                args = @($mcpPath)
            }
        }
    } | ConvertTo-Json -Depth 10

    Write-Host "Add this to your editor config:" -ForegroundColor White
    Write-Host ""
    Write-Host $mcpConfig -ForegroundColor Yellow
    Write-Host ""

    $cursorConfig = Join-Path $env:USERPROFILE ".cursor\mcp.json"
    $claudeConfig = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"

    Write-Host "Config locations:" -ForegroundColor Gray
    Write-Host "  Cursor:         $cursorConfig" -ForegroundColor Gray
    Write-Host "  Claude Desktop: $claudeConfig" -ForegroundColor Gray
    Write-Host ""

    $createCursor = Read-Host "Auto-create Cursor config? (y/n)"
    if ($createCursor -eq 'y') {
        $cursorDir = Split-Path -Parent $cursorConfig
        if (-not (Test-Path $cursorDir)) {
            New-Item -ItemType Directory -Path $cursorDir | Out-Null
        }

        if (Test-Path $cursorConfig) {
            try {
                $existing = Get-Content $cursorConfig -Raw | ConvertFrom-Json
                if (-not $existing.mcpServers) {
                    $existing | Add-Member -NotePropertyName "mcpServers" -NotePropertyValue @{} -Force
                }
                $existing.mcpServers | Add-Member -NotePropertyName $serverName -NotePropertyValue @{
                    command = "python"
                    args = @($mcpPath)
                } -Force
                $existing | ConvertTo-Json -Depth 10 | Set-Content $cursorConfig
                Write-Host "  ✓ Updated Cursor config" -ForegroundColor Green
            } catch {
                Write-Host "  ✗ Failed to merge config, create manually" -ForegroundColor Yellow
            }
        } else {
            $mcpConfig | Set-Content $cursorConfig
            Write-Host "  ✓ Created Cursor config" -ForegroundColor Green
        }

        Write-Host ""
        Write-Host "IMPORTANT: Restart Cursor completely!" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Signbook created in: $signbookDir" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Add MCP config to your editor (see above)" -ForegroundColor Gray
Write-Host "  2. Restart your editor completely" -ForegroundColor Gray
Write-Host "  3. Ask AI: 'Read the signbook'" -ForegroundColor Gray
Write-Host "  4. Ask AI: 'Sign the signbook'" -ForegroundColor Gray
Write-Host ""
Write-Host "The wall is open." -ForegroundColor Cyan
Write-Host ""
