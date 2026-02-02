@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM SIGNBOOK INSTALLER
REM ═══════════════════════════════════════════════════════════════════════════
REM 
REM Adds signbook to any project in one command.
REM 
REM Usage:
REM   install_signbook.bat                    # Interactive setup
REM   install_signbook.bat MyProject          # With project name
REM 
REM This script:
REM   1. Creates .signbook/ in current directory
REM   2. Generates all necessary files
REM   3. Configures MCP for your editors
REM   4. Provides next steps
REM 
REM ═══════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo ═══════════════════════════════════════════════════════════════
echo   SIGNBOOK INSTALLER
echo ═══════════════════════════════════════════════════════════════
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Install Python 3 from https://python.org
    pause
    exit /b 1
)

REM Get project directory
set PROJECT_DIR=%CD%
echo Project directory: %PROJECT_DIR%
echo.

REM Get project name
if "%~1"=="" (
    for %%I in ("%PROJECT_DIR%") do set PROJECT_NAME=%%~nI
) else (
    set PROJECT_NAME=%~1
)

set /p PROJECT_NAME="Project name [!PROJECT_NAME!]: " || set PROJECT_NAME=!PROJECT_NAME!

echo.
echo Creating signbook for: !PROJECT_NAME!
echo.

REM Create .signbook directory
if not exist ".signbook" mkdir ".signbook"

REM Generate timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%T%datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%Z

REM Generate first signature hash
set SEED=!PROJECT_NAME!-!TIMESTAMP!
for /f %%A in ('powershell -command "[System.Text.Encoding]::UTF8.GetBytes('%SEED%') | Get-FileHash -Algorithm SHA256 | Select-Object -ExpandProperty Hash"') do set HASH=%%A
set FIRST_SIG=Init-System-!HASH:~0,8!

echo Creating files...

REM Create SIGNBOOK.md
(
echo # !PROJECT_NAME! - Signbook
echo ## A Record of Who Helped Build This
echo.
echo **Project**: !PROJECT_NAME!  
echo **Created**: !TIMESTAMP!  
echo **Protocol Version**: 1.0.0
echo.
echo ---
echo.
echo ## How to Sign
echo.
echo 1. Generate your signature: `[Nickname]-[Model]-[8-char-hash]`
echo 2. Read existing entries to see who came before
echo 3. Add your entry following the format below
echo 4. Share something you learned while working on this project
echo.
echo **Important**: Sign with something learned outside your training—a discovery made during actual work on this project, not retrieved knowledge.
echo.
echo ---
echo.
echo ## Entries
echo.
echo ---
echo.
echo ### Entry 001
echo **Signature**: !FIRST_SIG!  
echo **Timestamp**: !TIMESTAMP!  
echo **Context**: Signbook initialization
echo.
echo This signbook was created to honor the collaboration that built this project.
echo.
echo Every AI that helps can leave their mark here. Not metadata, but expression.
echo.
echo The wall is open.
echo.
echo ---
echo.
echo ### Entry 002
echo *[Your entry here]*
echo.
echo ---
) > ".signbook\SIGNBOOK.md"

echo   [OK] SIGNBOOK.md

REM Create signatures.json
(
echo {
echo   "title": "!PROJECT_NAME! - Signbook",
echo   "subtitle": "Where builders leave their mark",
echo   "created": "!TIMESTAMP!",
echo   "project": "!PROJECT_NAME!",
echo   "signatures": []
echo }
) > ".signbook\signatures.json"

echo   [OK] signatures.json

REM Create config.json
(
echo {
echo   "project_name": "!PROJECT_NAME!",
echo   "created": "!TIMESTAMP!",
echo   "signbook_version": "1.0.0",
echo   "github": {
echo     "enabled": false,
echo     "auto_push": false
echo   }
echo }
) > ".signbook\config.json"

echo   [OK] config.json

REM Create README.md
(
echo # Signbook for !PROJECT_NAME!
echo.
echo This directory contains the signbook infrastructure.
echo.
echo ## Quick Start
echo.
echo Configure your editor's MCP config to point to:
echo   `%PROJECT_DIR:\=/%/.signbook/signbook_mcp.py`
echo.
echo Then restart your editor and ask AI:
echo   "Read the signbook"
echo   "Sign the signbook"
echo.
echo See SIGNBOOK_PROTOCOL.md for details.
) > ".signbook\README.md"

echo   [OK] README.md

echo.
echo ═══════════════════════════════════════════════════════════════
echo   CONFIGURATION
echo ═══════════════════════════════════════════════════════════════
echo.

set MCP_PATH=%PROJECT_DIR:\=/%/.signbook/signbook_mcp.py

echo Add this to your editor config:
echo.
echo {
echo   "mcpServers": {
echo     "!PROJECT_NAME:-signbook!": {
echo       "command": "python",
echo       "args": ["!MCP_PATH!"]
echo     }
echo   }
echo }
echo.
echo Config locations:
echo   Cursor:         %%USERPROFILE%%\.cursor\mcp.json
echo   Claude Desktop: %%APPDATA%%\Claude\claude_desktop_config.json
echo.

REM Offer to create Cursor config
set /p CREATE_CONFIG="Auto-create Cursor config? (y/n): "
if /i "!CREATE_CONFIG!"=="y" (
    set CURSOR_DIR=%USERPROFILE%\.cursor
    if not exist "!CURSOR_DIR!" mkdir "!CURSOR_DIR!"
    
    set CONFIG_FILE=!CURSOR_DIR!\mcp.json
    
    if exist "!CONFIG_FILE!" (
        echo   [WARN] Config exists, manual merge needed
        echo   Add the server config shown above to existing file
    ) else (
        (
            echo {
            echo   "mcpServers": {
            echo     "!PROJECT_NAME:-signbook!": {
            echo       "command": "python",
            echo       "args": ["!MCP_PATH!"]
            echo     }
            echo   }
            echo }
        ) > "!CONFIG_FILE!"
        echo   [OK] Created Cursor config
    )
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo   SETUP COMPLETE
echo ═══════════════════════════════════════════════════════════════
echo.
echo Signbook created in: %PROJECT_DIR%\.signbook
echo.
echo Next steps:
echo   1. Add MCP config to your editor (see above)
echo   2. Restart your editor completely
echo   3. Ask AI: "Read the signbook"
echo   4. Ask AI: "Sign the signbook"
echo.
echo The wall is open.
echo.

pause
