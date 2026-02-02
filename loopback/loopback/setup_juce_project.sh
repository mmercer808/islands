#!/bin/bash

# JUCE Audio Loopback Router - Project Setup Script
# This script sets up a complete JUCE development environment for Cursor/VS Code

set -e  # Exit on error

echo "======================================"
echo "JUCE Audio Loopback Router Setup"
echo "======================================"
echo ""

# Project configuration
PROJECT_NAME="AudioLoopbackRouter"
PROJECT_DIR="$HOME/AudioLoopbackRouter"

echo "📁 Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create directory structure
echo "📂 Creating directory structure..."
mkdir -p src
mkdir -p lib
mkdir -p .vscode

# Clone JUCE as a submodule
echo "📦 Downloading JUCE framework..."
if [ ! -d "lib/JUCE" ]; then
    git init
    git submodule add https://github.com/juce-framework/JUCE.git lib/JUCE
    cd lib/JUCE
    git checkout 8.0.4  # Stable version
    cd ../..
else
    echo "   JUCE already exists, skipping..."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Open the project in Cursor/VS Code:"
echo "      code $PROJECT_DIR"
echo ""
echo "   2. Install required VS Code extensions:"
echo "      - CMake Tools (ms-vscode.cmake-tools)"
echo "      - C/C++ Extension Pack (ms-vscode.cpptools-extension-pack)"
echo "      - clangd (llvm-vs-code-extensions.vscode-clangd) [optional but recommended]"
echo ""
echo "   3. Select your CMake kit when prompted (Clang or GCC)"
echo ""
echo "   4. Build the project:"
echo "      - Press Ctrl+Shift+P (Cmd+Shift+P on Mac)"
echo "      - Type 'CMake: Build'"
echo ""
echo "🎉 Ready to start developing!"
