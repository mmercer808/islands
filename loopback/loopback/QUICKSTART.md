# 🚀 QUICK START - Get Running in 5 Minutes

## Prerequisites Check
```bash
# Check you have these installed:
cmake --version    # Need 3.22+
git --version      # Any recent version
clang --version    # Or gcc/MSVC
```

## 1️⃣ Create Project (1 minute)
```bash
# Run the setup script
chmod +x setup_juce_project.sh
./setup_juce_project.sh

# This creates ~/AudioLoopbackRouter with JUCE included
```

## 2️⃣ Copy Files (30 seconds)

### For Phase 1 (Basic Passthrough - START HERE):
```bash
cd ~/AudioLoopbackRouter
cp /path/to/CMakeLists.txt ./
cp /path/to/Main.cpp ./src/
cp /path/to/MainComponent.h ./src/
cp /path/to/MainComponent_Phase1.cpp ./src/MainComponent.cpp
mkdir -p .vscode
cp /path/to/settings.json ./.vscode/
```

## 3️⃣ Open & Build (2 minutes)
```bash
# Open in Cursor/VS Code
cd ~/AudioLoopbackRouter
code .

# When prompted, select your CMake kit (Clang/GCC/Visual Studio)

# Build it:
# Press: Ctrl+Shift+P (Cmd+Shift+P on Mac)
# Type: CMake: Build
# Hit Enter
```

## 4️⃣ Run It! (30 seconds)
```bash
# macOS
./build/AudioLoopbackRouter_artefacts/Debug/AudioLoopbackRouter.app/Contents/MacOS/AudioLoopbackRouter

# Linux
./build/AudioLoopbackRouter_artefacts/Debug/AudioLoopbackRouter

# Windows
./build/AudioLoopbackRouter_artefacts/Debug/AudioLoopbackRouter.exe
```

## ✅ What You Should See
- Window titled "Audio Loopback Router"
- "Audio Settings" button
- Green and blue level meters
- Status showing your sample rate

## 🎛️ First Test
1. Click "Audio Settings"
2. Select your audio interface
3. Enable input channels (where audio comes from)
4. Enable output channels (where audio goes to)
5. Click outside dialog to close
6. Make some noise into your interface → meters should move!

## 🔄 Upgrade to Phase 2 (Circular Buffer)

Once Phase 1 works:
```bash
cd ~/AudioLoopbackRouter/src
cp /path/to/MainComponent_Phase2.h ./MainComponent.h
cp /path/to/MainComponent_Phase2.cpp ./MainComponent.cpp

# Rebuild
cd ..
cmake --build build
```

Now you'll have:
- Circular buffer storing last 10 seconds (configurable)
- Buffer length slider
- All the same passthrough functionality

## 🆘 Quick Fixes

**"CMake not found"**
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake

# Windows
# Download from cmake.org
```

**"No compiler found"**
```bash
# macOS
xcode-select --install

# Ubuntu
sudo apt install build-essential

# Windows
# Install Visual Studio with C++ tools
```

**"Can't find JUCE"**
```bash
cd ~/AudioLoopbackRouter
git submodule update --init --recursive
```

## 📍 You Are Here
```
✅ Phase 1: Basic Passthrough  ← START HERE
⏭️  Phase 2: Circular Buffer   ← UPGRADE TO THIS
⏭️  Phase 3: Advanced GUI
⏭️  Phase 4: Creative Features
```

## 🎯 Next Step
Once you have Phase 1 running, see README.md for full details on Phase 2 and beyond!
