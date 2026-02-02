# Audio Loopback Router

A JUCE-based audio routing application with circular buffer functionality for creative audio workflows.

## 🎯 Project Goal

Route audio from Ableton Live to Virtual DJ through an intelligent circular buffer that:
- Stores the last X seconds of audio (configurable 1-30 seconds)
- Integrates with audio interface loopback functionality
- Allows Ableton to record the loopback channel simultaneously
- Provides a "safety net" for performance and creative loop-based workflows

## 📋 Prerequisites

### Required Software
- **CMake** (3.22 or higher)
- **C++ Compiler**:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Windows: Visual Studio 2019+ or Clang
  - Linux: GCC 9+ or Clang 10+
- **Git**
- **Cursor** or **VS Code**

### Required VS Code Extensions
1. **CMake Tools** (`ms-vscode.cmake-tools`)
2. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
3. **clangd** (`llvm-vs-code-extensions.vscode-clangd`) [optional but recommended]

## 🚀 Quick Start

### Step 1: Run Setup Script

```bash
chmod +x setup_juce_project.sh
./setup_juce_project.sh
```

This will:
- Create the project directory structure
- Clone JUCE as a submodule
- Initialize git repository

### Step 2: Copy Project Files

Copy the following files to your project directory (`~/AudioLoopbackRouter`):

**Phase 1 (Basic Passthrough):**
```bash
cp CMakeLists.txt ~/AudioLoopbackRouter/
cp Main.cpp ~/AudioLoopbackRouter/src/
cp MainComponent.h ~/AudioLoopbackRouter/src/
cp MainComponent_Phase1.cpp ~/AudioLoopbackRouter/src/MainComponent.cpp
cp settings.json ~/AudioLoopbackRouter/.vscode/
```

**OR Phase 2 (With Circular Buffer):**
```bash
cp CMakeLists.txt ~/AudioLoopbackRouter/
cp Main.cpp ~/AudioLoopbackRouter/src/
cp MainComponent_Phase2.h ~/AudioLoopbackRouter/src/MainComponent.h
cp MainComponent_Phase2.cpp ~/AudioLoopbackRouter/src/MainComponent.cpp
cp settings.json ~/AudioLoopbackRouter/.vscode/
```

### Step 3: Open in Cursor/VS Code

```bash
cd ~/AudioLoopbackRouter
code .
```

### Step 4: Configure CMake

1. When VS Code opens, you'll be prompted to select a CMake kit
2. Choose your compiler:
   - **macOS**: Select `Clang` (from Xcode)
   - **Windows**: Select `Visual Studio` or `Clang`
   - **Linux**: Select `GCC` or `Clang`

### Step 5: Build the Project

**Method 1: Using Command Palette**
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "CMake: Build"
3. Press Enter

**Method 2: Using Terminal**
```bash
cd ~/AudioLoopbackRouter
cmake -B build
cmake --build build
```

### Step 6: Run the Application

The built application will be in:
- **macOS**: `build/AudioLoopbackRouter_artefacts/Debug/AudioLoopbackRouter.app`
- **Windows**: `build/AudioLoopbackRouter_artefacts/Debug/AudioLoopbackRouter.exe`
- **Linux**: `build/AudioLoopbackRouter_artefacts/Debug/AudioLoopbackRouter`

## 📂 Project Structure

```
AudioLoopbackRouter/
├── CMakeLists.txt           # CMake configuration
├── lib/
│   └── JUCE/                # JUCE framework (git submodule)
├── src/
│   ├── Main.cpp             # Application entry point
│   ├── MainComponent.h      # Main component header
│   └── MainComponent.cpp    # Main component implementation
├── build/                   # Build output (generated)
└── .vscode/
    └── settings.json        # VS Code configuration
```

## 🔄 Switching Between Phases

### Phase 1: Basic Audio Passthrough
Simple input → output routing with level meters.

**Use these files:**
- `MainComponent.h` (original)
- `MainComponent_Phase1.cpp` → rename to `MainComponent.cpp`

### Phase 2: Circular Buffer Implementation
Full circular buffer with configurable length.

**Use these files:**
- `MainComponent_Phase2.h` → rename to `MainComponent.h`
- `MainComponent_Phase2.cpp` → rename to `MainComponent.cpp`

Then rebuild:
```bash
cmake --build build
```

## 🎛️ Using the Application

### Phase 1 (Basic)
1. Click "Audio Settings" to select your audio interface
2. Choose input channels (from Ableton)
3. Choose output channels (to interface loopback)
4. Audio passes through with real-time level monitoring

### Phase 2 (Circular Buffer)
1. Click "Audio Settings" to configure audio interface
2. Adjust "Buffer Length" slider (1-30 seconds)
3. Monitor input/output levels
4. Audio is stored in circular buffer and routed to output

## 🔧 Audio Interface Setup

### Recommended Configuration
1. **Ableton Output**: Send to channels 1-2 of your interface
2. **This App Input**: Read from channels 1-2
3. **This App Output**: Send to channels 3-4 (or your loopback channels)
4. **Interface Loopback**: Configure channels 3-4 to loop back
5. **Virtual DJ Input**: Read from loopback channels
6. **Ableton Recording**: Record from loopback channels

### Example with Focusrite Scarlett
- Ableton → Interface Outputs 1-2
- App Input ← Interface Inputs 1-2
- App Output → Interface Outputs 3-4
- Loopback 3-4 enabled in Focusrite Control
- Virtual DJ ← Loopback 3-4
- Ableton Record ← Loopback 3-4

## 🐛 Troubleshooting

### CMake Configuration Fails
```bash
# Clean and reconfigure
rm -rf build
cmake -B build
```

### No Audio
1. Check "Audio Settings" in the app
2. Verify audio interface is selected
3. Ensure input/output channels are active
4. Check system audio settings

### Build Errors
1. Make sure JUCE submodule is initialized:
   ```bash
   git submodule update --init --recursive
   ```
2. Verify CMake version: `cmake --version` (should be 3.22+)
3. Check compiler is installed

### IntelliSense Not Working
1. Open Command Palette (`Ctrl+Shift+P`)
2. Run "C/C++: Edit Configurations (UI)"
3. Add include path: `${workspaceFolder}/lib/JUCE/modules`

## 📚 Next Development Steps

### Phase 3: Advanced GUI
- [ ] Waveform visualization of buffer contents
- [ ] Loop trigger button
- [ ] Variable playback offset control
- [ ] Visual buffer position indicator

### Phase 4: Creative Features
- [ ] Multiple buffer slots
- [ ] Speed/pitch control
- [ ] Reverse playback
- [ ] Buffer freeze function

### Phase 5: Production Polish
- [ ] Settings persistence
- [ ] Preset system
- [ ] CPU usage monitoring
- [ ] Stability testing

## 📖 Learning Resources

- [JUCE Documentation](https://docs.juce.com/)
- [JUCE Tutorials](https://docs.juce.com/master/tutorial_processing_audio_input.html)
- [CMake with JUCE](https://melatonin.dev/blog/how-to-use-cmake-with-juce/)
- [JUCE Forum](https://forum.juce.com/)

## 🤝 Contributing

This is a personal project, but feel free to fork and adapt for your own needs!

## 📝 License

This project uses JUCE framework. Check [JUCE licensing](https://juce.com/juce-7-licence/) for terms.

## 🎉 Success!

Once you see the app window with level meters responding to audio, you're ready to integrate with Ableton and Virtual DJ!
