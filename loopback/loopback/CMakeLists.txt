cmake_minimum_required(VERSION 3.22)

# Project name and version
project(AudioLoopbackRouter VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add JUCE
add_subdirectory(lib/JUCE)

# Define your app target
juce_add_gui_app(AudioLoopbackRouter
    PRODUCT_NAME "Audio Loopback Router"
    COMPANY_NAME "YourCompany"
    BUNDLE_ID "com.yourcompany.audioloopbackrouter"
    
    # Icon files (optional)
    # ICON_BIG ...
    # ICON_SMALL ...
)

# Source files
target_sources(AudioLoopbackRouter PRIVATE
    src/Main.cpp
    src/MainComponent.h
    src/MainComponent.cpp
)

# Preprocessor definitions
target_compile_definitions(AudioLoopbackRouter PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:AudioLoopbackRouter,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:AudioLoopbackRouter,JUCE_VERSION>"
)

# Link to JUCE modules
target_link_libraries(AudioLoopbackRouter PRIVATE
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)

# Binary data (if you have any assets)
# juce_add_binary_data(AppData SOURCES ...)
# target_link_libraries(AudioLoopbackRouter PRIVATE AppData)

# Platform-specific settings
if(APPLE)
    set_target_properties(AudioLoopbackRouter PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.yourcompany.audioloopbackrouter"
    )
endif()

if(MSVC)
    target_compile_options(AudioLoopbackRouter PRIVATE /W4)
else()
    target_compile_options(AudioLoopbackRouter PRIVATE -Wall -Wextra -Wpedantic)
endif()
