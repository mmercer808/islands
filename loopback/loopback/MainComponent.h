#pragma once

#include <JuceHeader.h>

//==============================================================================
/**
 * Main audio component for the Audio Loopback Router
 * 
 * This component handles:
 * - Audio input/output configuration
 * - Basic audio passthrough
 * - GUI controls for device selection and monitoring
 */
class MainComponent : public juce::AudioAppComponent
{
public:
    //==============================================================================
    MainComponent();
    ~MainComponent() override;

    //==============================================================================
    // Audio callbacks
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;
    void releaseResources() override;

    //==============================================================================
    // GUI
    void paint (juce::Graphics& g) override;
    void resized() override;

private:
    //==============================================================================
    // GUI Components
    juce::Label titleLabel;
    juce::Label statusLabel;
    juce::TextButton audioSettingsButton;
    
    // Audio level monitoring
    float inputLevel = 0.0f;
    float outputLevel = 0.0f;
    
    //==============================================================================
    void updateStatus (const juce::String& message);
    void showAudioSettings();
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
