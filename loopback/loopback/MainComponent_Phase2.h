#pragma once

#include <JuceHeader.h>

//==============================================================================
/**
 * Main audio component for the Audio Loopback Router
 * 
 * Phase 2: Includes circular buffer implementation
 * 
 * This component handles:
 * - Audio input/output configuration
 * - Circular buffer for looping recent audio
 * - GUI controls for device selection and buffer configuration
 * - Real-time level monitoring
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
    // Circular Buffer Components
    juce::AudioBuffer<float> circularBuffer;
    juce::AbstractFifo fifo {0};
    int writePosition = 0;
    double currentSampleRate = 0.0;
    
    //==============================================================================
    // GUI Components
    juce::Label titleLabel;
    juce::Label statusLabel;
    juce::TextButton audioSettingsButton;
    juce::Label bufferLengthLabel;
    juce::Slider bufferLengthSlider;
    
    // Audio level monitoring
    float inputLevel = 0.0f;
    float outputLevel = 0.0f;
    
    //==============================================================================
    // Circular buffer methods
    void writeToCircularBuffer (juce::AudioBuffer<float>* source, 
                               int numSamples,
                               int numChannels);
                               
    void readFromCircularBuffer (juce::AudioBuffer<float>* dest,
                                int numSamples,
                                int numChannels,
                                int delayInSamples);
                                
    void resizeCircularBuffer();
    
    //==============================================================================
    void updateStatus (const juce::String& message);
    void showAudioSettings();
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
