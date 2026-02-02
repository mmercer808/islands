#include "MainComponent.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Set up GUI components
    addAndMakeVisible (titleLabel);
    titleLabel.setText ("Audio Loopback Router", juce::dontSendNotification);
    titleLabel.setFont (juce::Font (24.0f, juce::Font::bold));
    titleLabel.setJustificationType (juce::Justification::centred);
    
    addAndMakeVisible (statusLabel);
    statusLabel.setText ("Initializing...", juce::dontSendNotification);
    statusLabel.setJustificationType (juce::Justification::centred);
    
    addAndMakeVisible (audioSettingsButton);
    audioSettingsButton.setButtonText ("Audio Settings");
    audioSettingsButton.onClick = [this] { showAudioSettings(); };
    
    // Set up audio - request 2 input channels and 2 output channels
    setAudioChannels (2, 2);
    
    // Set window size
    setSize (600, 400);
    
    updateStatus ("Ready - Passthrough mode active");
}

MainComponent::~MainComponent()
{
    shutdownAudio();
}

//==============================================================================
void MainComponent::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    // Called before audio playback starts
    // You can use this to initialize any resources
    
    juce::String message = "Audio ready: ";
    message << juce::String (sampleRate, 0) << " Hz, ";
    message << juce::String (samplesPerBlockExpected) << " samples/block";
    
    updateStatus (message);
}

void MainComponent::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    // This is the real-time audio callback
    // For now, we'll just pass the input through to the output
    
    auto* device = deviceManager.getCurrentAudioDevice();
    auto activeInputChannels = device->getActiveInputChannels();
    auto activeOutputChannels = device->getActiveOutputChannels();
    auto maxInputChannels = activeInputChannels.getHighestBit() + 1;
    auto maxOutputChannels = activeOutputChannels.getHighestBit() + 1;
    
    // Calculate levels for monitoring
    inputLevel = 0.0f;
    outputLevel = 0.0f;
    
    for (auto channel = 0; channel < maxOutputChannels; ++channel)
    {
        if ((!activeOutputChannels[channel]) || maxInputChannels == 0)
        {
            bufferToFill.buffer->clear (channel, bufferToFill.startSample, bufferToFill.numSamples);
        }
        else
        {
            auto actualInputChannel = channel % maxInputChannels;
            
            if (!activeInputChannels[actualInputChannel])
            {
                bufferToFill.buffer->clear (channel, bufferToFill.startSample, bufferToFill.numSamples);
            }
            else
            {
                // Copy input to output (passthrough)
                auto* inBuffer = bufferToFill.buffer->getReadPointer (actualInputChannel,
                                                                      bufferToFill.startSample);
                auto* outBuffer = bufferToFill.buffer->getWritePointer (channel,
                                                                        bufferToFill.startSample);
                
                // Calculate input level
                for (auto sample = 0; sample < bufferToFill.numSamples; ++sample)
                {
                    float sampleValue = inBuffer[sample];
                    inputLevel = juce::jmax (inputLevel, std::abs (sampleValue));
                    outBuffer[sample] = sampleValue;  // Passthrough
                    outputLevel = juce::jmax (outputLevel, std::abs (sampleValue));
                }
            }
        }
    }
    
    // Trigger repaint to update level meters
    juce::MessageManager::callAsync ([this] { repaint(); });
}

void MainComponent::releaseResources()
{
    // Called when audio playback stops
    updateStatus ("Audio stopped");
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    
    // Draw level meters
    auto bounds = getLocalBounds().reduced (20);
    auto meterArea = bounds.removeFromBottom (80);
    
    // Input level meter
    auto inputMeterArea = meterArea.removeFromLeft (getWidth() / 2 - 10);
    g.setColour (juce::Colours::grey);
    g.fillRect (inputMeterArea.reduced (5));
    g.setColour (juce::Colours::green);
    auto inputWidth = static_cast<int> (inputMeterArea.getWidth() * inputLevel);
    g.fillRect (inputMeterArea.getX() + 5, inputMeterArea.getY() + 5, 
                inputWidth - 10, inputMeterArea.getHeight() - 10);
    
    g.setColour (juce::Colours::white);
    g.drawText ("Input", inputMeterArea.removeFromTop (20), juce::Justification::centred);
    
    // Output level meter
    auto outputMeterArea = meterArea.removeFromRight (getWidth() / 2 - 10);
    g.setColour (juce::Colours::grey);
    g.fillRect (outputMeterArea.reduced (5));
    g.setColour (juce::Colours::blue);
    auto outputWidth = static_cast<int> (outputMeterArea.getWidth() * outputLevel);
    g.fillRect (outputMeterArea.getX() + 5, outputMeterArea.getY() + 5,
                outputWidth - 10, outputMeterArea.getHeight() - 10);
    
    g.setColour (juce::Colours::white);
    g.drawText ("Output", outputMeterArea.removeFromTop (20), juce::Justification::centred);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds().reduced (20);
    
    titleLabel.setBounds (bounds.removeFromTop (40));
    bounds.removeFromTop (10);
    
    statusLabel.setBounds (bounds.removeFromTop (30));
    bounds.removeFromTop (20);
    
    audioSettingsButton.setBounds (bounds.removeFromTop (30).reduced (100, 0));
}

//==============================================================================
void MainComponent::updateStatus (const juce::String& message)
{
    statusLabel.setText (message, juce::dontSendNotification);
}

void MainComponent::showAudioSettings()
{
    // Create and show the audio device selector dialog
    auto* selector = new juce::AudioDeviceSelectorComponent (
        deviceManager,
        0, 256,     // min/max input channels
        0, 256,     // min/max output channels
        false,      // show MIDI input
        false,      // show MIDI output
        false,      // show channels as stereo pairs
        false);     // hide advanced options
    
    selector->setSize (500, 450);
    
    juce::DialogWindow::LaunchOptions options;
    options.content.setOwned (selector);
    options.dialogTitle = "Audio Settings";
    options.componentToCentreAround = this;
    options.dialogBackgroundColour = getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId);
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    
    options.launchAsync();
}
