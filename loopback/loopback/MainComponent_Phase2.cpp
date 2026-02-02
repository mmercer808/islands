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
    
    addAndMakeVisible (bufferLengthLabel);
    bufferLengthLabel.setText ("Buffer Length (seconds):", juce::dontSendNotification);
    
    addAndMakeVisible (bufferLengthSlider);
    bufferLengthSlider.setRange (1.0, 30.0, 1.0);
    bufferLengthSlider.setValue (10.0);
    bufferLengthSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 60, 20);
    bufferLengthSlider.onValueChange = [this] { resizeCircularBuffer(); };
    
    // Initialize circular buffer with default size (10 seconds at 48kHz)
    int initialBufferSize = static_cast<int>(bufferLengthSlider.getValue() * 48000);
    circularBuffer.setSize (2, initialBufferSize);
    circularBuffer.clear();
    fifo.setTotalSize (initialBufferSize);
    
    // Set up audio - request 2 input channels and 2 output channels
    setAudioChannels (2, 2);
    
    // Set window size
    setSize (600, 500);
    
    updateStatus ("Ready - Circular buffer active");
}

MainComponent::~MainComponent()
{
    shutdownAudio();
}

//==============================================================================
void MainComponent::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    // Store sample rate for buffer calculations
    currentSampleRate = sampleRate;
    
    // Resize buffer based on current sample rate
    resizeCircularBuffer();
    
    juce::String message = "Audio ready: ";
    message << juce::String (sampleRate, 0) << " Hz, ";
    message << juce::String (samplesPerBlockExpected) << " samples/block, ";
    message << "Buffer: " << juce::String (bufferLengthSlider.getValue(), 1) << "s";
    
    updateStatus (message);
}

void MainComponent::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    // This is the real-time audio callback
    
    auto* device = deviceManager.getCurrentAudioDevice();
    auto activeInputChannels = device->getActiveInputChannels();
    auto activeOutputChannels = device->getActiveOutputChannels();
    auto maxInputChannels = activeInputChannels.getHighestBit() + 1;
    auto maxOutputChannels = activeOutputChannels.getHighestBit() + 1;
    
    // Calculate levels for monitoring
    inputLevel = 0.0f;
    outputLevel = 0.0f;
    
    // STEP 1: Write incoming audio to circular buffer
    writeToCircularBuffer (bufferToFill.buffer, bufferToFill.numSamples, maxInputChannels);
    
    // STEP 2: Read from circular buffer and output
    // For now, we'll read from the current position (0 delay)
    // Later you can add a delay offset for creative effects
    readFromCircularBuffer (bufferToFill.buffer, bufferToFill.numSamples, 
                           maxOutputChannels, 0);
    
    // Trigger repaint to update level meters
    juce::MessageManager::callAsync ([this] { repaint(); });
}

void MainComponent::releaseResources()
{
    updateStatus ("Audio stopped");
}

//==============================================================================
void MainComponent::writeToCircularBuffer (juce::AudioBuffer<float>* source, 
                                          int numSamples, 
                                          int numChannels)
{
    int bufferSize = circularBuffer.getNumSamples();
    
    for (int channel = 0; channel < juce::jmin (numChannels, circularBuffer.getNumChannels()); ++channel)
    {
        // Calculate input level
        const float* sourceData = source->getReadPointer (channel);
        for (int i = 0; i < numSamples; ++i)
        {
            inputLevel = juce::jmax (inputLevel, std::abs (sourceData[i]));
        }
        
        // Write to circular buffer with wrap-around
        int space = bufferSize - writePosition;
        int firstPartSize = juce::jmin (numSamples, space);
        
        // Write first part (up to end of buffer)
        circularBuffer.copyFrom (channel, writePosition, *source, channel, 0, firstPartSize);
        
        // Write second part (wrapped to beginning) if necessary
        if (firstPartSize < numSamples)
        {
            int secondPartSize = numSamples - firstPartSize;
            circularBuffer.copyFrom (channel, 0, *source, channel, firstPartSize, secondPartSize);
        }
    }
    
    // Update write position with wrap-around
    writePosition = (writePosition + numSamples) % bufferSize;
}

void MainComponent::readFromCircularBuffer (juce::AudioBuffer<float>* dest,
                                           int numSamples,
                                           int numChannels,
                                           int delayInSamples)
{
    int bufferSize = circularBuffer.getNumSamples();
    
    // Calculate read position (write position - delay, wrapped)
    int readPos = (writePosition - delayInSamples + bufferSize) % bufferSize;
    
    for (int channel = 0; channel < juce::jmin (numChannels, circularBuffer.getNumChannels()); ++channel)
    {
        int space = bufferSize - readPos;
        int firstPartSize = juce::jmin (numSamples, space);
        
        // Read first part
        dest->copyFrom (channel, 0, circularBuffer, channel, readPos, firstPartSize);
        
        // Read second part (wrapped) if necessary
        if (firstPartSize < numSamples)
        {
            int secondPartSize = numSamples - firstPartSize;
            dest->copyFrom (channel, firstPartSize, circularBuffer, channel, 0, secondPartSize);
        }
        
        // Calculate output level
        const float* destData = dest->getReadPointer (channel);
        for (int i = 0; i < numSamples; ++i)
        {
            outputLevel = juce::jmax (outputLevel, std::abs (destData[i]));
        }
    }
}

void MainComponent::resizeCircularBuffer()
{
    if (currentSampleRate <= 0)
        return;
    
    int newBufferSize = static_cast<int>(bufferLengthSlider.getValue() * currentSampleRate);
    
    // Resize the buffer (this will clear it)
    circularBuffer.setSize (2, newBufferSize, false, true, true);
    fifo.setTotalSize (newBufferSize);
    writePosition = 0;
    
    DBG ("Buffer resized to " << newBufferSize << " samples (" 
         << bufferLengthSlider.getValue() << " seconds at " 
         << currentSampleRate << " Hz)");
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    
    // Draw buffer info
    auto bounds = getLocalBounds().reduced (20);
    auto infoBounds = bounds.removeFromTop (100);
    
    g.setColour (juce::Colours::white);
    g.setFont (14.0f);
    
    juce::String bufferInfo = "Circular Buffer: ";
    bufferInfo << juce::String (circularBuffer.getNumSamples()) << " samples (";
    bufferInfo << juce::String (bufferLengthSlider.getValue(), 1) << " seconds)";
    
    g.drawText (bufferInfo, infoBounds, juce::Justification::centred);
    
    // Draw level meters
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
    bounds.removeFromTop (20);
    
    auto sliderArea = bounds.removeFromTop (30);
    bufferLengthLabel.setBounds (sliderArea.removeFromLeft (200));
    bufferLengthSlider.setBounds (sliderArea);
}

//==============================================================================
void MainComponent::updateStatus (const juce::String& message)
{
    statusLabel.setText (message, juce::dontSendNotification);
}

void MainComponent::showAudioSettings()
{
    auto* selector = new juce::AudioDeviceSelectorComponent (
        deviceManager,
        0, 256,
        0, 256,
        false, false, false, false);
    
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
