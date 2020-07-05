# DRAKIFY

## About

An easy-to-use audio effect library featuring high quality effects and unique presets. 

## Dependencies

This library uses a few basic python libraries: numpy, pylab, scipy(fftpack,signal) and soundfile.

## Features 
- Filters
   - Lowpass, Highpass, Bandpass IIR filters (Butterworth)
   - FFT brickwall filters
- Speed up and down
- Delay
- Dynamic range compression 
- Limiter 
- Distortion (numerous shapes)
- Stereo conversion
- Mono conversion
- Chorus
- Waveform viewing
- Waveform snipping
- Mixing
- Normalization
- Fun and unique presets designed for real world purposes.

## How to use?

Generally, using a function is as simple as inputting the filename into the function as a string along with relevant parameters, 
such as the number of poles in the filter, or threshold and timing for dynamic range compression. For many functions,
I have defined default values that, to me, sound good. This way you won't get caught up in details.

The functions, by default, will process and render the file to a new *<.wav>* file with an appended string. The next section
will document the details of the functions. The fine details of each function can be found in the documentation strings.

## Details of functions

###### inputwav

Reads in the audio file and determines if the source is mono or stereo. Returns decoded array in bits and dB, sample rate, number of 
channels, and length of input in samples.

###### normalize

Normalizes the audio to 0 dB. 

###### FFT_brickwall_<>

A series of FFT filters meant for brickwall highpass or lowpass or band reject filters. Sometimes you just really want that *one* frequency to be gone 
and not just attenuated.
These generally do not sound good.

![Alt text](pic/FFT_Filter.svg?raw=true "FFT HPF at 200 Hz. Notice the ringing introduced in the time domain in the filtered data.")

###### Filters (LPF, HPF, BP)

Quick and easy implementation of the scipy Butterworth IIR filters to process audio.


![Alt text](pic/Compressor.png?raw=true "Screenshot 1")