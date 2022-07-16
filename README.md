# 📻 DRAKIFY
The world's first slowed and reverb'ed algorithm implemented using basic scipy/numpy functions in Python. Written by Shayan Gheidi. 
## About
An easy-to-use audio effect library featuring high quality effects and unique presets. 

Getting your favorite song "slowed and reverbed" is as simple as

    drakify('songname.wav')
    
 Listen to "drakify_example.wav" for before and after versions showing my Drakify algorithm. (The song is my 2017 remix to Steve James - Renaissance)
  
## Dependencies

This library uses a few basic python libraries: numpy, pylab, scipy(fftpack,signal) and soundfile.

    pip install -r requirements.txt
    
## In a nutshell?

    !git clone https://github.com/amirblaese/drakify
    cd drakify
    import audiosfx as aud
    aud.drakify('my_favorite_song.wav')

## Features 
- Filters
   - Lowpass, highpass, bandpass IIR filters (Butterworth)
   - FFT brickwall filters
- Speed up and down
- Convolution reverb
- Delay
- Dynamic range compression 
- Limiter 
- Distortion (numerous shapes)
- Stereo conversion
- Mono conversion
- Waveform viewing
- Waveform snipping
- Mixing
- Normalization
- Fun and unique presets designed `for real world purposes`.

## How to use?

Generally, using a function is as simple as inputting the filename into the function as a string along with relevant parameters, 
such as the number of poles in the filter, or threshold and timing for dynamic range compression. For many functions,
I have defined default values that, to me, sound good. This way you won't get caught up in details.

The functions, by default, will process and render the file to a new *<.wav>* file with an appended string. The next section
will document the details of the functions. The fine details of each function can be found in the documentation strings.

## Details of functions

###### Inputwav

Reads in the audio file and determines if the source is mono or stereo. Returns decoded array in bits and dB, sample rate, number of 
channels, and length of input in samples.

###### Normalize

Normalizes the audio to 0 dB. 

###### FFT_brickwall_<>

A series of FFT filters meant for brickwall highpass or lowpass or band reject filters. Sometimes you just really want that *one* frequency to be gone 
and not just attenuated.
These generally do not sound good.

![Alt text](pic/FFT_Filter.png?raw=true "FFT HPF at 200 Hz. Notice the ringing introduced in the time domain in the filtered data.")

###### Filters (LPF, HPF, BP)

Quick and easy implementation of the scipy Butterworth IIR filters to process audio.

![Alt text](pic/Filter.png?raw=true "Single pole HPF at 200 Hz.")

###### Slow

Resample the audio to convert to higher or lower speed (default slows by 10%).

###### Convolution Reverb (NEW!)

Convolution reverb implementation, much faster than the previous iteration and sounds like real reverb. You can use your own impulse response file if you wish.

###### Verb

Delay algorithm. Currently a bit finnicky to use but can sound quite nice with correct settings, see the **drakifyL** function for some nice numbers.

![Alt text](pic/delay.png?raw=true "Illustration of the delay algorithm. The original signal is repeated <y> times with reduced amplitude at <x> sample spacing")

###### Compress (limit)

Dynamic range compressor. Contains almost all standard settings of compressors (threshold, ratio, attack, release, makeup). Currently only uses a hard knee and does **not** calculate makeup gain automatically. Also performs gain smoothing algorithm. Beware of low attack times that may cause distortion. Limit applies compression at high ratio.

![Alt text](pic/Compressor.png?raw=true "Dynamic range compression and gain reduction curves.")

###### Distort

Arctan waveshaper.

###### Stereo/Mono

Converts mono(stereo) input to stereo(mono).

###### Snip

Cuts starting and ending point of audio file.

###### Mix

Mix two signals at desired ratio.

###### Drakify

The main goal behind this project was to convert music into a "slowed and reverbed" type songs that may be found on Youtube as remixes. This function emulates this using presets of above effects.
