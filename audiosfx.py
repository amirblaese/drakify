"""
Created on Sat Jun 06 20:47:37 2020
@author: Shayan
"""
import numpy as np
import pylab as py
from scipy.fftpack import fft,ifft,rfft,irfft
from scipy.signal import blackman,hamming,chebwin,resample,stft,butter,lfilter
from scipy.signal import freqz
import soundfile as sf
import os
import time


def inputwav(filename):
    """
    Decodes input file using "soundfile" library. Determines if file is
    mono or stereo and converts data to dB.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
    Returns
    -------
    n: length of audio in samples
  
    data: array containing the signal in bits.
  
    data_dB: array containing the signal in dB.
  
    sr: sample rate
  
    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    data, sr = sf.read(filename)
    print('Decoding "'+filename+'"...')
    print('Sample rate is '+str(sr)+'...')
    try:
        ch=len(data[0,])
    except:
        ch=1
    print('File contains '+str(ch)+' audio channel(s)...')
    #Reshape the data so other functions can interpret the array if mono.
    #basically transposing the data
    if ch==1:
        data=data.reshape(-1,1)
    n=len(data)
    #This prevents log(data) producing nan when data is 0
    data[np.where(data==0)]=0.00001
    #convert to dB
    data_dB=20*np.log10(abs(data))
    return n, data,data_dB,sr, ch

def normalize(filename,wout=True):
    """
    Normalize volume to 0 dB.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
      
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
    Returns
    -------
    data_norm: Normalized data array in bits
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    if ch==1:
        diff=0-max(data_dB)
    if ch==2:
        d1=0-max(data_dB[:,0])
        d2=0-max(data_dB[:,1])
        diff=max(d1,d2)
    print('Adding '+str(diff)+' dB...')
    data_dB_norm=data_dB+diff
    data_norm=10.0**((data_dB_norm)/20.0)
    #sign the bits appropriately:
    for k in range (ch):
        for i in range (n):
            if data[i,k]<0.0:
                data_norm[i,k]=-1.0*data_norm[i,k]
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_normalized.wav',data_norm,sr,'PCM_16')
    print('Done!')
    return data_norm

def FFT_brickwallHPF(filename,cutoff,wout=True,plot=True):
    """
    Deletes frequencies below cutoff using brickwall. Beware of phase issues.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
      
    cutoff: int (Hz)
        Frequency of the filter cutoff below which data is filtered out.
  
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
      
    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    data_filtered: array containing filtered audio in bits
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    print('Applying FFT...')
    yfreq=rfft(data,axis=0)
    xfreq=np.linspace(0,sr/(2.0),n)
    yfreqBHPF=np.zeros((n,ch))
    yfreqBHPF[0:n,:]=yfreq
    print('Applying brickwall at '+str(cutoff)+' Hz...')
    yfreqBHPF[0:np.searchsorted(xfreq,cutoff),:]=0.0
    data_filtered=(irfft(yfreqBHPF,axis=0))
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_brickwallHPF.wav',data_filtered,sr,'PCM_16')
    if plot==True:
        print('Plotting...')
        py.close()
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(xfreq,20*np.log10(abs(yfreq[0:n,0]+.0001)),'k-',lw=0.5)
        ax1.semilogx(xfreq,20*np.log10(abs(yfreqBHPF[0:n//1,0]+.0001)),'m-',lw=0.1)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude')
        ax2.plot(data,'k-',label='Raw')
        ax2.plot(data_filtered,'m-',label='Filtered')
        ax2.set_xlim(0,10000)
        ax2.set_ylim(-1,1)
        ax2.set_ylabel('Amplitude (Norm Bits)')
        ax2.set_xlabel('Samples')
        ax2.legend(loc=2)
    print('Done!')
    return data_filtered

def FFT_brickwallBR(filename,start,stop,wout=True,plot=True):
    """
    Deletes frequencies below cutoff using brickwall. Beware of phase issues.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
      
    cutoff: int (Hz)
        Frequency of the filter cutoff below which data is filtered out.
  
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
      
    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    data_filtered: array containing filtered audio in bits
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    print('Applying FFT...')
    yfreq=fft(data,axis=0)
    xfreq=np.linspace(0,sr/(2.0),n//2)
    yfreqBHPF=np.zeros((n,ch),dtype=complex)
    yfreqBHPF[0:n,:]=yfreq
    print('Applying brickwall centered at '+str((start+stop)/2)+' Hz...')
    yfreqBHPF[np.searchsorted(xfreq,start):np.searchsorted(xfreq,stop),:]=0.00001
    data_filtered=(ifft(yfreqBHPF,axis=0))
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_brickwallHPF.wav',data_filtered.real,sr,'PCM_16')
    if plot==True:
        print('Plotting...')
        py.close()
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(xfreq,20*np.log10(abs(yfreq[0:n//2])),'k-',lw=0.5)
        ax1.semilogx(xfreq,20*np.log10(abs(yfreqBHPF[0:n//2,0])),'m-',lw=0.1)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude (dB)')
        ax2.plot(data,'k-')
        ax2.plot(data_filtered,'m-')
        ax2.set_xlim(0,1000)
      # ax2.set_ylim(-1,1)
        ax2.set_ylabel('Amplitude (Norm Bits)')
        ax2.set_xlabel('Samples')
        ax2.legend(loc=2)
    print('Done!')
    return data_filtered

def FFT_brickwallLPF(filename,cutoff,wout=True,plot=True):
    """
    Deletes frequencies above cutoff using brickwall.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
      
    cutoff: int (Hz)
        Frequency of the filter cutoff above which data is filtered out.
  
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
      
    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples
  
    data: array containing the signal in bits.
  
    data_dB: array containing the signal in dB.
  
    sr: sample rate
  
    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start=time.time()
    n, data, data_dB,sr,ch=inputwav(filename)
    print('Applying FFT...')
    W=np.zeros((n,2))
    W[:,0]=1#blackman(n)
    W[:,1]=1#blackman(n)
    yfreq=rfft(data*W,axis=0)
    xfreq=np.linspace(0,sr/(2.0),n//1)
    yfreqBLPF=np.zeros((n,ch))
    yfreqBLPF[0:n,:]=yfreq
    print('Applying brickwall at '+str(cutoff)+' Hz...')
    yfreqBLPF[n:np.searchsorted(xfreq,cutoff):-1,:]=0.0
    data_filtered=(irfft(yfreqBLPF,axis=0))
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_brickwallLPF.wav',data_filtered,sr,'PCM_16')
    if plot==True:
        print('Plotting...')
        py.close()
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(xfreq,20*np.log10(abs(yfreq[0:n//1,:]+.0001)),'k-',lw=0.5)
        ax1.semilogx(xfreq,20*np.log10(abs(yfreqBLPF[0:n//1,:]+.0001)),'m-',lw=0.1)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude (dB)')
        ax2.plot(data,'k-',label='Raw')
        ax2.plot(data_filtered,'m-',lw=1,label='Filtered')
        ax2.set_xlim(0,10000)
        ax2.set_ylim(-1,1)
        ax2.set_ylabel('Amplitude (Norm Bits)')
        ax2.set_xlabel('Samples')
        ax2.legend(loc=2,frameon=False,ncol=2)
    print('Done!')
    end=time.time()
    elapsed=(end-start)
    print('Completed in '+str(elapsed)+' seconds.')
    return data_filtered


def LPF(filename,cutoff,Q=1,wout=True,plot=True):
    """
    Lowpass filter.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
      
    cutoff: int (Hz)
        Frequency of the filter cutoff above which data is filtered out.
      
    Q:  int, optional, default=1
        Number of poles in filter or steepness of filter edge.
  
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
      
    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples
  
    data: array containing the signal in bits.
  
    data_dB: array containing the signal in dB.
  
    sr: sample rate
  
    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start=time.time()
    n, data, data_dB,sr,ch=inputwav(filename)
    b, a = butter(Q,cutoff/sr,btype='low')
    data_filtered=lfilter(b,a,data,axis=0)
    print('Applying FFT...')
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_LPF.wav',data_filtered,sr,'PCM_16')
    if plot==True:
        print('Plotting...')
        py.close()
        w, h = freqz(b,a,worN=1024)
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(0.5*sr*w/np.pi,abs(h),'k--')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Rel. Amplitude')
        ax1.grid()
        ax1.set_ylim(0,1.1)
        ax1.set_xlim(1,20000)
        ax2.plot(data,'k-',label='Raw data')
        ax2.plot(data_filtered,'m-',lw=1,label='Filtered data')
        ax2.set_xlim(0,10000)
        ax2.set_ylim(-1,1)
        ax2.set_ylabel('Amplitude (Norm Bits)')
        ax2.set_xlabel('Samples')
        ax2.legend(loc=2,frameon=False,ncol=2)
        py.subplots_adjust(hspace=0.35)    
    print('Done!')
    end=time.time()
    elapsed=int(1000*(end-start))
    print('...............................')
    print('Completed in '+str(elapsed)+' milliseconds.')
    return data_filtered

def HPF(filename,cutoff,Q=1,wout=True,plot=True):
    """
    Deletes frequencies above cutoff using brickwall.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
      
    cutoff: int (Hz)
        Frequency of the filter cutoff above which data is filtered out.
  
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
      
    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples
  
    data: array containing the signal in bits.
  
    data_dB: array containing the signal in dB.
  
    sr: sample rate
  
    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start=time.time()
    n, data, data_dB,sr,ch=inputwav(filename)
    b, a = butter(Q,cutoff/sr,btype='high')
    data_filtered=lfilter(b,a,data,axis=0)
    print('Applying FFT...')
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_HPF.wav',data_filtered,sr,'PCM_16')
    if plot==True:
        print('Plotting...')
        py.close()
        w, h = freqz(b,a,worN=16384)
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(0.5*sr*w/np.pi,abs(h),'k--')
        ax1.set_title('Fiter Frequency Response')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Rel. Amplitude')
        ax1.grid()
        ax1.set_ylim(0,1.1)
        ax1.set_xlim(1,20000)
        ax2.plot(data,'k-',label='Raw data')
        ax2.plot(data_filtered,'m-',lw=1,label='Filtered data')
        ax2.set_xlim(0,10000)
        ax2.set_ylim(-1,1)
        ax2.set_ylabel('Amplitude (Norm Bits)')
        ax2.set_xlabel('Samples')
        ax2.legend(loc=2,frameon=False,ncol=2)
        py.subplots_adjust(hspace=0.35)    
    print('Done!')
    end=time.time()
    elapsed=int(1000*(end-start))
    print('...............................')
    print('Completed in '+str(elapsed)+' milliseconds.')
    return data_filtered

def bandpass(filename,f1,f2,Q,wout=True,plot=True):
    """
    Filters frequency content outside of [f1,f2] domain.
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
      
    f1: int (Hz)
        Start frequency of the filter cutoff above which data is filtered out.
      
    f2: int (Hz)
        Stop frequency of the filter cutoff below which data is filtered out.
      
    Q:  int, optional, default=1
        Number of poles in filter or steepness of filter edge.
      
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
      
    plot: True/False, optional, default=True
        Produces plot of raw wave form and processed waveform.
    Returns
    -------
    n: length of song in samples
  
    data: array containing the signal in bits.
  
    data_dB: array containing the signal in dB.
  
    sr: sample rate
  
    ch: number of audio channels. 1 = mono, 2 = stereo
    """
    start=time.time()
    n, data, data_dB,sr,ch=inputwav(filename)
    b, a = butter(Q,Wn=(f1/sr,f2/sr),btype='bandpass')
    data_filtered=lfilter(b,a,data,axis=0)
    print('Applying FFT...')
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_BP.wav',data_filtered,sr,'PCM_16')
    if plot==True:
        print('Plotting...')
        py.close()
        w, h = freqz(b,a,worN=16384)
        fig, (ax1, ax2) = py.subplots(nrows=2)
        ax1.semilogx(0.5*sr*w/np.pi,abs(h),'k-')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Rel. Amplitude')
        ax1.grid()
        ax1.set_ylim(0,1.1)
        ax1.set_xlim(1,20000)
        ax2.plot(data,'k-',label='Raw data')
        ax2.plot(data_filtered,'m-',lw=1,label='Filtered data')
        ax2.set_xlim(0,10000)
        ax2.set_ylim(-1,1)
        ax2.set_ylabel('Amplitude (Norm Bits)')
        ax2.set_xlabel('Samples')
        ax2.legend(loc=2,frameon=False,ncol=2)
        py.subplots_adjust(hspace=0.35)    
    print('Done!')
    end=time.time()
    elapsed=int(1000*(end-start))
    print('...............................')
    print('Completed in '+str(elapsed)+' milliseconds.')
    return data_filtered

def slow(filename,p=10,wout=True):
    """
    Resamples the input by p%, slowing it down by p%. Uses the scipy resample
    function.
  
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
  
    p: float/int, optional, default=10
        percentage the audio is slowed by.
  
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
    Returns
    -------
    f: resampled data in bits
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    if p>0:
        print('Slowing...')
    if p<0:
        print('Warning: You are speeding up the audio! Use positive value'
              +' for p to slow.')
    f=resample(data,int(len(data)*(1+p/100.0)))
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_slow.wav',f,sr,'PCM_16')
        print('Done!')
    return f

  
def verb(filename,l,t,d,wout=True): #l = predelay   d= decay smaller = less decay, t= number of delays
#low l turns into chorus
    """
    Repeats sound to form delay or reverb with appropriate parameters.
  
    Parameters
    ----------
    filename: string
        Name of the input <audio> file. Uses the soundfile python library
        to decode the input.
  
    l:  int
        Spacing between samples in units of sample. Serves as "predelay"
        or the delay spacing.
          
    t:  int
        Number of delays or repetitions produced.
      
    d:  float
        Characteristic decay time of the feedback.
  
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
    Returns
    -------
    data_verb: Data with reverb added in bit units.
    """
    start=time.time()
    n, data, data_dB,sr,ch=inputwav(filename)
    data_ex=np.zeros(((n+l*t),ch))
    data_ex[0:n,:]=data
    data_Rex=np.zeros((len(data_ex),t,ch))
    print('Applying reverb...')
    for k in range (ch):
        for i in range (len(data)):
            for j in range(t):
                data_Rex[i+l*(j+1),j,k]=data_ex[i,k]*np.exp(-d*(j+1))
    data_F=data_ex
    print('Mixing...')
    for i in range (t):
        data_F=data_F+1*data_Rex[:,i,:]
    data_F=1*data_F
    data_verb=data_F+data_ex
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_verbed.wav',data_verb,sr,'PCM_16')
    print('Done!')
    end=time.time()
    elapsed=int(1000*(end-start))
    print('...............................')
    print('Completed in '+str(elapsed)+' milliseconds.')
    return data_verb



def compress(filename,threshold,ratio,makeup,attack,release,wout=True,plot=False):
    """
    Reduces dynamic range of input signal by reducing volume above threshold.
    The gain reduction is smoothened according to the attack and release.
    Makeup gain must be added manually.
    Parameters
    ----------
    filename: string
        Name of the input *.wav file.
    threshold: scalar (dB)
        value in dB of the threshold at which the compressor engages in
        gain reduction.
    ratio: scalar
        The ratio at which volume is reduced for every dB above the threshold
        (i.e. r:1)
        For compression to occur, ratio should be above 1.0. Below 1.0, you
        are expanding the signal.
      
    makeup: scalar (dB)
        Amount of makeup gain to apply to the compressed signal
  
    attack: scalar (ms)
        Characteristic time required for compressor to apply full gain
        reduction. Longer times allow transients to pass through while short
        times reduce all of the signal. Distortion will occur if the attack
        time is too short.
  
    release: scalar (ms)
        Characteristic time that the compressor will hold the gain reduction
        before easing off. Both attack and release basically smoothen the gain
        reduction curves.
      
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process.
      
    plot: True/False, optional, default=True
        Produces plot of waveform and gain reduction curves.
  
      
    Returns
    -------
    data_Cs: array containing the compressed waveform in dB
    data_Cs_bit: array containing the compressed waveform in bits.
    """
    start=time.time()
    if ratio < 1.0:
        print('Ratio must be > 1.0 for compression to occur! You are expanding.')
    if ratio==1.0:
        print('Signal is unaffected.')
    n, data, data_dB,sr,ch=inputwav(filename)
    #Array for the compressed data in dB
    dataC=data_dB.copy()
    #attack and release time constant
    a=np.exp(-np.log10(9)/(44100*attack*1.0E-3))
    re=np.exp(-np.log10(9)/(44100*release*1.0E-3))
    #apply compression
    print('Compressing...')
    for k in range(ch):
        for i in range (n):
            if dataC[i,k]>threshold:
                dataC[i,k]=threshold+(dataC[i,k]-threshold)/(ratio)
    #gain and smooth gain initialization
    gain=np.zeros(n)
    sgain=np.zeros(n)
    #calculate gain
    gain=np.subtract(dataC,data_dB)
    sgain=gain.copy()
    #smoothen gain
    print('Smoothing...')
    for k in range(ch):
        for i in range (1,n):
            if sgain[i-1,k]>=sgain[i,k]:
                sgain[i,k]=a*sgain[i-1,k]+(1-a)*sgain[i,k]
            if sgain[i-1,k]<sgain[i,k]:
                sgain[i,k]=re*sgain[i-1,k]+(1-re)*sgain[i,k]    
    #Array for the smooth compressed data with makeup gain applied
    dataCs=np.zeros(n)
    dataCs=data_dB+sgain+makeup
    #Convert our dB data back to bits
    dataCs_bit=10.0**((dataCs)/20.0)
    #sign the bits appropriately:
    for k in range (ch):
        for i in range (n):
            if data[i,k]<0.0:
                dataCs_bit[i,k]=-1.0*dataCs_bit[i,k]
    #Plot the data:
    if plot==True:
        print('Plotting...')
        t=np.linspace(0,n/(1.0*sr),n)
        py.close()
        fig, (ax1, ax2) = py.subplots(nrows=2)  
        ax2.plot(t,gain,'k-',linewidth=0.1,label='Gain Reduction')
        ax2.plot(t,sgain,'r-',linewidth=1, label='Gain Reduction Smooth')
        ax1.plot(t,data,'k-',linewidth=1,label=filename)
        ax1.plot(t,dataCs_bit,'m-',linewidth=0.1,
                label=filename+' compressed')
        ax1.axhline(10**(threshold/20.0),linestyle='-',
                    color='cyan',linewidth=1)
        ax1.axhline(-10**(threshold/20.0),linestyle='-',
                    color='cyan',linewidth=1)
        ax1.legend()
        ax2.legend()
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Gain Reduction (dB)')
        ax1.set_ylabel('Amplitude (Rel. Bit)')
        ax1.set_xlabel('Time (s)')
    #write data to 16 bit file
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_compressed.wav',dataCs_bit,
                sr,'PCM_16')
    end=time.time()
    elapsed=int(1000*(end-start))
    print('Done!')
    print('...............................')
    print('Completed in '+str(elapsed)+' milliseconds.')    
    return dataCs,dataCs_bit



def limit(filename,threshold,makeup,wout=True,plot=False):
    """
    Limits the data above the threshold. Compression with high ratio and long
    release with fairly quick attack.
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    threshold : scalar (dB)
        value in dB of the threshold above which signal is limited
      
    makeup: scalar (dB)
        Amount of makeup gain to apply to the compressed signal
  
    attack: scalar (ms)
        Characteristic time required for compressor to apply full gain
        reduction. Longer times allow transients to pass through while short
        times reduce all of the signal. Distortion will occur if the attack
        time is too short.
  
    release: scalar (ms)
        Characteristic time that the compressor will hold the gain reduction
        before easing off. Both attack and release basically smoothen the gain
        reduction curves.
      
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.
      
    plot: True/False, optional, default =True
        Produces plot of waveform and gain reduction curves.
  
      
    Returns
    -------
    dataL: array containing the limited waveform in dB
  
    dataL_bit: array containing the limited waveform in bits.
    """
    start=time.time()
    n, data, data_dB,sr,ch=inputwav(filename)
    dataL,dataL_bit=compress(filename,threshold,1000.0,makeup,1.0,500.0,wout=False,plot=plot)
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_limit.wav',dataL_bit,44100,'PCM_16')
    end=time.time()
    elapsed=int(1000*(end-start))
    print('Done!')
    print('...............................')
    print('Completed in '+str(elapsed)+' milliseconds.')    
    return dataL,dataL_bit

def distort(filename,threshold=0.25,type='arctan',wout=True,plot=False):
    """
    Applies distortion to signal. Reshapes the signal above user input
    threshold (in bits).
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    threshold : scalar 
        lower = more distortion, try 0.25 for heavy distortion.
      
    type: string, optional, default=cubic
        Type of distortion that is applied. Default is arctan. more will be added
          soon.
      
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.
      
    plot: True/False, optional, default =True
        Produces plot of input and output waveforms.
  
    Returns
    -------
    dataD: array containing the limited waveform in bits
    """
    start=time.time()
    n, data, data_dB,sr,ch=inputwav(filename)
    dataD=np.zeros((len(data),ch))
    dataD[:len(data),:]=data#data_dB
    if type=='arctan':
        print('Applying arctan distortion...')
        for k in range(ch):
            for i in range (0,len(data)):
                dataD[i,k]=(2/np.pi)*np.arctan((np.pi / threshold)*dataD[i,k])
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_distort.wav',dataD,sr,'PCM_16')
    #PLOTTING#
    if plot==True:
        print('Plotting...')
        t=np.linspace(0,n/(1.0*sr),n)
        py.close()
        fig, (ax1) = py.subplots(nrows=1)  
        ax1.plot(t,data,'k-',linewidth=1,label=filename)
        ax1.plot(t,dataD,'m-',linewidth=0.5,
                label=filename+' distorted')
        ax1.axhline(threshold,linestyle='-',
                    color='cyan',linewidth=1)
        ax1.axhline(-threshold,linestyle='-',
                    color='cyan',linewidth=1)
        ax1.legend(loc=1)
        ax1.set_ylabel('Amplitude (Rel. Bit)')
        ax1.set_xlabel('Time (s)')
    end=time.time()
    elapsed=int(1000*(end-start))
    print('Done!')
    print('...............................')
    print('Completed in '+str(elapsed)+' milliseconds.')        
    return dataD

def stereo(filename,time,wout=True):
    """
    Produces stereo effect. If file is mono, two channels are created and the
    R channel is delayed to simulate stereo width. Beware of phase issues.
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    time : scalar (ms)
        Amount of time the right channel is delayed by, in milliseconds.
      
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.
      
    Returns
    -------
    data_st: array containing the stereo waveform in normalized bits.
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    s_shift=int(sr*time*1E-3)
    R=np.zeros(n)
    L=np.zeros(n)
    if  ch==2:
        L[:]=data[:,0]
        R[:]=data[:,1]
    if ch==1:
        L[:]=data[:,0]
        R[:]=data[:,0]
    print('Applying stereo width...')
    for i in range (n-s_shift):
        R[i]=R[i+s_shift]
    data_st=np.zeros((n,2))
    data_st[:,0]=L[:]
    data_st[:,1]=R[:]
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_stereo.wav',data_st,44100,'PCM_16')
    print ('Done!')
    return data_st

def view(filename):
    """
    Plots waveform of input audio file. Note that every 100th data point is
    plotted for faster performance
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    t=np.linspace(0,n/sr,n)
    py.close()
    fig, (ax1) = py.subplots(nrows=1)  
    ax1.plot(t[0:n:100],data[0:n:100],'k-',linewidth=1,label=filename)
    ax1.legend(loc=1)
    ax1.set_ylabel('Amplitude (Rel. Bit)')
    ax1.set_xlabel('Time (s)')
    

def mono(filename,wout=True):
    """
    Converts a stereo track to mono.
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.
      
    Returns
    -------
    data_m: array containing the stereo waveform in normalized bits.
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    if  ch==2:
        print('Converting to mono...')
        L=data[:,0]
        R=data[:,1]
        n=len(data)
        data_m=np.zeros((n,1))
        data_m=L/2.0+R/2.0
        if wout==True:
            print('Exporting...')
            sf.write(filename[0:len(filename)-4]+'_mono.wav',data_m,sr,'PCM_16')
        print('Done!')
        return data_m
    else:
        print( "Error: input is already mono stoooooooooopid!")
        
        
def snip(filename,s,e,wout=True):
    """
    Converts a stereo track to mono.
    Parameters
    ----------
    filename : string
        Name of the input *.wav file.
    wout: True/False, optional, default=True
        Writes the data to a 16 bit *.wav file. Equating to false will suppress
        *.wav output, for example if you want to chain process without creating
        too many files.
      
    Returns
    -------
    data_m: array containing the stereo waveform in normalized bits.
    """
    n, data, data_dB,sr,ch=inputwav(filename)
    st=int(s*44100)
    en=int(e*44100)
    data_s=data[st:en,:]
    if wout==True:
        print('Exporting...')
        sf.write(filename[0:len(filename)-4]+'_snipped.wav',data_s,sr,'PCM_16')
    print('Done!')
    return data_s

def mix(f1,f2,r):
    n, data, data_dB,sr,ch=inputwav(f1)
    n1, data1, data_dB1,sr1,ch1=inputwav(f2)
    data_sum=r*data+(1-r)*data1
    sf.write(f1[0:len(f1)-4]+f2,data_sum,sr,'PCM_16')


def drakify(filename):
    slow(filename,wout=True)
    verb(filename[0:len(filename)-4]+'_slow.wav',5000,4,0.5)
    os.replace(filename[0:len(filename)-4]+'_slow_verbed.wav',filename[0:len(filename)-4]+'_drakify.wav')
    os.remove(filename[0:len(filename)-4]+'_slow.wav')
    #print(3)
    
def drakifyD(filename):
    slow(filename,wout=True)
    distort(filename[0:len(filename)-4]+'_slow.wav',0.1,type='flat')
    verb(filename[0:len(filename)-4]+'_slow_distort.wav',5000,4,0.5)
    os.replace(filename[0:len(filename)-4]+'_slow_distort_verbed.wav',filename[0:len(filename)-4]+'_drakifyD.wav')
    os.remove(filename[0:len(filename)-4]+'_slow.wav')
    #print(3)    
    
def drakifyL(filename):
    slow(filename,wout=True)
    limit(filename[0:len(filename)-4]+'_slow.wav',-20,7)
    verb(filename[0:len(filename)-4]+'_slow_limit.wav',5000,4,0.5)
    os.replace(filename[0:len(filename)-4]+'_slow_limit_verbed.wav',filename[0:len(filename)-4]+'_drakifyL.wav')
    os.remove(filename[0:len(filename)-4]+'_slow.wav')
    #print(3)       
  
def robot(filename,wout=True):
    verb(filename,1000,10,.000000009)
    os.replace(filename[0:len(filename)-4]+'_verbed.wav',filename[0:len(filename)-4]+'_robot.wav')
  
def choverb(filename,wout=True):
    verb(filename,100,1,.00000000001)
    os.replace(filename[0:len(filename)-4]+'_verbed.wav',filename[0:len(filename)-4]+'_choverb.wav')
  
def verberator(filename,wout=True):
    verb(filename,10000,3,.01)
    os.replace(filename[0:len(filename)-4]+'_verbed.wav',filename[0:len(filename)-4]+'_verberator.wav')
  
def robot2(filename,wout=True):
    verb(filename,500,10,.000001)
    os.replace(filename[0:len(filename)-4]+'_verbed.wav',filename[0:len(filename)-4]+'_robot2.wav')
  
def long(filename,wout=True):
    verb(filename,5000,20,.1)
    os.replace(filename[0:len(filename)-4]+'_verbed.wav',filename[0:len(filename)-4]+'_long.wav')
  
def telephone(filename,wout=True):
    bandpass(filename,300,3400,3)
    os.replace(filename[0:len(filename)-4]+'_BP.wav',filename[0:len(filename)-4]+'_telephone.wav')

def bluetoothspeaker(filename):
    normalize(filename)
    LPF(filename[0:len(filename)-4]+'_normalized.wav',400,Q=2)
    distort(filename[0:len(filename)-4]+'_normalized_LPF.wav',0.41,type='digital')
    os.replace(filename[0:len(filename)-4]+'_normalized_LPF_distort.wav',filename[0:len(filename)-4]+'_btspeaker.wav')
    os.remove(filename[0:len(filename)-4]+'_normalized_LPF.wav')
    os.remove(filename[0:len(filename)-4]+'_normalized.wav')
