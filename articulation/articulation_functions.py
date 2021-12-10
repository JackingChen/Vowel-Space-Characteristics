
import math
import numpy as np
import sys
import pysptk
from parselmouth.praat import call
import parselmouth 
import pandas as pd

def bark(f):
	x=(f*0.00076)
	x2=(f/7500)**2
	b=[]
	for i in range (0,len(f)):
		b.append(13*( math.atan(x[i]) )+3.5*( math.atan(x2[i]))) #Bark scale values
	return (b)

def barke(x,Fs, nfft=2048, nB=25):
		"""
		e: Energy in frequency bands according to the Bark scale
		x: signal
		Fs: sampling frequency
            nfft: number of points for the Fourier transform
            nB: number of bands for energy computation
		"""
		eps = 1e-30
		y = fftsolp(x,nfft)
		f = (Fs/2)*(np.linspace(0,1,int(nfft/2+1)))
		barkScale = bark(f)
		barkIndices = []
		for i in range (0,len(barkScale)):
			barkIndices.append(int(barkScale[i]))

		barkIndices = np.asarray(barkIndices)

		barkEnergy=[]
		for i in range (nB):
			brk = np.nonzero(barkIndices==i)
			brk = np.asarray(brk)[0]
			sizeb=len(brk)
			if (sizeb>0):
				barkEnergy.append(sum(np.abs(y[brk]))/sizeb)
			else:
				barkEnergy.append(0)


		e = np.asarray(barkEnergy)+eps
		e = np.log(e)
		return e


def fftsolp(x,nfft):
     """
     STFT for compute the energy in Bark scale
     x: signal
     nffft: number of points of the Fourier transform
     """
     window = np.hamming(len(x)/4)
     noverlap = np.ceil(len(window)/2)

     nx = len(x)
     nwind = len(window)

     ncol = np.fix((nx-noverlap)/(nwind-noverlap))
     ncol = int(ncol)
     colindex = (np.arange(0,ncol))*(nwind-noverlap)
     colindex = colindex.astype(int)

     rowindex = np.arange(0,nwind)
     rowindex = rowindex.astype(int)
     rowindex = rowindex[np.newaxis]
     rowindex = rowindex.T

     y = np.zeros((nwind,ncol),dtype=np.int)
     d = np.ones((nwind,ncol),dtype=np.int)


     y = x[d*(rowindex+colindex)]
     window = window.astype(float)
     window = window[np.newaxis]
     window = window.T
     new = window*d
     y = new*y
     y = y[:,0]

     y = np.fft.fft(y,nfft)
     y = (y[0:int(nfft/2+1)])
     return y




def extractTrans(segments, fs, size_frameS, size_stepS, nB=22, nMFCC=12, nfft=2048):
    frames=[]
    size_frame_full=int(2**np.ceil(np.log2(size_frameS)))
    fill=int(size_frame_full-size_frameS)
    overlap=size_stepS/size_frameS
    for j in range(len(segments)):
        if (len(segments[j])>size_frameS):
            nF=int((len(segments[j])/size_frameS)/overlap)-1
            for iF in range(nF):
                frames.append(np.hamming(size_frameS)*segments[j][int(iF*size_stepS):int(iF*size_stepS+size_frameS)])

    BarkEn=np.zeros((len(frames),nB))
    MFCC=np.zeros((len(frames),nMFCC))
    for j in range(len(frames)):
        frame_act=np.hstack((frames[j], np.zeros(fill)))
        BarkEn[j,:]=barke(frame_act,fs, nfft, nB)
        MFCC[j,:]=pysptk.sptk.mfcc(frame_act, order=nMFCC, fs=fs, alpha=0.97, num_filterbanks=32, cepslift=22, use_hamming=True)
    return BarkEn, MFCC


def V_UV(F0, data_audio, fs, transition, size_tran=0.04):
    segment=[]
    time_stepF0=int(len(data_audio)/len(F0))
    #print(F0)
    for j in range(2, len(F0)):
        if transition=='onset':
            if F0[j-1]==0 and F0[j]!=0:
                border=j*time_stepF0
                initframe=int(border-size_tran*fs)
                endframe=int(border+size_tran*fs)
                segment.append(data_audio[initframe:endframe])
        elif transition=='offset':
            if F0[j-1]!=0 and F0[j]==0:
                border=j*time_stepF0
                initframe=int(border-size_tran*fs)
                endframe=int(border+size_tran*fs)
                segment.append(data_audio[initframe:endframe])
    return segment


def measureFormants(sound, f0min,f0max, time_step=0.0025,MaxnumForm=5,Maxformant=5000,framesize=0.025):
    sound = parselmouth.Sound(sound) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", time_step, MaxnumForm, Maxformant, framesize, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']

    return f1_list, f2_list

def measurePitch(voiceID, f0min, f0max, unit):
    columns=['duration', 'intensity_mean', 'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'aqpq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    if duration >=0.064:
        intensity = sound.to_intensity()
        intensity_mean=np.mean(intensity.values)
    else:
        intensity_mean=-1
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    df=pd.DataFrame(np.array([duration, intensity_mean, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]),index=columns)
    return df.T
