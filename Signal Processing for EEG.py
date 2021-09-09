# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:27:42 2021

@author: tdang9
"""

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft
import scipy.fftpack
from scipy.signal import chirp, find_peaks, peak_widths

#Copy and Paste Address location of File | There is Code to automate this using a GUI
df = pd.read_csv (r'C:/Users/thoma/Desktop/ARL/EEG/session/1/eeg_studyLevel2_ARL_RWNVDEDP_session_1_task_RealworldDriving_subjectLabId_3032_ARL_CANCTA_RWNVD_021_R1_EEG_D_REC1_uVj_recording_1.csv')
times = df.Time


print(times)
FinalIndex = len(df.index) - 1  #sample starts at index 1
print(FinalIndex)
MaxTime = df.iloc[FinalIndex, 0] #End time of experiment (milliseconds)
print(MaxTime)

#P300 Indexing
f = 256 # (Hz) Sampling Rate of ABM B-Alert x24 **Not constant across experiments

EventStartTime = 2723.0156 #Seconds 
EventEndTime = EventStartTime + .650 #Seconds
#EventEndTime = 

StartIndex = round(EventStartTime*1000/(1000/f))
EndIndex = round(((EventEndTime*1000))/(1000/f)) 

print(StartIndex)
print(EndIndex)

#Channels (Assign Variables to each electode)
channel1 = (df.iloc[StartIndex:EndIndex, 1:20]) #rows, columns Selects range of index values
#channel1 = df.F3 #Assigning Channels to variables
channel2 = df.F1
channel3 = df.Fz
channel4 = df.F2
channel5 = df.F4
channel6 = df.C3
channel7 = df.C1
channel8 = df.Cz
channel9 = df.C2
channel10 = df.C4
channel11 = df.CPz
channel12 = df.P3
channel13 = df.P1
channel14 = df.Pz
channel15 = df.P2
channel16 = df.P4
channel17 = df.POz
channel18 = df.O1
channel19 = df.Oz
channel20 = df.O2


N = EndIndex-StartIndex #Number of Samples
T= (1/f) #Period between Samples
print(N)


#create x-axis for time length of signal
x = (np.linspace(0, N*T, N)) + EventStartTime
#create array that corresponds to values in signal
y = channel1
#perform FFT on signal
yf = fft(y)
#create new x-axis: frequency from signal
xf = (np.linspace(0.0, 1.0/(2.0*T), N//2)) 

#Generating PSD
psd = np.abs(yf) ** 2
xpsd = fftfreq(len(psd),T)
i = xpsd > 0
#Generating Frequency Bands
# Define EEG bands
eeg_bands = {'Delta': (0.1, 3),
             'Theta': (4, 7),
             'Alpha': (8, 15),
             'Beta': (16, 30),
             'Gamma': (31, 100)}

# Take the mean of the fft amplitude for each EEG band
eeg_band_fft = dict()
for band in eeg_bands:  
    freq_ix = np.where((xpsd >= eeg_bands[band][0]) & 
                       (xpsd <= eeg_bands[band][1]))[0]
    eeg_band_fft[band] = np.mean(psd[freq_ix])
     
def calc_bands_power(x, dt, bands):
    from scipy.signal import welch
    f, psd = welch(x, fs=1. / dt)
    power = {band: np.mean(psd[np.where((f >= lf) & (f <= hf))]) for band, (lf, hf) in bands.items()}
    return power

#Waveform Parameters

#Max Amplitude

my_array = channel1.to_numpy()
print(len(my_array))
amp = (np.max(my_array))
max = str(amp)
print('The maximum amplitude = ' + max) 


#AmpIndex = df.loc[df['F3'] == amp].index[0]
AmpIndex = False 
i = -1

while (AmpIndex == False):
    i = i + 1
    AmpIndex = np.any(my_array[i] == amp)
   
AmpIndex = i

print('The Index of the amplitude = ' + str(AmpIndex)) 

#PulseWidth
limit = 0.02 * amp

StartWidth = False
i2 = -1

while (StartWidth == False):
    i2 = i2 + 1
    StartWidth = np.any(my_array[i2] >= limit)
    
StartWidth = i2
#StartWidth = df.loc[df['F3'] >= limit].index[0]
print('The Start Index of the Start Width = ' + str(StartWidth))


EndWidth = False
i3 = i

while (EndWidth == False):
    i3 = i3 + 1
    EndWidth = np.any(my_array[i3] <= limit)
EndWidth = i3  
print('The Start Index of the End Width = ' + str(EndWidth))

EndWidth = (df.iloc[EndWidth, 0])
StartWidth = (df.iloc[StartWidth, 0])

PulseWidth = EndWidth - StartWidth
print(StartWidth)
print(EndWidth)
print(PulseWidth)

#AmpIndex[0] = int(AmpIndex)
#scipy.signal.peak_widths(x, AmpIndex, rel_height=0.5, prominence_data=None, wlen=None)[source]
#R,C = np.where(np.triu(y,1)>=limit)

#Plots

#Original Signal
plt.figure(1)
plt.plot(x,y, label = 'F3')
plt.title('Original Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend(loc=1)
plt.grid()
#FFT of Signal
plt.figure(2)
plt.plot(xf, yf[0:N//2], label = 'F3')
#plt.ylim(0, 500)
plt.title('FFT Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
#Power Spectral Density
plt.figure(3)
plt.plot(xpsd[i], (10 * np.log10(psd[i])), label = 'F3')
plt.xlim(0, 130)
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)')
plt.legend()
plt.grid()
#Frequency Bands
plt.figure(4)
df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")
print(df)
plt.grid()
plt.show()



#697092
#697258
#166
#166
#The maximum amplitude = 9.5426
#The Index of the amplitude = 697191