# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:50:15 2023

@author: ShuttleBox
"""

import tkinter
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np


# basic Tkinter settings
root = tkinter.Tk()
# root.withdraw()
plt.ioff()

# Pick file with EOD recording
fname = filedialog.askopenfilename(title = "Select File wir EOD Recordings", filetypes = (("Recording", "*.csv"), ("All files", "*")))

# Read EOD data
df = pd.read_csv(fname, sep=";", decimal=",")

# Recording settings
fs = 20000

f, t, Sxx = signal.spectrogram(df['ch 0'], fs, nperseg = 2**12)

# FFT
y = np.array(df['ch 0'])
# The fourier transform of y:
yf=fft(y, norm='forward')
# Note: see  help(fft) --> norm. I chose 'forward' because it gives the amplitudes we put in.
# Otherwise, by default, yf will be scaled by a factor of n: the number of points

# The frequency scale
n = y.size   # The number of points in the data
freq = fftfreq(n, d=1/fs)

pos_vals = np.where(freq>0)

freq = freq[pos_vals]
yf = yf[pos_vals]

# Let's find the peaks with height_threshold >=0.05
# Note: We use the magnitude (i.e the absolute value) of the Fourier transform

height_threshold=0.05 # We need a threshold. 


# peaks_index contains the indices in x that correspond to peaks:
peaks_index, properties = signal.find_peaks(np.abs(yf), height=height_threshold)



# Create plot
fig = plt.figure(figsize = (7,7))
voltage = fig.add_subplot(3,1,1)
voltage.set_title("Voltage")
voltage.plot(df['Time [ms]']/1000, df['ch 0'])
voltage.set_xlabel('Time [sec]')
voltage.set_ylabel('Voltage [V]')

spec = fig.add_subplot(3,1,2)
spec.pcolormesh(t, f, Sxx, shading='gouraud')
spec.set_xlabel('Time [sec]')
spec.set_ylabel('Frequency [Hz]')
spec.set_ylim(100,2000)

power_sd = fig.add_subplot(3,1,3)
power_sd.plot(freq, np.abs(yf),'-', freq[peaks_index],properties['peak_heights'],'x')
for i in range(len(peaks_index)):
    power_sd.text(freq[peaks_index][i], properties['peak_heights'][i], str(round(freq[peaks_index][i])))
power_sd.set_xlabel('Frequency [Hz]')
power_sd.set_ylabel('Amplitude')
power_sd.set_xlim(0,2000)

plt.show()




