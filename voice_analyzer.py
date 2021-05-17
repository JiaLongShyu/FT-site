import numpy as np
import pandas as pd

import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.fft  import rfft,fft
from scipy.signal import hann
import math


from math import ceil

# The parameters
filename_df = []
sinusitis_df = []
VLHR_window_df = []
VLHR_LTAS_df = []
DB_df = []
filename = 'test.wav'

def analyze(file, filename, window_time, noise_percentile, VLHR_threshold = 300):
    sr, y_original = scipy.io.wavfile.read(file)
    y = y_original 
    y = y[:sr*10]
    duration = y.shape[0] / sr
    
    """
    print(duration)
    print(y_original.shape)
    print(y.shape)
    """
    
    raw_mean = np.mean(y)
    print(f'raw_mean = {raw_mean}')
    y = y - raw_mean
    mean_check = np.mean(y)
    print(f'mean_check= {mean_check}')

    #sr = 22050
    #window_time = 0.02
    noise_time = 0.1*duration
    window_sample_number= round(sr*window_time)
    n_fft = round(sr*window_time)
    #VLHR_threshold = 300

    def section(title = 'title'):
                print()
                print( '*' * 30)
                print(title)
                print( '*' * 30)
    print(duration)

    # Windowing of the signal

    # pad the sequence so that the windows are equally sized
    y = np.pad(y, (0, ceil(sr*duration/window_sample_number)*window_sample_number - y.size), 'constant', constant_values=(0, 0))/30000
    print(y)
    print(y.size)
    y_framed = np.reshape(y, (ceil(sr*duration/window_sample_number), window_sample_number))
    print(f'y_framed.shape is {y_framed.shape}')

    y_rms = np.sqrt(np.mean(y_framed**2, axis=1))

    # Noise by pecentile
    noise_percentile_value = np.percentile(y_rms, noise_percentile)
    threshold = 3.16*noise_percentile_value
    print(f'PR threshold = {threshold}')
    plt.hist(y_rms, bins=1000)  # arguments are passed to np.histogram
    plt.title("Histogram with 1000 bins")
    plt.show()

    plt.figure(figsize=(20,6))
    #librosa.display.waveplot(y, color='b', alpha=0.5, x_axis='s')
    plt.title(f'Original Wave Plot with Threshold')
    plt.plot(y)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.ylim(bottom=0)
    plt.show()

    print(f'threshold = {threshold}')

    y_rms_sorted = np.sort(y_rms)

    plt.title('Original per window RMS plot')
    plt.plot(y_rms_sorted)
    plt.show()

    # Noise Reduction
    null_window_holder = []
    for i in range(0,y_framed[:,0].size):
        if (y_rms[i] < threshold):
            y_framed[i,:] = 0
            null_window_holder = np.append(null_window_holder, int(i))

    y_framed = np.delete(y_framed, null_window_holder.astype(int), axis=0)

    np.set_printoptions(suppress=True, precision=3)

    # Par window normalization
    y_denoised_rms = np.sqrt(np.mean(y_framed**2, axis=1))
    w = hann(y_framed[0,:].size)

    for i in range(0,y_framed[:,0].size):
        if (y_denoised_rms[i] !=0):
            y_framed[i,:] = y_framed[i,:]*w
            y_denoised_rms[i] = np.sqrt(np.mean(y_framed[i,:]**2, axis=0))
            y_framed[i,:] = y_framed[i,:]/y_denoised_rms[i]
    print(y_framed.shape)
    print((y_denoised_rms!=0).argmax(axis=0))
    #print(y_denoised_rms)

    """
    y_framed_rms_check = np.sqrt(np.mean(y_framed**2, axis=1))
    print(y_framed_rms_check)
    print((y_framed_rms_check!=0).argmax(axis=0))
    """

    voice_window_number = np.count_nonzero(y_denoised_rms)

    # Print the denoised wave plot
    plt.figure(figsize=(20,6))
    plt.plot(y)
    plt.title('Normalized  Denoised Wave Plot')
    plt.ylim(bottom=0) 
    plt.show()

    # Short Time Fourier Transform
    y_FFT = []
    print(y_framed.shape)
    print(y_framed[0,:].shape)
    for i in range(0,voice_window_number):
        y_FFT = np.append(y_FFT,rfft(y_framed[i,:]), axis = 0)
    print(y_FFT.shape)
    y_FFT = np.reshape(y_FFT,(y_framed[:,0].size, rfft(y_framed[0,:]).size))
    print(y_FFT.shape)
    #

    # The normalization
    y_FFT = y_FFT/np.sqrt(window_sample_number**2/2)

    """
    section(f'Total power check')
    T = np.sum(np.abs(y_FFT)**2, axis=1)
    print('Total power')
    print(T)
    """

    # VLHR_window calculation
    section('VLHR_window calculation')
    VLHR_each_window = np.empty(y_FFT[:,0].size)
    for i in range(0,y_FFT[:,0].size):
        low_power = np.sum(np.abs(y_FFT[i,0:ceil(VLHR_threshold*window_time)])**2)
        high_power = np.sum(np.abs(y_FFT[i,ceil(VLHR_threshold*window_time):])**2)
        VLHR_each_window[i] = low_power/high_power
    VLHR_window_db = np.average(10*np.log10(VLHR_each_window))
    VLHR_window_df.append(VLHR_window_db)
    print(f'VLHR_window(dB) = {VLHR_window_db}')


    # The mean power over all windows for each frequency
    y_power = np.mean(np.abs(y_FFT**2), axis=0)

    # Long term average spectrum calculation
    section('Long term average spectrum')
    plt.title(f'LTAS(power)')
    plt.plot(y_power)
    plt.show()

    DB = np.zeros(y_power.size)
    for i in range(0,y_power.size):
        DB[i] = 10*math.log10(y_power[i])

    plt.title(f'LTAS(dB)')
    plt.ylim(-60,0) 
    frequency_axis = np.arange(DB.size)/window_time
    plt.plot(frequency_axis, DB)
    plt.show()
    print(f'Maximum decibel = {np.amax(DB)}')

    #VLHR calculation
    section('VLHR_LTAS calculation')
    low_power = np.sum(y_power[0:ceil(VLHR_threshold*window_time)])
    high_power = np.sum(y_power[ceil(VLHR_threshold*window_time):])
    VLHR_LTAS_db = 10*math.log10(low_power/high_power)
    print(f'VLHR_LTAS(dB) = {VLHR_LTAS_db}')
    print(f'VLHR_window(dB) = {VLHR_window_db}')

    # Result display
    section('Parameters')
    print(f'Filename = {filename}')
    print(f'duration = {duration} sec')
    print(f'noise_PR = {noise_percentile}')
    print(f'window_sample_number = {window_sample_number}')
    print(f'window_time= {window_time} sec')
    print()
    print(f'Total_window_number = {round(duration/window_time)}')
    print(f'voice_window_number = {voice_window_number}')