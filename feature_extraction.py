# Section 5.3
# Contains the extraction process of 4 features used in Void system


import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as ssig
import scipy.stats as stats
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
import math
import librosa


def _stft(y):
    n_fft, hop_length, _ = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)


def _stft_parameters():
    # n_fft = (num_freq - 1) * 2
    n_fft = 2048
    hop_length = 128
    # hop_length = int(frame_shift_ms / 1000 * sample_rate)
    # win_length = int(frame_length_ms / 1000 * sample_rate)
    win_length = 512
    return n_fft, hop_length, win_length


def LinearityDegreeFeatures(power_normal):
    # Calculate signal power linearity degree features
    # input: power_normal
    # output: FV_LDF

    # Normalize power_vec as power_normal:
    #power_normal = power_vec / np.sum(power_vec)
    # From power_normal, calculate cumulative distribution of spectral power power_cdf:
    power_cdf = np.cumsum(power_normal)
    # Compute the correlation coefficients of power_cdf and store the results as rho:
    pearson_co = stats.pearsonr(power_cdf, np.arange(power_cdf.size))
    rho = pearson_co[0]
    #print("rho =", rho)
    # Compute the quadratic coefficients of power_cdf and store the results as q:
    #x_values = np.arange(0, 8+8/(power_cdf.size-1), 8/(power_cdf.size-1))
    x_values = np.arange(0, 8+7/(power_cdf.size-1), 8/(power_cdf.size-1))
    #parameter_2 = np.polyfit(x_values, power_cdf, 2)
    parameter_2 = np.polyfit(power_cdf, x_values, 2)
    q = parameter_2[0]
    #print("q =", q)
    # Form rho and q as FV_LDF:
    FV_LDF = np.array([rho, q])
    '''
    # Plot power_cdf and its estimation if necessary:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure()
    plt.plot(x_values, power_cdf, color='red', marker='d', label=r'$\textbf{pow}_\textbf{cdf}$')
    plt.plot(x_values, np.polyval(parameter_2, x_values), color='blue', 
        marker='o', label=r'$\textbf{fitting curve to pow}_\textbf{cdf}$')
    plt.xlabel(r'\textbf{Frequency (kHz)}', fontsize=16)
    plt.ylabel(r'$\textbf{pow}_\textbf{cdf}$', fontsize=16) 
    # The coordinate of starting point and end point of the arrow should be chosen manually:
    plt.annotate('rho = %.3f \n q = %.3f' % (rho, q), xy=(0.4, np.polyval(parameter_2, 0.4)-0.1), 
        xytext=(1, 0.3), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"), fontsize=12)
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()
    '''
    # Return power_cdf for plotting figures:
    return power_cdf, FV_LDF


def HighPowerFrequencyFeatures(FV_LFP, omega):
    # Calculate high power frequency 
    # input: FV_LFP, omega
    # output: FV_HPF
    
    # 1. Find peaks from FV_LFP (returns the indices of found peaks):
    peaks_idx, _ = find_peaks(FV_LFP, height=0)
    # Obtain corresponding values of the peaks:
    peaks_val = FV_LFP[peaks_idx]
    # 2. Compute the threshold of selecting peaks using omega:
    T_peak = omega * max(peaks_val)
    # 3. Remove peaks lower than T_peak (insignificant peaks):
    peaks_idx = peaks_idx[np.where(peaks_val >= T_peak)]
    peaks_val = FV_LFP[peaks_idx]
    # 4. Obtain the number of remaining peaks:
    N_peak = peaks_idx.size
    # 5. Compute the mean of the locations of remaining peaks:
    mu_peak = peaks_idx.mean()
    # 6. Compute the standard deviation of the locations of remaining peaks:
    sigma_peak = np.std(peaks_idx)
    # 7. Use a 6-order polynomial to fit FV_LFP and take first 32 estimatied values as P_est:
    parameter_6 = np.polyfit(np.arange(FV_LFP.size), FV_LFP, 6)
    value_est = np.polyval(parameter_6, np.arange(FV_LFP.size))
    '''
    plt.figure()
    plt.plot(np.arange(FV_LFP.size), FV_LFP, 'r')
    plt.plot(np.arange(FV_LFP.size), value_est, 'b')
    plt.show()
    '''
    P_est = value_est[0:32]
    # Construct FV_HPF (insert N_peak, mu_peak and sigma_peak in fornt of P_est):
    FV_HPF = np.insert(P_est, 0, [N_peak, mu_peak, sigma_peak])
    return FV_HPF


def lpc_to_lpcc(lpc):
    # Based on given LPC, calculate LPCC:
    lpcc = []
    order = lpc.size - 1
    # The 1st element equals ln(order):
    lpcc.append(math.log(order))
    lpcc.append(lpc[1])
    for i in range(2, order+1):
        sum_1 = 0
        for j in range(1, i):
            sum_1 += j / i * lpc[i-j-1] * lpcc[j]
        c = -lpc[i-1] + sum_1
        lpcc.append(c)
    return lpcc[1:13]


def extract_lpcc(wav_path, order):
    y, _ = librosa.load(wav_path, sr=16000)
    lpc = librosa.lpc(y, order)
    lpcc = np.array(lpc_to_lpcc(lpc))
    return lpcc


# https://www.cnblogs.com/klchang/p/9280509.html
def calc_stft(signal, sample_rate=16000, frame_size=512, frame_stride=128, winfunc=np.hamming, NFFT=2048):

    # Calculate the number of frames from the signal
    signal_length = len(signal)
    frame_step = frame_stride
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_size)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_size
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples 
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_size)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
    '''
    # Calculate the number of frames from the signal
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples 
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + 
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
    '''
    return pow_frames
