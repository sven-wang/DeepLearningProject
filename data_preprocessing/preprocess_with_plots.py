"""
Reference:
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from matplotlib import cm


start_second = 10
end_second = 20

# sample_file = 'sample_data.wav'
# sample_file = '100304-f-sre2006-kacg-A-vad.wav'
sample_file = '../TAs/Jerry.wav'


sample_rate, signal = scipy.io.wavfile.read(sample_file)  # File assumed to be in the same directory
signal = signal[int(start_second * sample_rate): int(end_second * sample_rate)]  # Keep the first 3.5 seconds

print('sample_rate', sample_rate)

# Plot raw signals
times = numpy.arange(len(signal)) / float(sample_rate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times, signal, color='k')
plt.xlim(times[0], times[-1])

plt.ylim(-4000, 4000)
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig('plot.png', dpi=100)
# plt.show()


# Pre-Emphasis: apply a pre-emphasis filter on the signal to amplify the high frequencies
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# Framing: split the signal into short-time frames
frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]

# Window
frames *= numpy.hamming(frame_length)
# frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **


# Fourier-Transform and Power Spectrum
NFFT = 512

mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

print(pow_frames.shape)

fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
cax = ax.matshow(numpy.transpose(pow_frames), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
fig.colorbar(cax)
plt.title('Original Spectrogram')
plt.show()

# Filter Banks: applying triangular filters on a Mel-scale to the power spectrum to extract frequency bands
nfilt = 40
low_freq_bound = 500
high_freq_bound = 3000

low_freq_mel = (2595 * numpy.log10(1 + low_freq_bound / 700))
high_freq_mel = (2595 * numpy.log10(1 + high_freq_bound / 700))  # Convert Hz to Mel
# high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

# applying the filter bank to the power spectrum (periodogram) of the signal
fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB

"""
# Mel-frequency Cepstral Coefficients (MFCCs)

num_ceps = 12

mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

cep_lifter =22

(nframes, ncoeff) = mfcc.shape
n = numpy.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
mfcc *= lift  #*
"""

# Mean Normalization

filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)

# mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

ig, ax = plt.subplots(figsize=(30, 4))
mfcc_data= numpy.swapaxes(filter_banks, 0, 1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.rainbow, origin='lower', aspect='auto')
ax.set_title('Filter Banks')
#Showing mfcc_data
plt.show()
plt.savefig('foo.png')



