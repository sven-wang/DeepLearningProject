"""
Reference:
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""
import numpy
import scipy.io.wavfile
import os


def filterbank(sample_file, output_file):
    sample_rate, signal = scipy.io.wavfile.read(sample_file)  # File assumed to be in the same directory

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
    num_frames = int(numpy.ceil(
        float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal,
                              z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    # Window
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    # Fourier-Transform and Power Spectrum
    NFFT = 512

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Filter Banks: applying triangular filters on a Mel-scale to the power spectrum to extract frequency bands
    nfilt = 40  # 63
    low_freq_bound = 20  # 20
    high_freq_bound = 3600  # 3600

    low_freq_mel = (2595 * numpy.log10(1 + low_freq_bound / 700))
    high_freq_mel = (2595 * numpy.log10(1 + high_freq_bound / 700))  # Convert Hz to Mel
    # high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    # applying the filter bank to the power spectrum (periodogram) of the signal
    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    # Mean Normalization

    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)

    # fix time dimension to 30000
    max_len = 6000

    # discard audios shorter than 30 seconds
    if filter_banks.shape[0] < 100:
        print('%s too short, discarded' % wav_file)
        return

    # print(filter_banks.shape)

    # keep appending duplicates if shorter than max_len
    original = filter_banks
    while filter_banks.shape[0] < max_len:
        filter_banks = numpy.concatenate((filter_banks, original), axis=0)

    # print(filter_banks.shape)

    if filter_banks.shape[0] > max_len:
        filter_banks = filter_banks[:max_len, :]

    # print(filter_banks.shape)

    numpy.save(output_file, filter_banks)


if __name__ == '__main__':

    # TODO: change wav dir
    wav_dir = '../TAs/'
    files = os.listdir(wav_dir)

    for wav_file in files:
        print(wav_file)
        if not wav_file.endswith('.wav'):
            continue

        # TODO: change feature dir
        output_file = '../TAs/' + wav_file.split('.')[0]

        if os.path.exists(output_file + '.npy'):
            continue

        try:
            filterbank(wav_dir + wav_file, output_file)
        except:
            print('failed to parse %s' % wav_file)
