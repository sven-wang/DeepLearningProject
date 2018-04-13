from vad import VoiceActivityDetector

filename = '../data_preprocessing/100396-m-sre2008-fkffz-A.wav'
v = VoiceActivityDetector(filename)
v.plot_detected_speech_regions()