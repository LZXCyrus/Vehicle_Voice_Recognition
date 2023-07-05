import os
import wave
import pylab
import struct

def graph_spectrogram(wav_file,figure_name):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(0.84, 0.84))
    pylab.subplot(111) 
    pylab.axis('off')
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(figure_name,transparent=True)
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


#d = librosa.get_duration(y=x, sr=22050, S=None, n_fft=2048, hop_length=512, center=True, filename=None)
import librosa
#d = librosa.get_duration(y=x, sr=22050, S=None, n_fft=2048, hop_length=512, center=True, filename=None)
train_dir = 'C:/Users/11475/Desktop/jstj/sounds/ambulance'

train_files = [x for x in os.listdir(train_dir) if x.endswith('.wav')]
x ,sr =librosa.load(train_files[50])
import matplotlib.pyplot as plt
import librosa.display
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(20, 4))
librosa.display.waveshow(y=x, sr=sr,color='#1F77B4')
plt.figure(figsize=(5, 20))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()
import sklearn
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(5, 20))
# librosa.display.waveshow(y=x, sr=sr,color='#9AC9DB')
# plt.show()
mfcc_feat = librosa.feature.mfcc(y = x, sr = sr, n_mfcc = 13)

librosa.display.specshow(mfcc_feat, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar()
plt.show()
#print(train_files)
# for i in range(len(train_files)):
#     sr=librosa.get_samplerate(train_files[i])
#     #d = librosa.get_duration(y=x, sr=22050, S=None, n_fft=2048, hop_length=512, center=True, filename=None)
#     print(sr)
