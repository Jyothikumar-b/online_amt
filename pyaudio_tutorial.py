from transcribe import load_model, OnlineTranscriber

import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use('Qt5Agg')
import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
import argparse
from multiprocessing import Process
from mic_stream import MicrophoneStream

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output.wav"



def main(args):
    stream = MicrophoneStream(RATE, CHUNK, CHANNELS)
    model = load_model(args)
    transcriber = OnlineTranscriber(model, return_roll=False)

    piano_roll = np.zeros((88, 32))
    piano_roll[30, 0] = 1
    entire_frames = []
    plt.ion()
    fig, ax = plt.subplots()

    x = np.arange(0, 2* CHUNK,2)
    plt.show(block=False)
    img = ax.imshow(piano_roll)
    ax_background = fig.canvas.copy_from_bbox(ax.bbox)
    ax.invert_yaxis()
    fig.canvas.draw()
    ONSETS = []

    with MicrophoneStream(RATE, CHUNK, 1) as stream:
        # 마이크 데이터 핸들을 가져옴 
        audio_generator = stream.generator()
        print("* recording")        
        for i in range(1000):
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            time_a = time.time()
            # frame_output = transcriber.inference(decoded)
            onset, offset = transcriber.inference(decoded)
            time.sleep(1)
            # ONSETS += onset
            # print(time.time() - time_a)
            # print(ONSETS)
            # print(frame_output)
            # new_roll = np.zeros_like(piano_roll)
            # new_roll[:, :-1] = piano_roll[:,1:]
            # new_roll[:, -1] = frame_output
            # piano_roll = new_roll

            # # time_b = time.time()
            # # fig.canvas.restore_region(ax_background)
            # # img.set_data(piano_roll)
            # # ax.draw_artist(img)
            # # fig.canvas.blit(ax.bbox)
            # # fig.canvas.flush_events()
            # # time_c = time.time()
            # print(time_c-time_b, time_b-time_a)
        stream.closed = True
    print("* done recording")


    # librosa.output.write_wav('lib_out.wav', np.concatenate(entire_frames), sr=44100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='/Users/1112919/Documents/ar_model_weights/model-210000.pt')
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--n_class', default=5, type=int)
    parser.add_argument('--ac_model_type', default='simple_conv', type=str)
    parser.add_argument('--lm_model_type', default='lstm', type=str)
    parser.add_argument('--context_len', default=1, type=int)
    parser.add_argument('--no_recursive', action='store_true')
    args = parser.parse_args()

    main(args)