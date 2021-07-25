import torch as th
import numpy as np

class MobileModel:
    def __init__(self, return_roll=True):
        self.melspectrogram = th.load("melspectoram.pth")
        self.actoustic_model = th.jit.load("acoustic_model.pth")
        self.language_model = th.jit.load("language_model.pth")
        self.language_post = th.jit.load("language_post.pth")
        self.class_embedding = th.load("embedding_layer.pth")

        self.audio_buffer = th.zeros((1,5120)).to(th.float)
        self.mel_buffer = self.melspectrogram(self.audio_buffer)
        # complexity_conv = 64
        # complexity_lstm = 32
        h = th.zeros(2, 1, 512, device=th.device('cpu'))
        c = th.zeros(2, 1, 512, device=th.device('cpu'))
        self.hidden = (h, c)

        self.prev_output = th.zeros((1,1,88)).to(th.long)
        self.buffer_length = 0
        self.sr = 16000
        self.return_roll = return_roll
        
        self.inten_threshold = 0.05
        self.patience = 100
        self.num_under_thr = 0
        
        
    def lm_model_step(self, acoustic_out, hidden, prev_out):
        prev_embedding = self.class_embedding(prev_out).view(acoustic_out.shape[0], 1, 88*2)
        current_data = th.cat((acoustic_out, prev_embedding), dim=2)
        current_out, hidden = self.language_model(current_data, hidden)
        current_out = self.language_post(current_out)
        current_out = current_out.view((acoustic_out.shape[0], 1, 88, 5))
        current_out = th.softmax(current_out, dim=3)
        return current_out, hidden

    def update_buffer(self, audio):
        t_audio = th.tensor(audio).to(th.float)
        new_buffer = th.zeros_like(self.audio_buffer)
        new_buffer[0, :-len(t_audio)] = self.audio_buffer[0, len(t_audio):]
        new_buffer[0, -len(t_audio):] = t_audio
        self.audio_buffer = new_buffer

    def update_mel_buffer(self):
        self.mel_buffer[:,:,:6] = self.mel_buffer[:,:,1:7]
        self.mel_buffer[:,:,6:] = self.melspectrogram(self.audio_buffer[:, -2048:])

    def update_acoustic_out(self, mel):
        return self.actoustic_model(mel)
        
    def switch_on_or_off(self):
        pseudo_intensity = th.max(self.audio_buffer) - th.min(self.audio_buffer)
        if pseudo_intensity < self.inten_threshold:
            self.num_under_thr += 1
        else:
            self.num_under_thr = 0
    	
    def inference(self, audio):
        with th.no_grad():
            self.update_buffer(audio)
            self.switch_on_or_off()
            if self.num_under_thr > self.patience:
                if self.return_roll:
                    return [0]*88
                else:
                    return [], []
            self.update_mel_buffer()

            # # model 1
            # print("actual : model 1 in = ",sum(self.mel_buffer))
            acoustic_out = self.update_acoustic_out(self.mel_buffer.transpose(-1, -2))
            # acoustic_out = self.update_acoustic_out(audio.transpose(-1, -2))
            # model 2
            language_out, self.hidden = self.lm_model_step(acoustic_out, self.hidden, self.prev_output)
            
            language_out[0,0,:,3:5] *= 2
            self.prev_output = language_out.argmax(dim=3)
            out = self.prev_output[0,0,:].numpy()
        
        if self.return_roll:
            return (out == 2) + (out == 3)
            
        else: # return onset and offset only
            out[out==4]=3
            onset_pitches = np.squeeze(np.argwhere(out == 3)).tolist()
            off_pitches = np.squeeze(np.argwhere(out == 1)).tolist()
            if isinstance(onset_pitches, int):
                onset_pitches = [onset_pitches]
            if isinstance(off_pitches, int):
                off_pitches = [off_pitches]
            # print('after', onset_pitches, off_pitches)
            return onset_pitches, off_pitches

