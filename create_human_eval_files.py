import os
import glob
import shutil
import torch

import soundfile
from Utility.utils import float2pcm

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from InferenceInterfaces.InferenceArchitectures.InferenceBigVGAN import BigVGAN

def create_lj_audio_folder():
    
    output_dir = f"audios/lj_50_eval_samples"
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    for file_num in ["112", "181", "174", "126", "165", "072"]:
        
        wav_path = f"/mount/resources/speech/corpora/LJSpeech/16kHz/wav/LJ050-0{file_num}.wav"
        dst_path = output_dir + f"/LJ050-0{file_num}.wav"

        shutil.copy(wav_path, dst_path)

def create_lj50_audio_folder():
    
    output_dir = f"audios/lj50_GT_samples"
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    folder_path = "/mount/resources/speech/corpora/LJSpeech/16kHz/wav/LJ050-*.wav"
    list_of_files = glob.glob(folder_path)

    for f in list_of_files:
        
        wav_path = f
        dst_path = output_dir + "/" + f.split("/")[-1]

        shutil.copy(wav_path, dst_path)

def create_lj_vc_audio_folder():
    
    output_dir = f"audios/lj_50_vc_eval_samples"
    vocoder_model_path = "/mount/arbeitsdaten56/projekte/synthesis/srivasvv/thesis-playground/tts-postnet-eval/IMS-Toucan/Models/Avocodo/BigVGAN.pt"
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    mel2wav = BigVGAN(path_to_weights=vocoder_model_path).to(torch.device("cuda:0"))
    mel2wav.remove_weight_norm()
    mel2wav.eval()        
    
    l = ["112", "181", "174", "126", "165", "072"]

    for file_num in ["112", "181", "174", "126", "165", "072"]:
        
        wav_path = f"/mount/resources/speech/corpora/LJSpeech/16kHz/wav/LJ050-0{file_num}.wav"
        file_location = output_dir + f"/LJ050-0{file_num}.wav"
        
        wav, sr = soundfile.read(wav_path)

        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True)

        mel = ap.audio_to_mel_spec_tensor(audio=wav)
        wav = mel2wav(mel.to(torch.device("cuda:0")))

        wav = [val for val in wav.detach().cpu().numpy() for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
        soundfile.write(file=file_location, data=float2pcm(wav), samplerate=48000, subtype="PCM_16")


if __name__ == '__main__':
    create_lj50_audio_folder()