import os
import glob
import torch

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

def create_lj50_samples(model_id):
    uc = UtteranceCloner(model_id=model_id, device="cuda:7" if torch.cuda.is_available() else "cpu")
    
    output_dir = f"audios/uc_lj50_{model_id}"
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    txt_file_paths = sorted(glob.glob("/mount/resources/speech/corpora/LJSpeech/16kHz/txt/LJ050-*.txt"))
    wav_file_paths = sorted(glob.glob("/mount/resources/speech/corpora/LJSpeech/16kHz/wav/LJ050-*.wav"))

    for t_f, w_f in zip(txt_file_paths, wav_file_paths):

        base_file_path = t_f

        with open(base_file_path, 'r', encoding='utf8') as base_tf:
            transcript = base_tf.read()
        
        base_wav_path = w_f
        
        uc.clone_utterance(
            path_to_reference_audio=base_wav_path,
            reference_transcription=transcript,
            filename_of_result=output_dir + "/" + w_f.split("/")[-1],
            clone_speaker_identity=True,
            lang="en")

if __name__ == '__main__':
    create_lj50_samples(model_id="LJSpeech_FrozenUnivGAN_SG_0_001")