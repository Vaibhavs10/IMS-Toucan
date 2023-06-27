import os
import torch

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

def create_mos_survey_samples(model_id):
    uc = UtteranceCloner(model_id=model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = f"audios/MOS_Survey_{model_id}"
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    for file_num in ["112", "181", "174", "126", "165", "072"]:
        
        base_file_path = f"/mount/resources/speech/corpora/LJSpeech/16kHz/txt/LJ050-0{file_num}.txt"

        with open(base_file_path, 'r', encoding='utf8') as base_tf:
            transcript = base_tf.read()
        
        base_wav_path = f"/mount/resources/speech/corpora/LJSpeech/16kHz/wav/LJ050-0{file_num}.wav"
        
        uc.clone_utterance(
            path_to_reference_audio=base_wav_path,
            reference_transcription=transcript,
            filename_of_result=output_dir+f"/{file_num}.wav",
            clone_speaker_identity=False,
            lang="en")

if __name__ == '__main__':
    create_mos_survey_samples(model_id="LJSpeech_FrozenCNN2D_SG_0_001")
