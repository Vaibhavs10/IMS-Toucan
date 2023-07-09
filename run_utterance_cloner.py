import os
import glob
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

def get_text():
    for file_num in ["112", "181", "174", "126", "165", "072"]:
        
        base_file_path = f"/mount/resources/speech/corpora/LJSpeech/16kHz/txt/LJ050-0{file_num}.txt"

        with open(base_file_path, 'r', encoding='utf8') as base_tf:
            transcript = base_tf.read()
        
        print(transcript)

if __name__ == '__main__':
    create_lj50_samples(model_id="LJSpeech_No_0_003_scratch")
    # get_text()
    # uc.biblical_accurate_angel_mode(path_to_reference_audio="audios/test.wav",
    #                                 reference_transcription="Hello world, this is a test.",
    #                                 filename_of_result="audios/test_cloned_angelic.wav",
    #                                 list_of_speaker_references_for_ensemble=["audios/speaker_references_for_testing/female_high_voice.wav",
    #                                                                          "audios/speaker_references_for_testing/female_mid_voice.wav",
    #                                                                          "audios/speaker_references_for_testing/male_low_voice.wav",
    #                                                                          "audios/LibriTTS/174/168635/174_168635_000019_000001.wav",
    #                                                                          "audios/test.wav"],
    #                                 lang="en")
