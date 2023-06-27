import torch

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

def create_mos_survey_samples(model_id):
    uc = UtteranceCloner(model_id=model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    for file_num in ["112", "181", "174", "126", "165", "072"]:
        
        base_file_path = f"/mount/resources/speech/corpora/LJSpeech/16kHz/txt/LJ050-0{}.txt"

        with open(base_file_path, 'r', encoding='utf8') as base_tf:
            transcript = tf.read()
        
        base_wav_path = f"/mount/resources/speech/corpora/LJSpeech/16kHz/wav/LJ050-0{}.wav"
        
        uc.clone_utterance(
            path_to_reference_audio=base_wav_path,
            reference_transcription=transcript,
            filename_of_result=f"audios/no_postnet{file_num}.wav",
            clone_speaker_identity=False,
            lang="en")


if __name__ == '__main__':

    # uc.biblical_accurate_angel_mode(path_to_reference_audio="audios/test.wav",
    #                                 reference_transcription="Hello world, this is a test.",
    #                                 filename_of_result="audios/test_cloned_angelic.wav",
    #                                 list_of_speaker_references_for_ensemble=["audios/speaker_references_for_testing/female_high_voice.wav",
    #                                                                          "audios/speaker_references_for_testing/female_mid_voice.wav",
    #                                                                          "audios/speaker_references_for_testing/male_low_voice.wav",
    #                                                                          "audios/LibriTTS/174/168635/174_168635_000019_000001.wav",
    #                                                                          "audios/test.wav"],
    #                                 lang="en")
