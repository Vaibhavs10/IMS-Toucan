import torch

from InferenceInterfaces.UtteranceCloner import UtteranceCloner

if __name__ == '__main__':
    # uc = UtteranceCloner(model_id="Models/PortaSpeech_LJSpeech_Flow_0_001/checkpoint_100170.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    uc = UtteranceCloner(model_id="Models/PortaSpeech_LJSpeech_Flow_0_001/checkpoint_100170.pt", device="cpu")

    uc.clone_utterance(path_to_reference_audio="audios/lj_50/wav/LJ050-0001.wav",
                       reference_transcription="For more information, or to volunteer, please visit librivox dot org. Report of the President's Commission on the Assassination of President Kennedy.",
                       filename_of_result="audios/test_cloned_spk_id_t.wav",
                       clone_speaker_identity=False,
                       lang="en")

    # uc.biblical_accurate_angel_mode(path_to_reference_audio="audios/test.wav",
    #                                 reference_transcription="Hello world, this is a test.",
    #                                 filename_of_result="audios/test_cloned_angelic.wav",
    #                                 list_of_speaker_references_for_ensemble=["audios/speaker_references_for_testing/female_high_voice.wav",
    #                                                                          "audios/speaker_references_for_testing/female_mid_voice.wav",
    #                                                                          "audios/speaker_references_for_testing/male_low_voice.wav",
    #                                                                          "audios/LibriTTS/174/168635/174_168635_000019_000001.wav",
    #                                                                          "audios/test.wav"],
    #                                 lang="en")
