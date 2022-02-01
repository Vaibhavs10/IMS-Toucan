import os

import torch

from InferenceInterfaces.LJSpeech_FastSpeech import LJSpeech_FastSpeechInference
from InferenceInterfaces.LJSpeech_TransformerTTS import LJSpeech_TransformerTTSInference
from InferenceInterfaces.LibriTTS_FastSpeech import LibriTTS_FastSpeechInference
from InferenceInterfaces.LibriTTS_TransformerTTS import LibriTTS_TransformerTTSInference
from InferenceInterfaces.Nancy_FastSpeech import Nancy_FastSpeechInference
from InferenceInterfaces.Nancy_TransformerTTS import Nancy_TransformerTTSInference
from InferenceInterfaces.Thorsten_FastSpeech import Thorsten_FastSpeechInference
from InferenceInterfaces.Thorsten_TransformerTTS import Thorsten_TransformerTTSInference

tts_dict = {
    "fast_thorsten" : Thorsten_FastSpeechInference,
    "fast_lj"       : LJSpeech_FastSpeechInference,
    "fast_libri"    : LibriTTS_FastSpeechInference,
    "fast_nancy"    : Nancy_FastSpeechInference,

    "trans_thorsten": Thorsten_TransformerTTSInference,
    "reformer_3x_trans_lj" : LJSpeech_TransformerTTSInference,
    "trans_libri"   : LibriTTS_TransformerTTSInference,
    "trans_nancy"   : Nancy_TransformerTTSInference
    }


def read_texts(model_id, sentence, filename, device="cpu", speaker_embedding=None):
    tts = tts_dict[model_id](device=device, speaker_embedding=speaker_embedding)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def read_harvard_sentences(model_id, device):
    tts = tts_dict[model_id](device=device, speaker_embedding="default_speaker_embedding.pt")

    with open("Utility/test_sentences_combined_3.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_03_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

    with open("Utility/test_sentences_combined_6.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/harvard_06_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

def read_prosody_test_sentences(model_id, device):
    tts = tts_dict[model_id](device=device, speaker_embedding=None)

    with open("Utility/prosodic_test_sentences.txt", "r", encoding="utf8") as f:
        sents = f.read().split("\n")
    output_dir = "audios/prosody_test_{}".format(model_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for index, sent in enumerate(sents):
        tts.read_to_file(text_list=[sent], file_location=output_dir + "/{}.wav".format(index))

if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    read_prosody_test_sentences(model_id="reformer_3x_trans_lj")
