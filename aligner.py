import re
import os
import statistics

import librosa
import librosa.core as lb
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import phonemizer
import pyloudnorm as pyln
import soundfile as sf
import torch
import torch.multiprocessing
import torch.nn as nn
from cleantext import clean
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchaudio.transforms import MuLawDecoding
from torchaudio.transforms import MuLawEncoding
from torchaudio.transforms import Resample
from torchaudio.transforms import Vad as VoiceActivityDetection

####################################
#      The important functions     #
####################################

def build_path_to_transcript_dict_3xljspeech():
    path_to_transcript = dict()
    for transcript_file in os.listdir("/mount/arbeitsdaten/synthesis/attention_projects/LJSpeech_3xlong_stripped/txt_long"):
        with open("/mount/arbeitsdaten/synthesis/attention_projects/LJSpeech_3xlong_stripped/txt_long/" + transcript_file, 'r', encoding='utf8') as tf:
            transcript = tf.read()
        wav_path = "/mount/arbeitsdaten/synthesis/attention_projects/LJSpeech_3xlong_stripped/wav_long/" + transcript_file.rstrip(".txt") + ".wav"
        path_to_transcript[wav_path] = transcript
    return path_to_transcript

def find_samples_with_highest_ctc(path_to_transcript_dict, path_to_aligner="aligner.pt", device="cpu"):
    """
    Supply a path_to_transcript_dict and this function will tell you which samples have an unusually high CTC loss.

    Works only for English, although it would be extremely easy to extend this to any language.
    """
    tf = ArticulatoryCombinedTextFrontend()
    paths = list(path_to_transcript_dict.keys())
    _, sr = sf.read(paths[0])
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=True)

    acoustic_model = Aligner()
    acoustic_model.load_state_dict(torch.load(path_to_aligner, map_location='cpu')["asr_model"]).to(device)
    ctc_losses = list()

    for path in paths:
        text = tf.string_to_tensor(path_to_transcript_dict[path])
        audio, _ = sf.read(path)
        melspec = ap.audio_to_mel_spec_tensor(audio)
        _, ctc_loss = acoustic_model.inference(mel=melspec.to(device),
                                               tokens=text.to(device))
        ctc_losses.append(ctc_loss)
    mean_ctc = sum(ctc_losses) / len(ctc_losses)
    std_dev = statistics.stdev(ctc_losses)
    threshold = mean_ctc + std_dev
    for index in range(len(ctc_losses), 0, -1):
        if ctc_losses[index - 1] > threshold:
            print(f"{paths.pop(index - 1)} is potentially problematic")

####################################
#      The rest is dependency      #
####################################

class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        return x


class Aligner(torch.nn.Module):

    def __init__(self,
                 n_mels=80,
                 num_symbols=145,
                 lstm_dim=512,
                 conv_dim=512):
        super().__init__()
        self.convs = nn.ModuleList([
            BatchNormConv(n_mels, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            BatchNormConv(conv_dim, conv_dim, 3),
            nn.Dropout(p=0.5),
            ])
        self.rnn = torch.nn.LSTM(conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(2 * lstm_dim, num_symbols)
        self.tf = ArticulatoryCombinedTextFrontend()
        self.ctc_loss = CTCLoss(blank=144, zero_infinity=True)
        self.vector_to_id = dict()
        for phone in self.tf.phone_to_vector:
            self.vector_to_id[tuple(self.tf.phone_to_vector[phone])] = self.tf.phone_to_id[phone]

    def forward(self, x, lens=None):
        for conv in self.convs:
            x = conv(x)
        if lens is not None:
            x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        if lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.proj(x)
        return x

    def inference(self, mel, tokens):
        tokens_indexed = list()
        for vector in tokens:
            tokens_indexed.append(self.vector_to_id[tuple(vector.cpu().detach().numpy().tolist())])
        tokens = np.asarray(tokens_indexed)
        pred = self(mel.unsqueeze(0))
        return self.ctc_loss(pred.transpose(0, 1).log_softmax(2), torch.LongTensor(tokens), torch.LongTensor([len(pred[0])]),
                             torch.LongTensor([len(tokens)])).item()


class ArticulatoryCombinedTextFrontend:

    def __init__(self,
                 use_word_boundaries=False,
                 use_explicit_eos=True,
                 use_prosody=False,
                 use_lexical_stress=False,
                 allow_unknown=False,
                 add_silence_to_end=True,
                 strip_silence=True):
        self.strip_silence = strip_silence
        self.use_word_boundaries = use_word_boundaries
        self.allow_unknown = allow_unknown
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress
        self.add_silence_to_end = add_silence_to_end
        self.clean_lang = "en"
        self.g2p_lang = "en-us"
        self.expand_abbreviations = english_text_expansion
        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = {
            '~': 0,
            '#': 1,
            '?': 2,
            '!': 3,
            '.': 4,
            'ɜ': 5,
            'ɫ': 6,
            'ə': 7,
            'ɚ': 8,
            'a': 9,
            'ð': 10,
            'ɛ': 11,
            'ɪ': 12,
            'ᵻ': 13,
            'ŋ': 14,
            'ɔ': 15,
            'ɒ': 16,
            'ɾ': 17,
            'ʃ': 18,
            'θ': 19,
            'ʊ': 20,
            'ʌ': 21,
            'ʒ': 22,
            'æ': 23,
            'b': 24,
            'ʔ': 25,
            'd': 26,
            'e': 27,
            'f': 28,
            'g': 29,
            'h': 30,
            'i': 31,
            'j': 32,
            'k': 33,
            'l': 34,
            'm': 35,
            'n': 36,
            'ɳ': 37,
            'o': 38,
            'p': 39,
            'ɡ': 40,
            'ɹ': 41,
            'r': 42,
            's': 43,
            't': 44,
            'u': 45,
            'v': 46,
            'w': 47,
            'x': 48,
            'z': 49,
            'ʀ': 50,
            'ø': 51,
            'ç': 52,
            'ɐ': 53,
            'œ': 54,
            'y': 55,
            'ʏ': 56,
            'ɑ': 57,
            'c': 58,
            'ɲ': 59,
            'ɣ': 60,
            'ʎ': 61,
            'β': 62,
            'ʝ': 63,
            'ɟ': 64,
            'q': 65,
            'ɕ': 66,
            'ʲ': 67,
            'ɭ': 68,
            'ɵ': 69,
            'ʑ': 70,
            'ʋ': 71,
            'ʁ': 72,
            }  # for the states of the ctc loss and dijkstra in the aligner
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}

    def string_to_tensor(self, text, view=False, device="cpu", handle_missing=True, input_phonemes=False):
        if input_phonemes:
            phones = text
        else:
            phones = self.get_phone_string(text=text, include_eos_symbol=True)
        if view:
            print("Phonemes: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        for char in phones:
            if handle_missing:
                try:
                    phones_vector.append(self.phone_to_vector[char])
                except KeyError:
                    print("unknown phoneme: {}".format(char))
            else:
                phones_vector.append(self.phone_to_vector[char])  # leave error handling to elsewhere
        return torch.Tensor(phones_vector, device=device)

    def get_phone_string(self, text, include_eos_symbol=True):
        # clean unicode errors, expand abbreviations, handle emojis etc.
        utt = clean(text, fix_unicode=True, to_ascii=False, lower=False, lang=self.clean_lang)
        self.expand_abbreviations(utt)
        # phonemize
        phones = phonemizer.phonemize(utt,
                                      language_switch='remove-flags',
                                      backend="espeak",
                                      language=self.g2p_lang,
                                      preserve_punctuation=True,
                                      strip=True,
                                      punctuation_marks=';:,.!?¡¿—…"«»“”~/',
                                      with_stress=self.use_stress).replace(";", ",").replace("/", " ").replace("—", "") \
            .replace(":", ",").replace('"', ",").replace("-", ",").replace("...", ",").replace("-", ",").replace("\n", " ") \
            .replace("\t", " ").replace("¡", "").replace("¿", "").replace(",", "~").replace(" ̃", "").replace('̩', "").replace("̃", "")
        # less than 1 wide characters hidden here
        phones = re.sub("~+", "~", phones)
        if not self.use_prosody:
            # retain ~ as heuristic pause marker, even though all other symbols are removed with this option.
            # also retain . ? and ! since they can be indicators for the stop token
            phones = phones.replace("ˌ", "").replace("ː", "").replace("ˑ", "") \
                .replace("˘", "").replace("|", "").replace("‖", "")
        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")
        else:
            phones = re.sub(r"\s+", " ", phones)
            phones = re.sub(" ", "~", phones)
        if self.strip_silence:
            phones = phones.lstrip("~").rstrip("~")
        if self.add_silence_to_end:
            phones += "~"  # adding a silence in the end during add_silence_to_end produces more natural sounding prosody
        if include_eos_symbol:
            phones += "#"

        phones = "~" + phones
        phones = re.sub("~+", "~", phones)

        return phones


def english_text_expansion(text):
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in
                      [('Mrs.', 'misess'), ('Mr.', 'mister'), ('Dr.', 'doctor'), ('St.', 'saint'), ('Co.', 'company'), ('Jr.', 'junior'), ('Maj.', 'major'),
                       ('Gen.', 'general'), ('Drs.', 'doctors'), ('Rev.', 'reverend'), ('Lt.', 'lieutenant'), ('Hon.', 'honorable'), ('Sgt.', 'sergeant'),
                       ('Capt.', 'captain'), ('Esq.', 'esquire'), ('Ltd.', 'limited'), ('Col.', 'colonel'), ('Ft.', 'fort')]]
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def generate_feature_table():
    ipa_to_phonemefeats = {
        '~': {'symbol_type': 'silence'},
        '#': {'symbol_type': 'end of sentence'},
        '?': {'symbol_type': 'questionmark'},
        '!': {'symbol_type': 'exclamationmark'},
        '.': {'symbol_type': 'fullstop'},
        'ɜ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'unrounded',
            },
        'ɫ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'lateral-approximant',
            },
        'ə': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'mid',
            'vowel_roundedness': 'unrounded',
            },
        'ɚ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'mid',
            'vowel_roundedness': 'unrounded',
            },
        'a': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'unrounded',
            },
        'ð': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'dental',
            'consonant_manner': 'fricative'
            },
        'ɛ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'unrounded',
            },
        'ɪ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front_central',
            'vowel_openness'   : 'close_close-mid',
            'vowel_roundedness': 'unrounded',
            },
        'ᵻ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'unrounded',
            },
        'ŋ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'nasal'
            },
        'ɔ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'rounded',
            },
        'ɒ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'rounded',
            },
        'ɾ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'tap'
            },
        'ʃ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'postalveolar',
            'consonant_manner': 'fricative'
            },
        'θ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'dental',
            'consonant_manner': 'fricative'
            },
        'ʊ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central_back',
            'vowel_openness'   : 'close_close-mid',
            'vowel_roundedness': 'unrounded'
            },
        'ʌ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'unrounded'
            },
        'ʒ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'postalveolar',
            'consonant_manner': 'fricative'
            },
        'æ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open-mid_open',
            'vowel_roundedness': 'unrounded'
            },
        'b': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'stop'
            },
        'ʔ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'glottal',
            'consonant_manner': 'stop'
            },
        'd': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'stop'
            },
        'e': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'unrounded'
            },
        'f': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'fricative'
            },
        'g': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'stop'
            },
        'h': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'glottal',
            'consonant_manner': 'fricative'
            },
        'i': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'unrounded'
            },
        'j': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'approximant'
            },
        'k': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'stop'
            },
        'l': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'lateral-approximant'
            },
        'm': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'nasal'
            },
        'n': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'nasal'
            },
        'ɳ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'nasal'
            },
        'o': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'rounded'
            },
        'p': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'stop'
            },
        'ɡ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'stop'
            },
        'ɹ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'approximant'
            },
        'r': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'trill'
            },
        's': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'fricative'
            },
        't': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'stop'
            },
        'u': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'rounded',
            },
        'v': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'fricative'
            },
        'w': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labial-velar',
            'consonant_manner': 'approximant'
            },
        'x': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'fricative'
            },
        'z': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolar',
            'consonant_manner': 'fricative'
            },
        'ʀ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'trill'
            },
        'ø': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close-mid',
            'vowel_roundedness': 'rounded'
            },
        'ç': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'fricative'
            },
        'ɐ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'unrounded'
            },
        'œ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'rounded'
            },
        'y': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front',
            'vowel_openness'   : 'close',
            'vowel_roundedness': 'rounded'
            },
        'ʏ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'front_central',
            'vowel_openness'   : 'close_close-mid',
            'vowel_roundedness': 'rounded'
            },
        'ɑ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'back',
            'vowel_openness'   : 'open',
            'vowel_roundedness': 'unrounded'
            },
        'c': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'stop'
            },
        'ɲ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'nasal'
            },
        'ɣ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'velar',
            'consonant_manner': 'fricative'
            },
        'ʎ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'lateral-approximant'
            },
        'β': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'bilabial',
            'consonant_manner': 'fricative'
            },
        'ʝ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'fricative'
            },
        'ɟ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'stop'
            },
        'q': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'stop'
            },
        'ɕ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'unvoiced',
            'consonant_place' : 'alveolopalatal',
            'consonant_manner': 'fricative'
            },
        'ʲ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',
            'consonant_manner': 'approximant'
            },
        'ɭ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'palatal',  # should be retroflex, but palatal should be close enough
            'consonant_manner': 'lateral-approximant'
            },
        'ɵ': {
            'symbol_type'      : 'phoneme',
            'vowel_consonant'  : 'vowel',
            'VUV'              : 'voiced',
            'vowel_frontness'  : 'central',
            'vowel_openness'   : 'open-mid',
            'vowel_roundedness': 'rounded'
            },
        'ʑ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'alveolopalatal',
            'consonant_manner': 'fricative'
            },
        'ʋ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'labiodental',
            'consonant_manner': 'approximant'
            },
        'ʁ': {
            'symbol_type'     : 'phoneme',
            'vowel_consonant' : 'consonant',
            'VUV'             : 'voiced',
            'consonant_place' : 'uvular',
            'consonant_manner': 'fricative'
            },
        }

    feat_types = set()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            [feat_types.add(feat) for feat in ipa_to_phonemefeats[ipa].keys()]

    feat_to_val_set = dict()
    for feat in feat_types:
        feat_to_val_set[feat] = set()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            for feat in ipa_to_phonemefeats[ipa]:
                feat_to_val_set[feat].add(ipa_to_phonemefeats[ipa][feat])

    # print(feat_to_val_set)

    value_list = set()
    for val_set in [feat_to_val_set[feat] for feat in feat_to_val_set]:
        for value in val_set:
            value_list.add(value)
    # print("{")
    # for index, value in enumerate(list(value_list)):
    #     print('"{}":{},'.format(value,index))
    # print("}")

    value_to_index = {
        "dental"             : 0,
        "postalveolar"       : 1,
        "mid"                : 2,
        "close-mid"          : 3,
        "vowel"              : 4,
        "silence"            : 5,
        "consonant"          : 6,
        "close"              : 7,
        "velar"              : 8,
        "stop"               : 9,
        "palatal"            : 10,
        "nasal"              : 11,
        "glottal"            : 12,
        "central"            : 13,
        "back"               : 14,
        "approximant"        : 15,
        "uvular"             : 16,
        "open-mid"           : 17,
        "front_central"      : 18,
        "front"              : 19,
        "end of sentence"    : 20,
        "labiodental"        : 21,
        "close_close-mid"    : 22,
        "labial-velar"       : 23,
        "unvoiced"           : 24,
        "central_back"       : 25,
        "trill"              : 26,
        "rounded"            : 27,
        "open-mid_open"      : 28,
        "tap"                : 29,
        "alveolar"           : 30,
        "bilabial"           : 31,
        "phoneme"            : 32,
        "open"               : 33,
        "fricative"          : 34,
        "unrounded"          : 35,
        "lateral-approximant": 36,
        "voiced"             : 37,
        "questionmark"       : 38,
        "exclamationmark"    : 39,
        "fullstop"           : 40,
        "alveolopalatal"     : 41
        }

    phone_to_vector = dict()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            phone_to_vector[ipa] = [0] * sum([len(values) for values in [feat_to_val_set[feat] for feat in feat_to_val_set]])
            for feat in ipa_to_phonemefeats[ipa]:
                if ipa_to_phonemefeats[ipa][feat] in value_to_index:
                    phone_to_vector[ipa][value_to_index[ipa_to_phonemefeats[ipa][feat]]] = 1

    for feat in feat_to_val_set:
        for value in feat_to_val_set[feat]:
            if value not in value_to_index:
                print(f"Unknown feature value in featureset! {value}")
    return phone_to_vector


class AudioPreprocessor:

    def __init__(self, input_sr, output_sr=None, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False):
        self.cut_silence = cut_silence
        self.sr = input_sr
        self.new_sr = output_sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mel_buckets = melspec_buckets
        self.vad = VoiceActivityDetection(sample_rate=input_sr)
        self.mu_encode = MuLawEncoding()
        self.mu_decode = MuLawDecoding()
        self.meter = pyln.Meter(input_sr)
        self.final_sr = input_sr
        if output_sr is not None and output_sr != input_sr:
            self.resample = Resample(orig_freq=input_sr, new_freq=output_sr)
            self.final_sr = output_sr
        else:
            self.resample = lambda x: x

    def apply_mu_law(self, audio):
        if isinstance(audio, torch.Tensor):
            return self.mu_encode(audio)
        else:
            return self.mu_encode(torch.Tensor(audio))

    def cut_silence_from_beginning_and_end(self, audio):
        silence = torch.zeros([20000])
        no_silence_front = self.vad(torch.cat((silence, torch.Tensor(audio), silence), 0))
        reversed_audio = torch.flip(no_silence_front, (0,))
        no_silence_back = self.vad(torch.Tensor(reversed_audio))
        unreversed_audio = torch.flip(no_silence_back, (0,))
        return unreversed_audio

    def to_mono(self, x):
        if len(x.shape) == 2:
            return lb.to_mono(numpy.transpose(x))
        else:
            return x

    def normalize_loudness(self, audio):
        loudness = self.meter.integrated_loudness(audio)
        loud_normed = pyln.normalize.loudness(audio, loudness, -30.0)
        peak = numpy.amax(numpy.abs(loud_normed))
        peak_normed = numpy.divide(loud_normed, peak)
        return peak_normed

    def logmelfilterbank(self, audio, sampling_rate, fmin=40, fmax=8000, eps=1e-10):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        # get amplitude spectrogram
        x_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=None, window="hann", pad_mode="reflect")
        spc = np.abs(x_stft).T
        # get mel basis
        fmin = 0 if fmin is None else fmin
        fmax = sampling_rate / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(sampling_rate, self.n_fft, self.mel_buckets, fmin, fmax)
        # apply log and return
        return torch.Tensor(np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))).transpose(0, 1)

    def normalize_audio(self, audio):
        audio = self.to_mono(audio)
        audio = self.normalize_loudness(audio)
        if self.cut_silence:
            audio = self.cut_silence_from_beginning_and_end(audio)
        else:
            audio = torch.Tensor(audio)
        audio = self.resample(audio)
        return audio

    def visualize_cleaning(self, unclean_audio):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        unclean_audio_mono = self.to_mono(unclean_audio)
        unclean_spec = self.audio_to_mel_spec_tensor(unclean_audio_mono, normalize=False).numpy()
        clean_spec = self.audio_to_mel_spec_tensor(unclean_audio_mono, normalize=True).numpy()
        lbd.specshow(unclean_spec, sr=self.sr, cmap='GnBu', y_axis='mel', ax=ax[0], x_axis='time')
        ax[0].set(title='Uncleaned Audio')
        ax[0].label_outer()
        if self.new_sr is not None:
            lbd.specshow(clean_spec, sr=self.new_sr, cmap='GnBu', y_axis='mel', ax=ax[1], x_axis='time')
        else:
            lbd.specshow(clean_spec, sr=self.sr, cmap='GnBu', y_axis='mel', ax=ax[1], x_axis='time')
        ax[1].set(title='Cleaned Audio')
        ax[1].label_outer()
        plt.show()

    def audio_to_wave_tensor(self, audio, normalize=True, mulaw=False):
        if normalize:
            audio = self.normalize_audio(audio)
        if mulaw:
            return self.apply_mu_law(audio)
        else:
            if isinstance(audio, torch.Tensor):
                return audio
            else:
                return torch.Tensor(audio)

    def audio_to_mel_spec_tensor(self, audio, normalize=True, explicit_sampling_rate=None):
        if explicit_sampling_rate is None:
            if normalize:
                audio = self.normalize_audio(audio)
                return self.logmelfilterbank(audio=audio, sampling_rate=self.final_sr)
            return self.logmelfilterbank(audio=audio, sampling_rate=self.sr)
        if normalize:
            audio = self.normalize_audio(audio)
        return self.logmelfilterbank(audio=audio, sampling_rate=explicit_sampling_rate)

file_dict = build_path_to_transcript_dict_3xljspeech()
find_samples_with_highest_ctc(file_dict)