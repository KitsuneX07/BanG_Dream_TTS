import os, re, logging, subprocess, requests, LangSegment, json, pdb, torch, librosa, sys
import numpy as np
import soundfile as sf
from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [current_dir] + sys.path

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from datetime import datetime

i18n = I18nAuto()


def generate_output_filename(character, output_path, sequence_number):
    current_date = datetime.now().strftime("%Y%m%d")
    filename = f"{character}_{current_date}_{sequence_number}.wav"
    return os.path.join(output_path, filename)


def synthesizer(GPT_model_path, SoVITS_model_path, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs, output_path, character):
    # Read reference text
    if not GPT_model_path or not SoVITS_model_path:
        raise ValueError("GPT_model_path and SoVITS_model_path must be provided.")
    if not ref_wav_path:
        raise ValueError("ref_wav_path must be provided.")
    if not prompt_text:
        raise ValueError("prompt_text must be provided.")
    # Change model weights
    change_gpt_weights(gpt_path = GPT_model_path)
    change_sovits_weights(sovits_path = SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path = ref_wav_path,
                                   prompt_text = prompt_text,
                                   prompt_language = prompt_language,
                                   text = text,
                                   text_language = text_language,
                                   how_to_cut = how_to_cut,
                                   top_k = top_k,
                                   top_p = top_p,
                                   temperature = temperature,
                                   ref_free = False,
                                   speed = speed,
                                   if_freeze = if_freeze,
                                   inp_refs = inp_refs)

    result_list = list(synthesis_result)
    if not output_path:
        output_path = "./output"

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        sequence_number = 1
        while os.path.exists(generate_output_filename(character, output_path, sequence_number)):
            sequence_number += 1
        output_wav_path = generate_output_filename(character, output_path, sequence_number)
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        return output_wav_path
