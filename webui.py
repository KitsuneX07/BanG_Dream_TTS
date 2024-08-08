import os, re, logging, subprocess, requests, LangSegment, json, pdb, torch, sys

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import gradio as gr
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [current_dir] + sys.path

from inference import synthesizer, generate_output_filename
from tools.i18n.i18n import I18nAuto
from unpacker import extract_package

i18n = I18nAuto()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

dict_language_v2 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("粤语"): "all_yue",  # 全部按中文识别
    i18n("韩文"): "all_ko",  # 全部按韩文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
    i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
}


def read_character():
    with open(r"./configs/character.json", 'r', encoding = "utf-8") as f:
        character = json.load(f)
    return character


def load_theme(theme_path):
    with open(theme_path, 'r', encoding = 'utf-8') as f:
        theme = json.load(f)
    return theme


def read_installed_model():
    model_dir = './models'
    subfolders = [f.name for f in os.scandir(model_dir) if f.is_dir()]
    return subfolders


def extract_model(package_path):
    key = b'Guess!'
    package_name = os.path.basename(package_path).split('.')[0]
    target_dir = f'./models/{package_name}'
    if os.path.exists(target_dir):
        return "模型已存在"
    else:
        os.makedirs(target_dir, exist_ok = True)
    extract_package(key, package_path, target_dir)


def read_refer_language(name):
    character_list = read_character()
    if name not in character_list:
        return gr.Dropdown(label = "参考音频语言选择", choices = ["角色不存在"], scale = 2, interactive = True, allow_custom_value = True)
    character = character_list[name]["character"]
    refer_language_path = os.path.join('./refer_audio', f"{character}")
    refer_language_list = [f.name for f in os.scandir(refer_language_path) if f.is_dir()]
    return gr.Dropdown(label = "参考音频语言选择", choices = list(refer_language_list), scale = 2, interactive = True, allow_custom_value = True)


def read_refer_audio(name, language):
    character_list = read_character()
    if name not in character_list:
        return gr.Dropdown(label = "参考音频选择", choices = ["参考音频不存在"], scale = 2, interactive = True, allow_custom_value = True)
    character = character_list[name]["character"]
    refer_audio_path = os.path.join('./refer_audio', f"{character}", f"{language}")
    refer_audio_list = [f.name for f in os.scandir(refer_audio_path) if f.is_dir()]
    return gr.Dropdown(label = "参考音频选择", choices = list(refer_audio_list), scale = 2, interactive = True, allow_custom_value = True)


def refresh_character_list():
    return gr.update(choices = list(read_installed_model()))


def refresh_language_list(name):
    return gr.update(choices = read_refer_language(name).choices)


def refresh_refer_audio_list(name, language):
    return gr.update(choices = read_refer_audio(name, language).choices)


def read_reference_files(ref_folder_path):
    ref_wav_path = None
    prompt_text = None
    for file in os.listdir(ref_folder_path):
        if file.endswith(".wav"):
            ref_wav_path = os.path.join(ref_folder_path, file)
        elif file.endswith(".lab"):
            with open(os.path.join(ref_folder_path, file), 'r', encoding = 'utf-8') as f:
                prompt_text = f.read()
    return ref_wav_path, prompt_text


def tts_inference(name, language, refer_audio, text, text_language, how_to_cut, top_k, top_p, temperature, speed, if_freeze, inp_refs):
    character_list = read_character()
    character = character_list[name]["character"]
    GPT_model_path = os.path.join('.\\models', name, f"{character}-e8.ckpt")
    SoVITS_model_path = os.path.join('.\\models', name, f"{character}_e6.pth")
    ref_folder_path = os.path.join('./refer_audio', character, language, refer_audio)

    ref_wav_path, prompt_text = read_reference_files(ref_folder_path)
    if not ref_wav_path or not prompt_text:
        raise FileNotFoundError("Reference .wav or .lab file not found in the specified folder.")

    output_path = ".\\output"
    os.makedirs(output_path, exist_ok = True)

    output_wav_path = synthesizer(
        GPT_model_path, SoVITS_model_path, ref_wav_path, prompt_text, language, text, text_language,
        how_to_cut, top_k, top_p, temperature, ref_free = False, speed = speed, if_freeze = if_freeze,
        inp_refs = inp_refs, output_path = output_path, character = character
    )
    return gr.Audio(label = "输出的语音", value = output_wav_path)


with gr.Blocks(theme = gr.Theme.load(".\\configs\\themes_theme_schema@0.0.1.json")) as app:
    gr.Markdown("""
    # BanG Dream TTS
    仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容。<br>
    本整合包完全免费，严禁以任何形式倒卖，如果你从任何地方**付费**购买了本整合包，请**立即退款**。<br> 
    整合包作者：[bilibili@阿狸不吃隼舞](https://space.bilibili.com/403335715) [Github@KitsuneX07](https://github.com/KitsuneX07)
    """)

    with gr.Tabs():
        with gr.TabItem(label = "模型管理"):
            gr.Markdown("""
            ## 安装模型
            模型下载地址:<br>
            请上传模型压缩包用于自动安装。<br>
            """)
            with gr.Row():
                package_path = gr.File(label = "请上传模型压缩包", type = "filepath", scale = 13)
                install_button = gr.Button("安装模型", scale = 1)

            install_button.click(fn = extract_model, inputs = [package_path], outputs = [install_button])

        with gr.TabItem(label = "GPT-SoVITS TTS"):
            gr.Markdown("""
            
            """)
            with gr.Row():
                with gr.Column(scale = 3):
                    character_list = list(read_character())
                    character_dropdown = gr.Dropdown(label = "角色选择", choices = list(read_installed_model()), scale = 2)
                    refresh_character_button = gr.Button("刷新角色列表", scale = 1)
                with gr.Column(scale = 3):
                    language_dropdown = gr.Dropdown(label = "参考音频语言选择", choices = ["请先选择说话人角色"], scale = 2, interactive = True, allow_custom_value = True)
                    refresh_language_button = gr.Button("刷新音频语言列表", scale = 1)
                with gr.Column(scale = 3):
                    refer_audio_dropdown = gr.Dropdown(label = "参考音频选择", choices = ["请先选择说话人角色和参考音频语言"], scale = 2, interactive = True, allow_custom_value = True)
                    refresh_refer_audio_button = gr.Button("刷新参考音频列表", scale = 1)
                inp_refs = gr.File(label = i18n("可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。如不填写此项，音色由左侧单个参考音频控制。如是微调模型，建议参考音频全部在微调训练集音色内，底模不用管。"), file_count = 'multiple', scale = 13)
            gr.Markdown(value = i18n("*请填写需要合成的目标文本和语种模式"))
            with gr.Row():
                with gr.Column(scale = 13):
                    text = gr.Textbox(label = i18n("需要合成的文本"), value = "", lines = 26, max_lines = 26)
                with gr.Column(scale = 7):
                    text_language = gr.Dropdown(
                        label = i18n("需要合成的语种") + i18n(".限制范围越小判别效果越好。"), choices = list(dict_language_v2.keys()), value = i18n("中文"), scale = 1
                    )
                    how_to_cut = gr.Dropdown(
                        label = i18n("怎么切"),
                        choices = [i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                        value = i18n("凑四句一切"),
                        interactive = True, scale = 1
                    )
                    gr.Markdown(value = i18n("语速调整，高为更快"))
                    if_freeze = gr.Checkbox(label = i18n("是否直接对上次合成结果调整语速和音色。防止随机性。"), value = False, interactive = True, show_label = True, scale = 1)
                    speed = gr.Slider(minimum = 0.6, maximum = 1.65, step = 0.05, label = i18n("语速"), value = 1, interactive = True, scale = 1)
                    gr.Markdown(i18n("GPT采样参数(无参考文本时不要太低。不懂就用默认)："))
                    top_k = gr.Slider(minimum = 1, maximum = 100, step = 1, label = i18n("top_k"), value = 15, interactive = True, scale = 1)
                    top_p = gr.Slider(minimum = 0, maximum = 1, step = 0.05, label = i18n("top_p"), value = 1, interactive = True, scale = 1)
                    temperature = gr.Slider(minimum = 0, maximum = 1, step = 0.05, label = i18n("temperature"), value = 1, interactive = True, scale = 1)

            with gr.Row():
                inference_button = gr.Button(i18n("合成语音"), variant = "primary", size = 'lg', scale = 25)
                output = gr.Audio(label = i18n("输出的语音"), scale = 14)

            character_dropdown.change(fn = read_refer_language, inputs = [character_dropdown], outputs = [language_dropdown])
            language_dropdown.change(fn = read_refer_audio, inputs = [character_dropdown, language_dropdown], outputs = [refer_audio_dropdown])
            refresh_character_button.click(fn = refresh_character_list, outputs = [character_dropdown])
            refresh_language_button.click(fn = refresh_language_list, inputs = [character_dropdown], outputs = [language_dropdown])
            refresh_refer_audio_button.click(fn = refresh_refer_audio_list, inputs = [character_dropdown, language_dropdown], outputs = [refer_audio_dropdown])
            inference_button.click(fn = tts_inference, inputs = [character_dropdown, language_dropdown, refer_audio_dropdown, text, text_language, how_to_cut, top_k, top_p, temperature, speed, if_freeze, inp_refs], outputs = [output])
    if __name__ == '__main__':
        app.launch(inbrowser = True)
