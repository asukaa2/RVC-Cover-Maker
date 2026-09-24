import sys, os
import subprocess
import torch
from functools import lru_cache
import shutil
import re
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment
from audio_separator.separator import Separator
import logging
import yaml

now_dir = os.getcwd()
sys.path.append(now_dir)

# ---------------------------------------------------------------------------
# RVC integration
#
# This project uses the `rvc` package (https://github.com/uziproj/rvc) for
# voice conversion. The package is installed via:
#     pip install git+https://github.com/uziproj/rvc.git
#
# Public API surface we rely on:
#   - rvc.Config(embedder_model=..., f0_method=..., is_half=..., cpu_mode=...)
#   - rvc.run_inference_script(config=..., pth_path=..., input_path=...,
#                              output_path=..., pitch=..., f0_method=...,
#                              index_path=..., embedder_model=..., ...)
#   - rvc.utils.HF_download_file(url, output_path)  for model downloads
#
# We import lazily inside the helpers below so the heavy PyTorch / fairseq
# stack is only loaded when an RVC inference is actually requested (and not
# at module import time, which would break e.g. the music-separation path).
# ---------------------------------------------------------------------------
from programs.music_separation_code.inference import proc_file

models_vocals = [
    {
        "name": "Mel-Roformer by KimberleyJSN",
        "path": os.path.join(now_dir, "models", "mel-vocals"),
        "model": os.path.join(now_dir, "models", "mel-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-vocals", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        "model_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
    },
    {
        "name": "BS-Roformer by ViperX",
        "path": os.path.join(now_dir, "models", "bs-vocals"),
        "model": os.path.join(now_dir, "models", "bs-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "bs-vocals", "config.yaml"),
        "type": "bs_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "model_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
    {
        "name": "MDX23C",
        "path": os.path.join(now_dir, "models", "mdx23c-vocals"),
        "model": os.path.join(now_dir, "models", "mdx23c-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-vocals", "config.yaml"),
        "type": "mdx23c",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml",
        "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
    },
]

karaoke_models = [
    {
        "name": "Mel-Roformer Karaoke by aufr33 and viperx",
        "path": os.path.join(now_dir, "models", "mel-kara"),
        "model": os.path.join(now_dir, "models", "mel-kara", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-kara", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
        "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    },
    {
        "name": "UVR-BVE",
        "full_name": "UVR-BVE-4B_SN-44100-1.pth",
        "arch": "vr",
    },
]

denoise_models = [
    {
        "name": "Mel-Roformer Denoise Normal by aufr33",
        "path": os.path.join(now_dir, "models", "mel-denoise"),
        "model": os.path.join(now_dir, "models", "mel-denoise", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-denoise", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    },
    {
        "name": "Mel-Roformer Denoise Aggressive by aufr33",
        "path": os.path.join(now_dir, "models", "mel-denoise-aggr"),
        "model": os.path.join(now_dir, "models", "mel-denoise-aggr", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-denoise-aggr", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    },
    {
        "name": "UVR Denoise",
        "full_name": "UVR-DeNoise.pth",
        "arch": "vr",
    },
]

dereverb_models = [
    {
        "name": "MDX23C DeReverb by aufr33 and jarredou",
        "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
        "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
        "type": "mdx23c",
        "config_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt",
    },
    {
        "name": "BS-Roformer Dereverb by anvuew",
        "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
        "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
        "type": "bs_roformer",
        "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml",
        "model_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt",
    },
    {
        "name": "UVR-Deecho-Dereverb",
        "full_name": "UVR-DeEcho-DeReverb.pth",
        "arch": "vr",
    },
    {
        "name": "MDX Reverb HQ by FoxJoy",
        "full_name": "Reverb_HQ_By_FoxJoy.onnx",
        "arch": "mdx",
    },
]

deecho_models = [
    {
        "name": "UVR-Deecho-Normal",
        "full_name": "UVR-De-Echo-Normal.pth",
        "arch": "vr",
    },
    {
        "name": "UVR-Deecho-Agggressive",
        "full_name": "UVR-De-Echo-Aggressive.pth",
        "arch": "vr",
    },
]


def get_config(embedder_model="contentvec", f0_method="rmvpe", is_half=None, cpu_mode=False):
    """Build (or reuse the singleton) `rvc.Config`.

    `rvc.Config` is itself a singleton via `@singleton` in the upstream
    package, so the FIRST call's parameters win — subsequent calls with
    different `embedder_model` / `f0_method` arguments are silently ignored
    by the package. We honour that contract here and intentionally do NOT
    wrap this function in `lru_cache`, because callers may pass different
    parameters (e.g. switching embedder between main and backing vocals)
    and we still need to return *a* Config object back to them.
    """
    from rvc import Config

    # Auto-detect fp16 support unless the caller pinned `is_half` explicitly.
    if is_half is None:
        if torch.cuda.is_available():
            is_half = check_fp16_support(torch.cuda.current_device())
        else:
            is_half = False

    return Config(
        embedder_model=embedder_model,
        f0_method=f0_method,
        is_half=is_half,
        cpu_mode=cpu_mode,
    )


def run_rvc_inference(
    model_path,
    index_path,
    input_audio_path,
    output_audio_path,
    embedder_model,
    pitch,
    f0_method,
    filter_radius,
    index_rate,
    volume_envelope,
    protect,
    split_audio,
    f0_autotune,
    hop_length,
    export_format,
):
    """Run a single RVC voice conversion using the `rvc` package.

    This wraps `rvc.run_inference_script` (function form) which loads the
    .pth model fresh on every call. For back-to-back conversions of the
    SAME model (e.g. main vocals + backing vocals) this is slightly less
    efficient than `rvc.RVClass` (which loads once and reuses), but it
    matches the original Cover Maker semantics where two different model
    paths may be passed in the same `full_inference_program` call.

    The `rvc` package normalises the embedder name internally, so legacy
    Cover-Maker values like "contentvec" are accepted here and mapped to
    the package's expected "contentvec_base" identifier automatically.
    """
    from rvc import run_inference_script

    # Normalise embedder names to the identifiers the `rvc` package expects.
    # The original Cover Maker UI exposes ["contentvec", "chinese-hubert-base",
    # "japanese-hubert-base", "korean-hubert-base"]; uziproj/rvc expects the
    # full filenames found in `assets/models/*.pt` (without the `.pt` suffix).
    embedder_aliases = {
        "contentvec": "contentvec_base",
        "chinese-hubert-base": "chinese_hubert_base",
        "japanese-hubert-base": "japanese_hubert_base",
        "korean-hubert-base": "korean_hubert_base",
    }
    embedder_model = embedder_aliases.get(embedder_model, embedder_model)

    # The Cover Maker UI historically offered a bare "crepe" option, but the
    # uziproj/rvc package requires an explicit size suffix. Map "crepe" ->
    # "crepe-full" for backwards compatibility with existing presets.
    f0_aliases = {
        "crepe": "crepe-full",
    }
    f0_method = f0_aliases.get(f0_method, f0_method)

    config = get_config(embedder_model=embedder_model, f0_method=f0_method)

    run_inference_script(
        config=config,
        input_path=input_audio_path,
        output_path=output_audio_path,
        pth_path=model_path,
        pitch=pitch,
        f0_method=f0_method,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        hop_length=hop_length,
        index_path=index_path,
        embedder_model=embedder_model,
        export_format=export_format.lower(),
        split_audio=split_audio,
        f0_autotune=f0_autotune,
    )


def download_file(url, path, filename):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return

    try:
        response = torch.hub.download_url_to_file(url, file_path)
        print(f"File '{filename}' downloaded successfully")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")


def get_model_info_by_name(model_name):
    all_models = (
        models_vocals
        + karaoke_models
        + dereverb_models
        + deecho_models
        + denoise_models
    )
    for model in all_models:
        if model["name"] == model_name:
            return model
    return None


def get_last_modified_file(pasta):
    if not os.path.isdir(pasta):
        raise NotADirectoryError(f"{pasta} is not a valid directory.")
    arquivos = [f for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))]
    if not arquivos:
        return None
    return max(arquivos, key=lambda x: os.path.getmtime(os.path.join(pasta, x)))


def search_with_word(folder, word):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    file_with_word = [file for file in os.listdir(folder) if word in file]
    if not file_with_word:
        return None
    most_recent_file = max(
        file_with_word, key=lambda file: os.path.getmtime(os.path.join(folder, file))
    )
    return most_recent_file


def search_with_two_words(folder, word1, word2):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    file_with_words = [
        file for file in os.listdir(folder) if word1 in file and word2 in file
    ]
    if not file_with_words:
        return None
    most_recent_file = max(
        file_with_words, key=lambda file: os.path.getmtime(os.path.join(folder, file))
    )
    return most_recent_file


def get_last_modified_folder(path):
    directories = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not directories:
        return None
    last_modified_folder = max(directories, key=os.path.getmtime)
    return last_modified_folder


def add_audio_effects(
    audio_path,
    reverb_size,
    reverb_wet,
    reverb_dry,
    reverb_damping,
    reverb_width,
    output_path,
):
    board = Pedalboard([])
    board.append(
        Reverb(
            room_size=reverb_size,
            dry_level=reverb_dry,
            wet_level=reverb_wet,
            damping=reverb_damping,
            width=reverb_width,
        )
    )
    with AudioFile(audio_path) as f:
        with AudioFile(output_path, "w", f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
    return output_path


def merge_audios(
    vocals_path,
    inst_path,
    backing_path,
    output_path,
    main_gain,
    inst_gain,
    backing_Vol,
    output_format,
):
    main_vocal_audio = AudioSegment.from_file(vocals_path, format="flac") + main_gain
    instrumental_audio = AudioSegment.from_file(inst_path, format="flac") + inst_gain
    backing_vocal_audio = (
        AudioSegment.from_file(backing_path, format="flac") + backing_Vol
    )
    combined_audio = main_vocal_audio.overlay(
        instrumental_audio.overlay(backing_vocal_audio)
    )
    combined_audio.export(output_path, format=output_format)
    return output_path


def check_fp16_support(device):
    i_device = int(str(device).split(":")[-1])
    gpu_name = torch.cuda.get_device_name(i_device)
    low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
    if any(gpu in gpu_name for gpu in low_end_gpus) and "V100" not in gpu_name.upper():
        print(f"Your GPU {gpu_name} not support FP16 inference. Using FP32 instead.")
        return False
    return True


def full_inference_program(
    model_path,
    index_path,
    input_audio_path,
    output_path,
    export_format_rvc,
    split_audio,
    autotune,
    vocal_model,
    karaoke_model,
    dereverb_model,
    deecho,
    deecho_model,
    denoise,
    denoise_model,
    reverb,
    vocals_volume,
    instrumentals_volume,
    backing_vocals_volume,
    export_format_final,
    devices,
    pitch,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    pitch_extract,
    hop_lenght,
    reverb_room_size,
    reverb_damping,
    reverb_wet_gain,
    reverb_dry_gain,
    reverb_width,
    embedder_model,
    delete_audios,
    use_tta,
    batch_size,
    infer_backing_vocals,
    infer_backing_vocals_model,
    infer_backing_vocals_index,
    change_inst_pitch,
    pitch_back,
    filter_radius_back,
    index_rate_back,
    rms_mix_rate_back,
    protect_back,
    pitch_extract_back,
    hop_length_back,
    export_format_rvc_back,
    split_audio_back,
    autotune_back,
    embedder_model_back,
):
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        devices = devices.replace("-", " ")
        print(f"Number of GPUs available: {n_gpu}")
        first_device = devices.split()[0]
        fp16 = check_fp16_support(first_device)
    else:
        devices = "cpu"
        print("Using CPU")
        fp16 = False

    music_folder = os.path.splitext(os.path.basename(input_audio_path))[0]

    # Vocals Separation
    model_info = get_model_info_by_name(vocal_model)
    model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
    if not os.path.exists(model_ckpt_path):
        download_file(
            model_info["model_url"],
            model_info["path"],
            "model.ckpt",
        )
    config_json_path = os.path.join(model_info["path"], "config.yaml")
    if not os.path.exists(config_json_path):
        download_file(
            model_info["config_url"],
            model_info["path"],
            "config.yaml",
        )
    if not fp16:
        with open(model_info["config"], "r") as file:
            config = yaml.safe_load(file)

        config["training"]["use_amp"] = False

        with open(model_info["config"], "w") as file:
            yaml.safe_dump(config, file)
    store_dir = os.path.join(now_dir, "audio_files", music_folder, "vocals")
    inst_dir = os.path.join(now_dir, "audio_files", music_folder, "instrumentals")
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(inst_dir, exist_ok=True)
    input_audio_basename = os.path.splitext(os.path.basename(input_audio_path))[0]
    search_result = search_with_word(store_dir, "vocals")
    if search_result:
        print("Vocals already separated"),
    else:
        print("Separating vocals")
        command = [
            "python",
            os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
            "--model_type",
            model_info["type"],
            "--config_path",
            model_info["config"],
            "--start_check_point",
            model_info["model"],
            "--input_file",
            input_audio_path,
            "--store_dir",
            store_dir,
            "--flac_file",
            "--pcm_type",
            "PCM_16",
            "--extract_instrumental",
        ]

        if devices == "cpu":
            command.append("--force_cpu")
        else:
            device_ids = [str(int(device)) for device in devices.split()]
            command.extend(["--device_ids"] + device_ids)

        subprocess.run(command)
        os.rename(
            os.path.join(
                store_dir,
                search_with_two_words(
                    store_dir,
                    os.path.basename(input_audio_path).split(".")[0],
                    "instrumental",
                ),
            ),
            os.path.join(
                inst_dir,
                f"{os.path.basename(input_audio_path).split('.')[0]}_instrumentals.flac",
            ),
        )
    inst_file = os.path.join(
        inst_dir,
        search_with_two_words(
            inst_dir, os.path.basename(input_audio_path).split(".")[0], "instrumentals"
        ),
    )

    # karaoke separation
    model_info = get_model_info_by_name(karaoke_model)
    store_dir = os.path.join(now_dir, "audio_files", music_folder, "karaoke")
    os.makedirs(store_dir, exist_ok=True)
    vocals_path = os.path.join(now_dir, "audio_files", music_folder, "vocals")
    input_file = search_with_word(vocals_path, "vocals")
    karaoke_exists = search_with_word(store_dir, "karaoke") is not None

    if karaoke_exists:
        print("Backing vocals already separated")
    else:
        if input_file:
            input_file = os.path.join(vocals_path, input_file)
        print("Separating Backing vocals")
        if model_info["name"] == "Mel-Roformer Karaoke by aufr33 and viperx":
            model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
            if not os.path.exists(model_ckpt_path):
                download_file(
                    model_info["model_url"],
                    model_info["path"],
                    "model.ckpt",
                )
            config_json_path = os.path.join(model_info["path"], "config.yaml")
            if not os.path.exists(config_json_path):
                download_file(
                    model_info["config_url"],
                    model_info["path"],
                    "config.yaml",
                )
            if not fp16:
                with open(model_info["config"], "r") as file:
                    config = yaml.safe_load(file)

                config["training"]["use_amp"] = False

                with open(model_info["config"], "w") as file:
                    yaml.safe_dump(config, file)

            command = [
                "python",
                os.path.join(
                    now_dir, "programs", "music_separation_code", "inference.py"
                ),
                "--model_type",
                model_info["type"],
                "--config_path",
                model_info["config"],
                "--start_check_point",
                model_info["model"],
                "--input_file",
                input_file,
                "--store_dir",
                store_dir,
                "--flac_file",
                "--pcm_type",
                "PCM_16",
                "--extract_instrumental",
            ]

            if devices == "cpu":
                command.append("--force_cpu")
            else:
                device_ids = [str(int(device)) for device in devices.split()]
                command.extend(["--device_ids"] + device_ids)

            subprocess.run(command)
        else:
            separator = Separator(
                model_file_dir=os.path.join(now_dir, "models", "karaoke"),
                log_level=logging.WARNING,
                normalization_threshold=1.0,
                output_format="flac",
                output_dir=store_dir,
                vr_params={
                    "batch_size": batch_size,
                    "enable_tta": use_tta,
                },
            )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)
            karaoke_path = os.path.join(now_dir, "audio_files", music_folder, "karaoke")
            vocals_result = search_with_two_words(
                karaoke_path,
                os.path.basename(input_audio_path).split(".")[0],
                "Vocals",
            )
            instrumental_result = search_with_two_words(
                karaoke_path,
                os.path.basename(input_audio_path).split(".")[0],
                "Instrumental",
            )
            if "UVR-BVE-4B_SN-44100-1" in os.path.basename(vocals_result):
                os.rename(
                    os.path.join(karaoke_path, vocals_result),
                    os.path.join(
                        karaoke_path,
                        f"{os.path.basename(input_audio_path).split('.')[0]}_karaoke.flac",
                    ),
                )
            if "UVR-BVE-4B_SN-44100-1" in os.path.basename(instrumental_result):
                os.rename(
                    os.path.join(karaoke_path, instrumental_result),
                    os.path.join(
                        karaoke_path,
                        f"{os.path.basename(input_audio_path).split('.')[0]}_instrumental.flac",
                    ),
                )

    # dereverb
    model_info = get_model_info_by_name(dereverb_model)
    store_dir = os.path.join(now_dir, "audio_files", music_folder, "dereverb")
    os.makedirs(store_dir, exist_ok=True)
    karaoke_path = os.path.join(now_dir, "audio_files", music_folder, "karaoke")
    input_file = search_with_word(karaoke_path, "karaoke")
    noreverb_exists = search_with_word(store_dir, "noreverb") is not None
    if noreverb_exists:
        print("Reverb already removed")
    else:
        if input_file:
            input_file = os.path.join(karaoke_path, input_file)
        print("Removing reverb")
        if (
            model_info["name"] == "BS-Roformer Dereverb by anvuew"
            or model_info["name"] == "MDX23C DeReverb by aufr33 and jarredou"
        ):
            model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
            if not os.path.exists(model_ckpt_path):
                download_file(
                    model_info["model_url"],
                    model_info["path"],
                    "model.ckpt",
                )
            config_json_path = os.path.join(model_info["path"], "config.yaml")
            if not os.path.exists(config_json_path):
                download_file(
                    model_info["config_url"],
                    model_info["path"],
                    "config.yaml",
                )
            if not fp16:
                with open(model_info["config"], "r") as file:
                    config = yaml.safe_load(file)

                config["training"]["use_amp"] = False

                with open(model_info["config"], "w") as file:
                    yaml.safe_dump(config, file)
            command = [
                "python",
                os.path.join(
                    now_dir, "programs", "music_separation_code", "inference.py"
                ),
                "--model_type",
                model_info["type"],
                "--config_path",
                model_info["config"],
                "--start_check_point",
                model_info["model"],
                "--input_file",
                input_file,
                "--store_dir",
                store_dir,
                "--flac_file",
                "--pcm_type",
                "PCM_16",
            ]

            if devices == "cpu":
                command.append("--force_cpu")
            else:
                device_ids = [str(int(device)) for device in devices.split()]
                command.extend(["--device_ids"] + device_ids)

            subprocess.run(command)
        else:
            if model_info["arch"] == "vr":
                separator = Separator(
                    model_file_dir=os.path.join(now_dir, "models", "dereverb"),
                    log_level=logging.WARNING,
                    normalization_threshold=1.0,
                    output_format="flac",
                    output_dir=store_dir,
                    output_single_stem="No Reverb",
                    vr_params={
                        "batch_size": batch_size,
                        "enable_tta": use_tta,
                    },
                )
            else:
                separator = Separator(
                    model_file_dir=os.path.join(now_dir, "models", "dereverb"),
                    log_level=logging.WARNING,
                    normalization_threshold=1.0,
                    output_format="flac",
                    output_dir=store_dir,
                    output_single_stem="No Reverb",
                )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)
            dereverb_path = os.path.join(
                now_dir, "audio_files", music_folder, "dereverb"
            )
            search_result = search_with_two_words(
                dereverb_path,
                os.path.basename(input_audio_path).split(".")[0],
                "No Reverb",
            )
            if "UVR-DeEcho-DeReverb" in os.path.basename(
                search_result
            ) or "MDX Reverb HQ by FoxJoy" in os.path.basename(search_result):
                os.rename(
                    os.path.join(dereverb_path, search_result),
                    os.path.join(
                        dereverb_path,
                        f"{os.path.basename(input_audio_path).split('.')[0]}_noreverb.flac",
                    ),
                )

    # deecho
    store_dir = os.path.join(now_dir, "audio_files", music_folder, "deecho")
    os.makedirs(store_dir, exist_ok=True)
    if deecho:
        no_echo_exists = search_with_word(store_dir, "noecho") is not None
        if no_echo_exists:
            print("Echo already removed")
        else:
            print("Removing echo")
            model_info = get_model_info_by_name(deecho_model)

            dereverb_path = os.path.join(
                now_dir, "audio_files", music_folder, "dereverb"
            )
            noreverb_file = search_with_word(dereverb_path, "noreverb")

            input_file = os.path.join(dereverb_path, noreverb_file)

            separator = Separator(
                model_file_dir=os.path.join(now_dir, "models", "deecho"),
                log_level=logging.WARNING,
                normalization_threshold=1.0,
                output_format="flac",
                output_dir=store_dir,
                output_single_stem="No Echo",
                vr_params={
                    "batch_size": batch_size,
                    "enable_tta": use_tta,
                },
            )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)
            deecho_path = os.path.join(now_dir, "audio_files", music_folder, "deecho")
            search_result = search_with_two_words(
                deecho_path,
                os.path.basename(input_audio_path).split(".")[0],
                "No Echo",
            )
            if "UVR-De-Echo-Normal" in os.path.basename(
                search_result
            ) or "UVR-Deecho-Agggressive" in os.path.basename(search_result):
                os.rename(
                    os.path.join(deecho_path, search_result),
                    os.path.join(
                        deecho_path,
                        f"{os.path.basename(input_audio_path).split('.')[0]}_noecho.flac",
                    ),
                )

    # denoise
    store_dir = os.path.join(now_dir, "audio_files", music_folder, "denoise")
    os.makedirs(store_dir, exist_ok=True)
    if denoise:
        no_noise_exists = search_with_word(store_dir, "dry") is not None
        if no_noise_exists:
            print("Noise already removed")
        else:
            model_info = get_model_info_by_name(denoise_model)
            print("Removing noise")
            input_file = (
                os.path.join(
                    now_dir,
                    "audio_files",
                    music_folder,
                    "deecho",
                    search_with_word(
                        os.path.join(now_dir, "audio_files", music_folder, "deecho"),
                        "noecho",
                    ),
                )
                if deecho
                else os.path.join(
                    now_dir,
                    "audio_files",
                    music_folder,
                    "dereverb",
                    search_with_word(
                        os.path.join(now_dir, "audio_files", music_folder, "dereverb"),
                        "noreverb",
                    ),
                )
            )

            if (
                model_info["name"] == "Mel-Roformer Denoise Normal by aufr33"
                or model_info["name"] == "Mel-Roformer Denoise Aggressive by aufr33"
            ):
                model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
                if not os.path.exists(model_ckpt_path):
                    download_file(
                        model_info["model_url"],
                        model_info["path"],
                        "model.ckpt",
                    )
                config_json_path = os.path.join(model_info["path"], "config.yaml")
                if not os.path.exists(config_json_path):
                    download_file(
                        model_info["config_url"], model_info["path"], "config.yaml"
                    )
                if not fp16:
                    with open(model_info["config"], "r") as file:
                        config = yaml.safe_load(file)

                    config["training"]["use_amp"] = False

                    with open(model_info["config"], "w") as file:
                        yaml.safe_dump(config, file)
                command = [
                    "python",
                    os.path.join(
                        now_dir, "programs", "music_separation_code", "inference.py"
                    ),
                    "--model_type",
                    model_info["type"],
                    "--config_path",
                    model_info["config"],
                    "--start_check_point",
                    model_info["model"],
                    "--input_file",
                    input_file,
                    "--store_dir",
                    store_dir,
                    "--flac_file",
                    "--pcm_type",
                    "PCM_16",
                ]

                if devices == "cpu":
                    command.append("--force_cpu")
                else:
                    device_ids = [str(int(device)) for device in devices.split()]
                    command.extend(["--device_ids"] + device_ids)

                subprocess.run(command)
            else:
                separator = Separator(
                    model_file_dir=os.path.join(now_dir, "models", "denoise"),
                    log_level=logging.WARNING,
                    normalization_threshold=1.0,
                    output_format="flac",
                    output_dir=store_dir,
                    output_single_stem="No Noise",
                    vr_params={
                        "batch_size": batch_size,
                        "enable_tta": use_tta,
                    },
                )
                separator.load_model(model_filename=model_info["full_name"])
                separator.separate(input_file)
                search_result = search_with_two_words(
                    deecho_path,
                    os.path.basename(input_audio_path).split(".")[0],
                    "No Noise",
                )
                if "UVR Denoise" in os.path.basename(search_result):
                    os.rename(
                        os.path.join(deecho_path, search_result),
                        os.path.join(
                            deecho_path,
                            f"{os.path.basename(input_audio_path).split('.')[0]}_dry.flac",
                        ),
                    )

    # RVC
    denoise_path = os.path.join(now_dir, "audio_files", music_folder, "denoise")
    deecho_path = os.path.join(now_dir, "audio_files", music_folder, "deecho")
    dereverb_path = os.path.join(now_dir, "audio_files", music_folder, "dereverb")

    denoise_audio = search_with_two_words(
        denoise_path, os.path.basename(input_audio_path).split(".")[0], "dry"
    )
    deecho_audio = search_with_two_words(
        deecho_path, os.path.basename(input_audio_path).split(".")[0], "noecho"
    )
    dereverb = search_with_two_words(
        dereverb_path, os.path.basename(input_audio_path).split(".")[0], "noreverb"
    )

    if denoise_audio:
        final_path = os.path.join(
            now_dir, "audio_files", music_folder, "denoise", denoise_audio
        )
    elif deecho_audio:
        final_path = os.path.join(
            now_dir, "audio_files", music_folder, "deecho", deecho_audio
        )
    elif dereverb:
        final_path = os.path.join(
            now_dir, "audio_files", music_folder, "dereverb", dereverb
        )
    else:
        final_path = None

    store_dir = os.path.join(now_dir, "audio_files", music_folder, "rvc")
    os.makedirs(store_dir, exist_ok=True)
    print("Making RVC inference")
    output_rvc = os.path.join(
        now_dir,
        "audio_files",
        music_folder,
        "rvc",
        f"{os.path.basename(input_audio_path).split('.')[0]}_rvc.wav",
    )
    run_rvc_inference(
        model_path=model_path,
        index_path=index_path,
        input_audio_path=final_path,
        output_audio_path=output_rvc,
        embedder_model=embedder_model,
        pitch=pitch,
        f0_method=pitch_extract,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=rms_mix_rate,
        protect=protect,
        split_audio=split_audio,
        f0_autotune=autotune,
        hop_length=hop_lenght,
        export_format=export_format_rvc,
    )
    backing_vocals = os.path.join(
        karaoke_path, search_with_word(karaoke_path, "instrumental")
    )

    if infer_backing_vocals:
        print("Infering backing vocals")
        karaoke_path = os.path.join(now_dir, "audio_files", music_folder, "karaoke")
        instrumental_file = search_with_word(karaoke_path, "instrumental")
        backing_vocals = os.path.join(karaoke_path, instrumental_file)
        output_backing_vocals = os.path.join(
            karaoke_path, f"{input_audio_basename}_instrumental_output.wav"
        )
        run_rvc_inference(
            model_path=infer_backing_vocals_model,
            index_path=infer_backing_vocals_index,
            input_audio_path=backing_vocals,
            output_audio_path=output_backing_vocals,
            embedder_model=embedder_model_back,
            pitch=pitch_back,
            f0_method=pitch_extract_back,
            filter_radius=filter_radius_back,
            index_rate=index_rate_back,
            volume_envelope=rms_mix_rate_back,
            protect=protect_back,
            split_audio=split_audio_back,
            f0_autotune=autotune_back,
            hop_length=hop_length_back,
            export_format=export_format_rvc_back,
        )
        backing_vocals = output_backing_vocals

    # post process
    if reverb:
        add_audio_effects(
            os.path.join(
                now_dir,
                "audio_files",
                music_folder,
                "rvc",
                get_last_modified_file(
                    os.path.join(now_dir, "audio_files", music_folder, "rvc")
                ),
            ),
            reverb_room_size,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_damping,
            reverb_width,
            os.path.join(
                now_dir,
                "audio_files",
                music_folder,
                "rvc",
                os.path.basename(input_audio_path),
            ),
        )
    if change_inst_pitch != 0:
        print("Changing instrumental pitch")
        inst_path = os.path.join(
            now_dir,
            "audio_files",
            music_folder,
            "instrumentals",
            search_with_word(
                os.path.join(now_dir, "audio_files", music_folder, "instrumentals"),
                "instrumentals",
            ),
        )
        audio = AudioSegment.from_file(inst_path)

        factor = 2 ** (change_inst_pitch / 12)

        new_frame_rate = int(audio.frame_rate * factor)
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})

        audio = audio.set_frame_rate(audio.frame_rate)
        output_dir_pitch = os.path.join(
            now_dir, "audio_files", music_folder, "instrumentals"
        )
        output_path_pitch = os.path.join(
            output_dir_pitch, "inst_with_changed_pitch.flac"
        )
        audio.export(output_path_pitch, format="flac")

    # merge audios
    store_dir = os.path.join(now_dir, "audio_files", music_folder, "final")
    os.makedirs(store_dir, exist_ok=True)

    vocals_path = os.path.join(now_dir, "audio_files", music_folder, "rvc")
    vocals_file = get_last_modified_file(
        os.path.join(now_dir, "audio_files", music_folder, "rvc")
    )
    vocals_file = os.path.join(vocals_path, vocals_file)

    karaoke_path = os.path.join(now_dir, "audio_files", music_folder, "karaoke")
    karaoke_file = search_with_word(karaoke_path, "Instrumental") or search_with_word(
        karaoke_path, "instrumental"
    )
    karaoke_file = os.path.join(karaoke_path, karaoke_file)
    final_output_path = os.path.join(
        now_dir,
        "audio_files",
        music_folder,
        "final",
        f"{os.path.basename(input_audio_path).split('.')[0]}_final.{export_format_final.lower()}",
    )
    print("Merging audios")
    result = merge_audios(
        vocals_file,
        inst_file,
        backing_vocals,
        final_output_path,
        vocals_volume,
        instrumentals_volume,
        backing_vocals_volume,
        export_format_final,
    )
    print("Audios merged!")
    if delete_audios:
        main_directory = os.path.join(now_dir, "audio_files", music_folder)
        folder_to_keep = "final"
        for folder_name in os.listdir(main_directory):
            folder_path = os.path.join(main_directory, folder_name)
            if os.path.isdir(folder_path) and folder_name != folder_to_keep:
                shutil.rmtree(folder_path)
    return (
        f"Audio file {os.path.basename(input_audio_path).split('.')[0]} converted with success",
        result,
    )


def _format_title(title):
    """Sanitise a string into a safe directory/file name.

    Re-implements the legacy `programs.applio_code.rvc.lib.utils.format_title`
    helper, since the upstream `rvc` package does not ship an equivalent.
    The behaviour matches Applio: keep alphanumerics, whitespace and dashes;
    collapse every other non-word char (including dots, parens, etc.) into
    a single underscore, then lowercase the result.
    """
    title = re.sub(r"[^\w\s.-]", "", title)
    title = re.sub(r"[\s._-]+", "_", title).strip("_")
    return title.lower()


def download_model(link):
    """Download an RVC voice model (.pth + optional .index) from a URL.

    Accepts either:
      * a direct file URL (e.g. `https://example.com/voice.pth`), or
      * a HuggingFace model-card URL whose path ends in `.pth`
        (e.g. `https://huggingface.co/user/model/resolve/main/voice.pth`).
        In that case we also try to fetch a sibling `.index` file from the
        same directory listing if one exists.

    Files are stored under `logs/<model_name>/`, mirroring the original
    Applio behaviour that the Cover Maker UI relies on.
    """
    from rvc.utils import HF_download_file

    # Normalise HuggingFace URLs to the canonical /resolve/ form.
    url = link.strip().replace("/blob/", "/resolve/").replace("?download=true", "")
    filename = os.path.basename(url.split("?")[0])

    if not filename.lower().endswith(".pth"):
        return (
            f"URL does not point to a .pth file: {link!r}. "
            "Please provide a direct link to a voice model .pth file."
        )

    model_name = _format_title(filename[:-4])
    model_dir = os.path.join(now_dir, "logs", model_name)
    os.makedirs(model_dir, exist_ok=True)

    pth_dst = os.path.join(model_dir, filename)
    HF_download_file(url, pth_dst)

    # Best-effort: try to fetch a sibling .index file from the same remote
    # directory. We do not treat failure here as an error — many models
    # ship without an index.
    index_url = re.sub(r"/[^/]+\.pth$", "/" + model_name + ".index", url)
    if index_url != url:
        try:
            index_dst = os.path.join(model_dir, model_name + ".index")
            HF_download_file(index_url, index_dst)
        except Exception as e:
            print(f"[INFO] No sibling .index found for {filename} ({e})")

    return f"Model downloaded with success: {pth_dst}"


def download_music(link):
    os.makedirs(os.path.join(now_dir, "audio_files", "original_files"), exist_ok=True)
    command = [
        "yt-dlp",
        "-x",
        "--output",
        os.path.join(now_dir, "audio_files", "original_files", "%(title)s.%(ext)s"),
        link,
    ]
    subprocess.run(command)
    return "Music downloaded with success"
