# app.py
import os
import sys
import shutil
import unicodedata
import regex as re
from pathlib import Path
from typing import Optional

import torch
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from gradio import Server
from gradio.data_classes import FileData

from core import (
    full_inference_program,
    download_model,
    download_music,
    _format_title,
)

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "audio_files", "original_files")
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4",
    "aac", "alac", "wma", "aiff", "webm", "ac3",
}

vocals_model_names = [
    "Mel-Roformer by KimberleyJSN",
    "BS-Roformer by ViperX",
    "MDX23C",
]
karaoke_models_names = [
    "Mel-Roformer Karaoke by aufr33 and viperx",
    "UVR-BVE",
]
denoise_models_names = [
    "Mel-Roformer Denoise Normal by aufr33",
    "Mel-Roformer Denoise Aggressive by aufr33",
    "UVR Denoise",
]
dereverb_models_names = [
    "MDX23C DeReverb by aufr33 and jarredou",
    "UVR-Deecho-Dereverb",
    "MDX Reverb HQ by FoxJoy",
    "BS-Roformer Dereverb by anvuew",
]
deeecho_models_names = ["UVR-Deecho-Normal", "UVR-Deecho-Aggressive"]


# ---------- Helpers ----------
def _list_models():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if file.endswith((".pth", ".onnx"))
        and not (file.startswith("G_") or file.startswith("D_"))
    ]
    return sorted(names)


def _list_indexes():
    return sorted(
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    )


def _list_audios():
    return sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    )


def _match_index(model_file_value):
    if not model_file_value:
        return ""
    model_folder = os.path.dirname(model_file_value)
    model_name = os.path.basename(model_file_value)
    pattern = r"^(.*?)_"
    match = re.match(pattern, model_name)
    for index_file in _list_indexes():
        if os.path.dirname(index_file) == model_folder:
            return index_file
        elif match and match.group(1) in os.path.basename(index_file):
            return index_file
        elif model_name in os.path.basename(index_file):
            return index_file
    return ""


def _output_path_fn(input_audio_path):
    base = os.path.basename(input_audio_path).rsplit(".", 1)[0]
    return os.path.join(
        os.path.dirname(input_audio_path), base + "_output.wav"
    )


def _format_title_ascii(title):
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    t = re.sub(r"[\u2500-\u257F]+", "", t)
    t = re.sub(r"[^\w\s.-]", "", t)
    t = re.sub(r"\s+", "_", t)
    return t


def _gpus():
    if torch.cuda.is_available():
        return "-".join(map(str, range(torch.cuda.device_count())))
    return "-"


# ---------- App ----------
app = Server()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio_files", StaticFiles(directory="audio_files"), name="audio_files")


@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open(os.path.join(now_dir, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


# ---------------- Meta / lists ----------------
@app.api(name="meta")
def meta() -> dict:
    return {
        "models": _list_models(),
        "indexes": _list_indexes(),
        "audios": _list_audios(),
        "vocals_models": sorted(vocals_model_names),
        "karaoke_models": sorted(karaoke_models_names),
        "denoise_models": sorted(denoise_models_names),
        "dereverb_models": sorted(dereverb_models_names),
        "deeecho_models": sorted(deeecho_models_names),
        "devices": _gpus(),
    }


@app.api(name="refresh_lists")
def refresh_lists() -> dict:
    return {
        "models": _list_models(),
        "indexes": _list_indexes(),
        "audios": _list_audios(),
    }


@app.api(name="match_index")
def match_index(model_file: str) -> str:
    return _match_index(model_file)


# ---------------- Upload audio ----------------
@app.api(name="upload_audio")
async def upload_audio(file: FileData) -> dict:
    src = file["path"] if isinstance(file, dict) else file.path
    formatted = _format_title_ascii(os.path.basename(src))
    target = os.path.join(audio_root_relative, formatted)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        os.remove(target)
    shutil.copy(src, target)
    return {"audio": target, "output": _output_path_fn(target)}


@app.api(name="delete_outputs")
def delete_outputs() -> str:
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and "_output" in name:
                os.remove(os.path.join(root, name))
    return "Outputs cleared"


@app.api(name="upload_file")
async def upload_file(file: FileData) -> str:
    """Generic file receive — returns the temp path."""
    return file["path"] if isinstance(file, dict) else file.path


# ---------------- Full inference ----------------
@app.api(name="full_inference")
def full_inference(
    model_file: str,
    index_file: str,
    audio: str,
    output_path: str,
    export_format_rvc: str,
    split_audio: bool,
    autotune: bool,
    vocal_model: str,
    karaoke_model: str,
    dereverb_model: str,
    deecho: bool,
    deeecho_model: str,
    denoise: bool,
    denoise_model: str,
    reverb: bool,
    vocals_volume: float,
    instrumentals_volume: float,
    backing_vocals_volume: float,
    export_format_final: str,
    devices: str,
    pitch: int,
    filter_radius: int,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
    pitch_extract: str,
    hop_length: int,
    reverb_room_size: float,
    reverb_damping: float,
    reverb_wet_gain: float,
    reverb_dry_gain: float,
    reverb_width: float,
    embedder_model: str,
    delete_audios: bool,
    use_tta: bool,
    batch_size: int,
    infer_backing_vocals: bool,
    infer_backing_vocals_model: str,
    infer_backing_vocals_index: str,
    change_inst_pitch: int,
    pitch_back: int,
    filter_radius_back: int,
    index_rate_back: float,
    rms_mix_rate_back: float,
    protect_back: float,
    pitch_extract_back: str,
    hop_length_back: int,
    export_format_rvc_back: str,
    split_audio_back: bool,
    autotune_back: bool,
    embedder_model_back: str,
):
    return full_inference_program(
        model_file, index_file, audio, output_path, export_format_rvc, split_audio,
        autotune, vocal_model, karaoke_model, dereverb_model, deecho, deeecho_model,
        denoise, denoise_model, reverb, vocals_volume, instrumentals_volume,
        backing_vocals_volume, export_format_final, devices, pitch, filter_radius,
        index_rate, rms_mix_rate, protect, pitch_extract, hop_length,
        reverb_room_size, reverb_damping, reverb_wet_gain, reverb_dry_gain,
        reverb_width, embedder_model, delete_audios, use_tta, batch_size,
        infer_backing_vocals, infer_backing_vocals_model, infer_backing_vocals_index,
        change_inst_pitch, pitch_back, filter_radius_back, index_rate_back,
        rms_mix_rate_back, protect_back, pitch_extract_back, hop_length_back,
        export_format_rvc_back, split_audio_back, autotune_back, embedder_model_back,
    )


# ---------------- Dl model ----------------
@app.api(name="download_model_url")
def download_model_url(link: str) -> str:
    return download_model(link)


@app.api(name="save_drop_model")
def save_drop_model(dropbox: FileData) -> str:
    path = dropbox["path"] if isinstance(dropbox, dict) else dropbox.path
    if "pth" not in path and "index" not in path:
        raise ValueError("The file you dropped is not a valid model file.")
    file_name = _format_title(os.path.basename(path))
    if ".pth" in path:
        model_name = _format_title(file_name.split(".pth")[0])
    else:
        if "v2" not in path and "added_" not in path and "_nprobe_1_" not in path:
            model_name = _format_title(file_name.split(".index")[0])
        else:
            if "v2" not in path:
                if "_nprobe_1_" in file_name and "_v1" in file_name:
                    model_name = _format_title(
                        file_name.split("_nprobe_1_")[1].split("_v1")[0]
                    )
                elif "added_" in file_name and "_v1" in file_name:
                    model_name = _format_title(
                        file_name.split("added_")[1].split("_v1")[0]
                    )
            else:
                if "_nprobe_1_" in file_name and "_v2" in file_name:
                    model_name = _format_title(
                        file_name.split("_nprobe_1_")[1].split("_v2")[0]
                    )
                elif "added_" in file_name and "_v2" in file_name:
                    model_name = _format_title(
                        file_name.split("added_")[1].split("_v2")[0]
                    )
    model_name = re.sub(r"\d+[se]", "", model_name)
    if "__" in model_name:
        model_name = model_name.replace("__", "")
    model_path = os.path.join(now_dir, "logs", model_name)
    os.makedirs(model_path, exist_ok=True)
    dst = os.path.join(model_path, file_name)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy(path, dst)
    return f"{file_name} saved in {model_path}"


# ---------------- Dl music ----------------
@app.api(name="download_music_url")
def download_music_url(link: str) -> str:
    return download_music(link)


if __name__ == "__main__":
    app.launch(
        show_error=True,
        share=True,
    )
