# RVC Integration — Migration Notes

This fork of [Eddycrack864/RVC-AI-Cover-Maker-UI](https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI)
has been refactored so that **voice conversion is now powered by the standalone
[`rvc` package](https://github.com/uziproj/rvc) maintained by uziproj**, instead
of the previously-vendored `programs/applio_code/rvc/` tree (which was missing
from this fork entirely).

## Why

The original `core.py` and `tabs/download_model.py` imported from
`programs.applio_code.rvc.infer.infer.VoiceConverter`,
`programs.applio_code.rvc.lib.tools.model_download.model_download_pipeline`,
`programs.applio_code.rvc.configs.config.Config`, and
`programs.applio_code.rvc.lib.utils.format_title` — but the
`programs/applio_code/` directory was never vendored into this repo, so
importing `core` would have raised `ModuleNotFoundError` immediately.

Switching to the pip-installable `rvc` package from uziproj gives us:

1. **A real, installable API** — `pip install git+https://github.com/uziproj/rvc.git`
2. **Lazily-auto-downloaded predictor + embedder models** — no separate
   `prerequisites_download.py` step.
3. **20+ pitch-extraction methods**, hybrid f0, formant shifting, autotune,
   REST API, CLI, ONNX export.
4. **Active maintenance** — the package's README lists dozens of bugs the
   upstream fixed vs. the original Applio codebase.

## What changed

### `core.py`
- **Removed imports:**
  ```python
  from programs.applio_code.rvc.infer.infer import VoiceConverter
  from programs.applio_code.rvc.lib.tools.model_download import model_download_pipeline
  ```
- **Added lazy imports** (inside helper functions, so the heavy torch /
  fairseq stack is only loaded when an RVC inference is actually requested):
  ```python
  from rvc import Config, run_inference_script
  from rvc.utils import HF_download_file
  ```
- **New helper `get_config(...)`** — wraps `rvc.Config(...)` with the cover
  maker's fp16 auto-detection logic.
- **New helper `run_rvc_inference(...)`** — wraps `rvc.run_inference_script`.
  Translates the original Cover Maker parameters to the uziproj/rvc
  function-form API. Also applies two backwards-compatibility aliases:
  - `"contentvec"` → `"contentvec_base"` (and the same for the other
    language-specific hubert-base embedders, which the `rvc` package
    expects with underscores rather than hyphens).
  - `"crepe"` → `"crepe-full"` (the uziproj package requires a size suffix;
    the Cover Maker UI exposed the bare name).
  - Lowercases `export_format` to match the `rvc` package's expected values
    (`"wav"`, `"flac"`, `"mp3"`, `"ogg"`, `"m4a"`).
- **Replaced the two `inference_vc.convert_audio(...)` call sites** in
  `full_inference_program(...)` with `run_rvc_inference(...)` calls.
- **Re-implemented `download_model(link)`** locally — the original called
  Applio's `model_download_pipeline`. Our version uses
  `rvc.utils.HF_download_file` to download the `.pth` into
  `logs/<sanitised-name>/`, and best-effort tries to fetch a sibling
  `.index` file from the same HuggingFace directory.
- **Re-implemented `_format_title(title)`** locally — replaces Applio's
  `programs.applio_code.rvc.lib.utils.format_title`. Used by
  `tabs/download_model.py`'s `save_drop_model` handler to sanitise the
  name of a user-dropped `.pth` / `.index` file into a safe directory name.

### `tabs/download_model.py`
- **Removed import:** `from programs.applio_code.rvc.lib.utils import format_title`
- **Added import:** `from core import download_model, _format_title`
- **Renamed every `format_title(...)` call to `_format_title(...)`** so the
  tab continues to use the new local helper.

### `requirements.txt`
- Keeps the `git+https://github.com/uziproj/rvc.git` line (this was already
  present in the original repo's `requirements.txt`, but it was unused because
  `core.py` imported from `programs.applio_code.*` instead). Now it actually
  drives the RVC stack.
- Added explicit `pedalboard`, `pydub`, `regex`, `PyYAML` entries — these are
  used directly by `core.py` but were previously transitively pulled in by
  the Applio vendored tree. We list them explicitly now so the cover maker
  installs cleanly when the vendored tree is gone.
- The `rvc` package itself pulls in `torch`, `torchaudio`, `fairseq` (via
  `rvc.lib.embedders`), `faiss-cpu`, `librosa`, `soundfile`, `einops`,
  `transformers`, `praat-parselmouth`, `tqdm`, `requests`, `edge-tts`,
  `fastapi`, `uvicorn`, `python-multipart`, etc.

### `run.sh` / `run.bat`
- Bumped the Conda environment Python from **3.9 → 3.10** (uziproj/rvc
  requires `>=3.10, <3.13`).
- Install the `rvc` package from git **with deps** (so fairseq, einops,
  faiss-cpu, etc. are pulled in automatically). The remaining entries in
  `requirements.txt` are installed with `--no-deps` to avoid clobbering
  torch with a CUDA-less build.
- Removed the `python programs/applio_code/rvc/lib/tools/prerequisites_download.py`
  step — the `rvc` package lazily auto-downloads predictor and embedder
  models from HuggingFace on first inference via
  `rvc.utils.check_predictors` / `check_embedders`.

### `update.sh` / `update.bat`
- Changed the keep-list from `programs/applio_code/rvc/models` to
  `assets/models` (the `rvc` package stores downloaded predictor / embedder
  models under `<cwd>/assets/models/`, as defined by `rvc.lib.config.PREDICTOR_MODEL`).
- Updated the repo URL to `asukaa2/RVC-Cover-Maker`.

## How to run

```bash
cd RVC-Cover-Maker
./run.sh        # Linux / macOS
run.bat         # Windows
```

Or, in an existing Python 3.10–3.12 environment:

```bash
pip install -r requirements.txt
python main.py --open
```

On the first RVC inference, the `rvc` package will print messages like
`Predictor model not found, downloading: rmvpe.pt` and `Hubert model not
found, downloading: contentvec_base.pt` — these are normal and only happen
once.

## Smoke test

A standalone test that mocks the heavy `rvc` package and verifies the
end-to-end integration lives at:

```
/home/z/my-project/scripts/verify_integration.py
```

Run it with:

```bash
python3 /home/z/my-project/scripts/verify_integration.py
```

It asserts that:
1. `core` imports cleanly when `rvc` is importable.
2. `_format_title` matches Applio's semantics (5 sample inputs).
3. `download_model(link)` produces the right HuggingFace URL and writes the
   `.pth` under `logs/<sanitised-name>/`.
4. `run_rvc_inference(...)` forwards every parameter to
   `rvc.run_inference_script` exactly as the uziproj/rvc API expects, and
   applies the aliasing for `contentvec` → `contentvec_base` and
   `crepe` → `crepe-full`.
5. `tabs/download_model.py` imports cleanly with no `applio_code` reference.

## Limitations / notes

- The `rvc.Config` class is a `@singleton` in the upstream package, so
  the **first** call to `Config(embedder_model=..., f0_method=...)` wins.
  If the user changes the embedder mid-session in the UI, the change will
  be silently ignored. This is a documented upstream behaviour — to switch
  embedders you must restart the cover maker.
- The `rvc` package's `VoiceConverter.convert_audio` signature does NOT
  include the legacy `f0_file`, `embedder_model_custom`, or `model_path`
  parameters that the original Cover Maker passed. The model is now loaded
  via the `pth_path=` kwarg of `run_inference_script` (one-shot, fresh
  model per call) — slightly less efficient than `rvc.RVClass` (which
  loads once and reuses), but matches the original Cover Maker semantics
  where two different model paths may be passed in the same
  `full_inference_program` call.
