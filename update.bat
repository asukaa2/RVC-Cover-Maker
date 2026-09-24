@echo off
setlocal

REM Define the repository URL
set REPO_URL=https://github.com/asukaa2/RVC-Cover-Maker

REM Navigate to the directory where the script is located
cd /d %~dp0

REM After the migration to the `rvc` pip package, the vendored
REM `programs/applio_code/` directory is gone. Predictor / embedder
REM models now live in `assets/models/` (managed by the `rvc` package
REM itself, which auto-downloads them from HuggingFace on first use).
REM Loop through all top-level directories except the ones we keep.
for /d %%D in (*) do (
    if /i not "%%D"=="env" if /i not "%%D"=="logs" if /i not "%%D"=="audio_files" if /i not "%%D"=="assets" (
        echo Deleting directory %%D
        rmdir /s /q "%%D"
    )
)

REM Inside `assets/`, only keep the `models` subdirectory.
if exist assets (
    for /d %%D in (assets\*) do (
        if /i not "%%D"=="assets\models" (
            echo Deleting directory %%D
            rmdir /s /q "%%D"
        )
    )
)

REM Loop through all files and delete them
for %%F in (*) do (
    if not "%%F"=="update.bat" (
        echo Deleting file %%F
        del /q "%%F"
    )
)

REM Initialize a new git repository if it doesn't exist
if not exist .git (
    git init
    git remote add origin %REPO_URL%
)

REM Fetch the latest changes from the repository
git fetch origin

REM Reset the working directory to match the latest commit
git reset --hard origin/main

pause
endlocal
