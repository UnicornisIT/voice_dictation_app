@echo off
setlocal
cd /d %~dp0
chcp 65001 > nul

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)

set "PY=.venv\Scripts\python.exe"

"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt Pillow

echo Checking binary wheels...
"%PY%" -c "from PIL import Image; import av, lxml.etree, numpy, ctranslate2; from PySide6.QtCore import Qt; from faster_whisper import WhisperModel; print('Import check OK')"
if errorlevel 1 (
    echo.
    echo Dependency import check failed. Reinstalling native wheels...
    "%PY%" -m pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt Pillow
    "%PY%" -c "from PIL import Image; import av, lxml.etree, numpy, ctranslate2; from PySide6.QtCore import Qt; from faster_whisper import WhisperModel; print('Import check OK')"
    if errorlevel 1 (
        echo.
        echo Dependency repair failed. Delete the .venv folder and run this script again from a normal, non-admin terminal.
        pause
        exit /b 1
    )
)

echo Converting icon...
"%PY%" convert_icon.py
if errorlevel 1 exit /b 1

echo Building executable...
"%PY%" -m PyInstaller --noconfirm --clean --windowed --name VoiceDictationSTT --icon=app_icon.ico --add-data "app_icon.ico;." --collect-submodules faster_whisper --collect-data faster_whisper --collect-binaries ctranslate2 --collect-binaries av --collect-binaries onnxruntime --hidden-import av._core main.py
if errorlevel 1 exit /b 1

echo.
echo Build finished. EXE folder: dist\VoiceDictationSTT\VoiceDictationSTT.exe
echo.
pause
