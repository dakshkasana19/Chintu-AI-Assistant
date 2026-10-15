@echo off
setlocal

if not exist venv\Scripts\activate (
    python -m venv venv
    if errorlevel 1 goto :failure
)
call venv\Scripts\activate
if errorlevel 1 goto :failure

python -m pip install -r requirements.txt
if errorlevel 1 goto :failure

if not exist models\silero_vad.onnx (
    python scripts\doctor.py
    if errorlevel 1 goto :failure
    python scripts\download_models.py
    if errorlevel 1 echo WARNING: Silero VAD download failed; Chintu will try the WebRTC VAD fallback.
)

if not exist models\wake_models.ready (
    python scripts\download_models.py --wake
    if errorlevel 1 goto :failure
)

python chintu_assistant.py
if errorlevel 1 goto :failure
pause
exit /b 0

:failure
echo Chintu setup or startup failed. See the error above.
pause
exit /b 1