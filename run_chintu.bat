@echo off
setlocal

if not exist venv\Scripts\activate (
    python -m venv venv
)
call venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if not exist models\ (
    python scripts\doctor.py
    python scripts\download_models.py
)

python chintu_assistant.py
pause