@echo off
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)
python chintu_assistant.py
pause