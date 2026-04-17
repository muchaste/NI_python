@echo off
:: Launch the multi-channel recording script (requires NI-DAQmx hardware + driver)
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" ni_python
python "%~dp0GUI_based\01_Record_ai_multichan.py"
pause
