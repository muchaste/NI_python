@echo off
:: Launch the multi-channel recording + LED sync script (requires NI-DAQmx hardware + driver)
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" ni_python
python "%~dp0GUI_based\04_Record_ai_multichan_sync_LED.py"
pause
