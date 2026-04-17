@echo off
:: Launch the playback recording script (requires NI-DAQmx hardware + driver)
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" ni_python
python "%~dp0GUI_based\Playback Experiments\01_Record_and_Playback_multichan.py"
pause
