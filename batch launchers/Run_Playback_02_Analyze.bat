@echo off
:: Launch the playback analysis tool (no NI hardware required)
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" ni_python
python "%~dp0GUI_based\Playback Experiments\02_Analyze_multichan_Record_and_Playback.py"
pause
