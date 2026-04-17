@echo off
:: Launch the WAV/FM analysis tool (no NI hardware required)
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" ni_python
python "%~dp0GUI_based\Playback Experiments\03_Analyze_wav_multichan_FM.py"
pause
