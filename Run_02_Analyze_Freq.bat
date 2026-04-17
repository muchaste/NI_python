@echo off
:: Launch the frequency analysis tool (no NI hardware required)
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" ni_python
python "%~dp0GUI_based\02_Analyze_ai_multichan_freq.py"
pause
