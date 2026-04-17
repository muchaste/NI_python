@echo off
:: Launch the pulse analysis tool (no NI hardware required)
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" ni_python
python "%~dp0GUI_based\03_Analyze_ai_multichan_pulse.py"
pause
