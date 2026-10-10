@echo off
cd /d "%~dp0\.."
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b %errorlevel%
pyinstaller --noconfirm --clean --onefile --windowed --name "iGEM Motion Studio" --paths src src\merged_abm_desktop_app.py
if errorlevel 1 exit /b %errorlevel%
echo Build complete: dist\iGEM Motion Studio.exe
