@echo off
rem Build the Windows installer of StrainLink. Double-click this file, or run it in a terminal.
rem Needs: Windows 10 or 11 (64-bit), Python 3.12 from python.org, and (for the installer) Inno Setup 6.
rem Output: dist\StrainLink-<version>-Windows-Setup.exe   (see packaging\BUILD.md)

setlocal enableextensions
cd /d "%~dp0\.."

echo.
echo === 1 of 6: Python environment ===
rem An automatic build (GitHub Actions) names its own Python in BUILD_PYTHON; on a PC the
rem script makes a private environment in build\winenv from Python 3.12.
if defined BUILD_PYTHON (
    set PY=%BUILD_PYTHON%
    goto :have_python
)
where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/windows/
    echo and tick "Add python.exe to PATH" in its installer. Then run this file again.
    goto :failed
)
if not exist build\winenv\Scripts\python.exe (
    py -3.12 -m venv build\winenv
    if errorlevel 1 (
        echo Python 3.12 was not found. This build is set up for Python 3.12: install it from python.org.
        goto :failed
    )
)
set PY=build\winenv\Scripts\python.exe
:have_python

echo.
echo === 2 of 6: Libraries (a few minutes the first time, about 600 MB) ===
%PY% -m pip install --upgrade pip
if errorlevel 1 goto :failed
%PY% -m pip install -e . pyinstaller==6.22.3
if errorlevel 1 goto :failed

echo.
echo === 3 of 6: Icon and chart library ===
%PY% -c "from strainlink.app.main import _ensure_chart_library; _ensure_chart_library()"
if errorlevel 1 goto :failed
%PY% packaging\icons.py
if errorlevel 1 goto :failed

echo.
echo === 4 of 6: Building the program (several minutes) ===
if exist dist\StrainLink rmdir /s /q dist\StrainLink
%PY% -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\StrainLink.spec
if errorlevel 1 goto :failed

echo.
echo === 5 of 6: Self-test of the built program ===
start "" /wait dist\StrainLink\StrainLink.exe --self-test
set SELFTEST=%errorlevel%
if exist "%APPDATA%\StrainLink\selftest.log" type "%APPDATA%\StrainLink\selftest.log"
if not "%SELFTEST%"=="0" (
    echo.
    echo The built program FAILED its self-test. Send the text above to whoever maintains StrainLink.
    goto :failed
)

echo.
echo === 6 of 6: Installer ===
for /f "delims=" %%v in ('%PY% -c "import strainlink; print(strainlink.__version__)"') do set VERSION=%%v
set ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" set ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" if defined CI goto :failed
if not exist "%ISCC%" (
    echo Inno Setup 6 was not found, so no Setup.exe was made.
    echo The program itself is ready in dist\StrainLink  ^(run StrainLink.exe^).
    echo To make the installer: install Inno Setup 6 ^(free^) from https://jrsoftware.org/isdl.php
    echo and run this file again.
    goto :done
)
"%ISCC%" /DAppVersion=%VERSION% packaging\strainlink.iss
if errorlevel 1 goto :failed
echo.
echo Installer ready: dist\StrainLink-%VERSION%-Windows-Setup.exe

:done
echo.
echo Finished.
rem An automatic build (CI) has nobody to press a key.
if not defined CI pause
exit /b 0

:failed
echo.
echo The build did not finish. Nothing was installed on this computer outside this folder.
if not defined CI pause
exit /b 1
