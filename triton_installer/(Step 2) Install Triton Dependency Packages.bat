@echo off

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set INCLUDE_LIBS_URL=https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip

setlocal enabledelayedexpansion

echo Installing Visual Studio Build Tools...
echo.
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK.20348"


echo.
echo Find installed cuda...
echo.
for /f "tokens=5 delims= " %%A in ('nvcc --version ^| findstr /C:"release"') do (
    for /f "tokens=1 delims=," %%B in ("%%A") do set cuda_version=%%B
)
for /f "tokens=1 delims=." %%a in ("%cuda_version%") do set cuda_major=%%a
for /f "tokens=2 delims=." %%b in ("%cuda_version%") do set cuda_minor=%%b

set cuda_version=!cuda_major!.!cuda_minor!


echo.
echo cuda version: %cuda_version%
echo.


echo.
echo Upgrading pip...
echo.
python_embeded\python.exe -m pip install --upgrade pip


echo.
echo Installing PyTorch...
echo.
::python_embeded\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu%cuda_version:.=%
::python_embeded\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu%cuda_version:.=%
python_embeded\python.exe -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu%cuda_version:.=%

echo.
echo Installing Triton...
echo.
python_embeded\python.exe -m pip install -U --pre triton-windows


echo.
echo Downloading Python include/libs from URL...
echo.
for %%F in ("%INCLUDE_LIBS_URL%") do (
    set FILE_NAME=%%~nxF
)
curl -L -o "%FILE_NAME%" "%INCLUDE_LIBS_URL%"


echo.
echo Extracting Python include/libs using tar...
echo.
tar -xf "%FILE_NAME%" -C python_embeded


echo.
echo Cloning SageAttention repository...
echo.
git clone https://github.com/thu-ml/SageAttention


echo.
echo Installing SageAttention...
echo.
python_embeded\python.exe -s -m pip install -e SageAttention


echo.
echo Cloning flow2-wan-video repository...
echo.
REM git clone https://github.com/Flow-Two/flow2-wan-video.git ComfyUI\custom_nodes\flow2-wan-video
python_embeded\python.exe -m pip install -r "ComfyUI\custom_nodes\flow2-wan-video\requirements.txt"

echo.
echo Cloning ComfyUI-VideoHelperSuite repository...
echo.
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git ComfyUI\custom_nodes\ComfyUI-VideoHelperSuite
python_embeded\python.exe -m pip install -r "ComfyUI\custom_nodes\ComfyUI-VideoHelperSuite\requirements.txt"

echo.
echo Success!
echo.

endlocal
pause