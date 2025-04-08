@echo off

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo Uninstalling Triton dependency...
echo.
python_embeded\python.exe -m pip uninstall -y triton-windows triton sageattention torch torchvision torchaudio

echo.
echo Removing SageAttention build files...
echo.
rmdir /s /q "SageAttention"
rmdir /s /q "python_embeded\libs"
rmdir /s /q "python_embeded\include"

del /f /q "python_3.12.7_include_libs.zip"

echo Success!
pause