@echo off
setlocal
cd /d "%~dp0"
call "D:\07_ide_base\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" -std=c++17 --expt-relaxed-constexpr -Xcompiler=/Zc:preprocessor,/utf-8 -arch=sm_89 -I "E:\000wsp\01repo\08sutlass\cutlass\include" gemm_128x256x64.cu -o cute_minimal.exe
exit /b %errorlevel%
