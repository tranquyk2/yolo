@echo off
REM ============================================================
REM  build_gpu.bat
REM  Build ban GPU cua QR Detection System bang PyInstaller.
REM  Chay file nay tu THU MUC GOC chua main.py, gui.py, ...
REM
REM  Yeu cau truoc khi chay:
REM    1. Da cai Python 3.11 (venv rieng cho build, khong dinh
REM       Python he thong).
REM    2. Da activate venv build va cai:
REM         pip install pyinstaller
REM         pip install torch --index-url https://download.pytorch.org/whl/cu121
REM         pip install -r requirements.txt
REM       (chon dung version CUDA khop driver GPU tren may build/may dich)
REM ============================================================

setlocal

REM ---- Thu muc dich cuoi cung ----
set "OUTPUT_ROOT=D:\Desktop\QR_Detector"
set "APP_NAME=QR_GPU"

echo ============================================================
echo  Building %APP_NAME% (GPU build)
echo ============================================================

REM ---- Xoa build cu neu co, tranh dinh cache PyInstaller ----
if exist "build" rmdir /s /q "build"
if exist "dist\%APP_NAME%" rmdir /s /q "dist\%APP_NAME%"
if exist "%APP_NAME%.spec" del /q "%APP_NAME%.spec"

REM ---- Build ----
pyinstaller --onedir --windowed --name %APP_NAME% ^
  --collect-all torch ^
  --collect-all nvidia ^
  --collect-all customtkinter ^
  --collect-all ultralytics ^
  --collect-all cv2 ^
  --collect-all zxingcpp ^
  --add-data "models;models" ^
  --add-data "config.json;." ^
  main.py

if errorlevel 1 (
    echo.
    echo [LOI] Build that bai. Xem log ben tren de biet nguyen nhan.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Build xong. Dang copy vao %OUTPUT_ROOT% ...
echo ============================================================

REM ---- Tao thu muc dich neu chua co ----
if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"

REM ---- Xoa ban cu trong thu muc dich (neu co) roi copy ban moi ----
if exist "%OUTPUT_ROOT%\%APP_NAME%" rmdir /s /q "%OUTPUT_ROOT%\%APP_NAME%"
xcopy /e /i /y "dist\%APP_NAME%" "%OUTPUT_ROOT%\%APP_NAME%"

echo.
echo ============================================================
echo  XONG! App nam tai:
echo    %OUTPUT_ROOT%\%APP_NAME%\%APP_NAME%.exe
echo.
echo  De chuyen sang may khac: copy nguyen thu muc
echo    %OUTPUT_ROOT%\%APP_NAME%
echo  (bao gom ca folder _internal ben trong, khong duoc thieu)
echo ============================================================
pause