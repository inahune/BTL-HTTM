@echo off
chcp 65001 >nul
echo Đang khởi động Hệ thống Web Phát hiện Người Đuối Nước...
start "" "http://127.0.0.1:8501"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0web_app\server.py"
) else (
    "C:\Users\ASUS\AppData\Local\Programs\Python\Python313\python.exe" "%~dp0web_app\server.py"
)
pause
