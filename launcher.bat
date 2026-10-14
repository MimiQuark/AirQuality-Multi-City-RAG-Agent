@echo off
chcp 65001 >/dev/null
title 多城市空气质量监测 AI 系统 - 启动器
cd /d "%~dp0"

echo ============================================
echo  多城市空气质量监测 AI 系统 v2.1
echo ============================================
echo.

REM 1. 检查 Python
python --version >/dev/null 2>&1
if errorlevel 1 (
    echo [错误] 未找到 python，请检查环境变量 PATH
    pause
    exit /b 1
)
for /f "delims=" %%v in ('python --version 2^>^&1') do echo [检查] 使用 %%v

REM 2. 预检 tkinter（GUI 必需，缺失是闪退的常见原因）
python -c "import tkinter" 2>_preflight.err
if errorlevel 1 (
    echo [错误] 当前 Python 缺少 tkinter（图形界面库），无法启动 GUI：
    type _preflight.err
    del _preflight.err 2>/dev/null
    echo.
    echo 解决方案：改用 启动服务-免GUI.bat，或重装 Python 时勾选 tcl/tk。
    pause
    exit /b 1
)
del _preflight.err 2>/dev/null

REM 3. 检查 pythonw
where pythonw >/dev/null 2>&1
if errorlevel 1 (
    echo [错误] 未找到 pythonw.exe
    pause
    exit /b 1
)

REM 4. 清除旧崩溃日志后启动 GUI
del launcher_error.log 2>/dev/null
start "" pythonw launcher.pyw

REM 5. 等待 4 秒，若 GUI 闪退会留下 launcher_error.log，直接显示原因
timeout /t 4 /nobreak >/dev/null
if exist launcher_error.log (
    echo.
    echo [错误] 启动器闪退，崩溃日志如下：
    echo --------------------------------------------
    type launcher_error.log
    echo --------------------------------------------
    echo.
    echo 请把以上内容截图反馈，或改用 启动服务-免GUI.bat
    pause
    exit /b 1
)

REM GUI 正常运行，本窗口自动关闭
exit /b 0
