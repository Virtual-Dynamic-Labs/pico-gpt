@echo off
REM Pico GPT CLI Client Launcher for Windows
REM This batch file makes it easy to run the CLI client

echo.
echo ===============================================
echo   Pico GPT CLI Client Launcher
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "cli_client.py" (
    echo ERROR: cli_client.py not found
    echo Please run this batch file from the pico-gpt directory
    pause
    exit /b 1
)

REM Check if model files exist
if not exist "*.pt" (
    echo WARNING: No model files (.pt) found
    echo Please train a model first using train_large.py or train_small.py
    echo.
)

REM Parse command line arguments
set ARGS=
:parse_args
if "%1"=="" goto run_client
set ARGS=%ARGS% %1
shift
goto parse_args

:run_client
REM Run the CLI client
echo Starting Pico GPT CLI Client...
echo.
python cli_client.py %ARGS%

REM Pause if there was an error
if errorlevel 1 (
    echo.
    echo Program exited with error code %errorlevel%
    pause
)
