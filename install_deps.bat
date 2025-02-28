@echo off
setlocal

REM -- FIND PYTHON INTERPRETER --
SET PY_INTERPRETER=

echo [INFO] Discovering Python Interpreter
if defined VIRTUAL_ENV (
    echo [INFO] Found active virtual environment
    set "PY_INTERPRETER=%VIRTUAL_ENV%\Scripts\python.exe"
    goto py_install_deps
)

py -V > NUL
if %ERRORLEVEL%==0 (
    echo [INFO] Found python launcher. Setting up virtual environment.
    py -m venv .\venv
    if %ERRORLEVEL%==0 (
        SET PY_INTERPRETER=.\venv\Scripts\python.exe
        goto py_install_deps
    ) else (
        echo [WARN] Failed to set up virtual environment. Using standard Python installation.
        SET PY_INTERPRETER=py
        goto py_install_deps
    )
)

python -V > NUL
if %ERRORLEVEL%==0 (
    echo [INFO] Found python installation. Setting up virtual environment.
    py -m venv .\venv
    if %ERRORLEVEL%==0 (
        SET PY_INTERPRETER=.\venv\Scripts\python.exe
        goto py_install_deps
    ) else (
        echo [WARN] Failed to set up virtual environment. Using standard Python installation.
        SET PY_INTERPRETER=python
        goto py_install_deps
    )
)

python3 -V > NUL
if %ERRORLEVEL%==0 (
    echo [INFO] Found python installation. Setting up virtual environment.
    py -m venv .\venv
    if %ERRORLEVEL%==0 (
        SET PY_INTERPRETER=.\venv\Scripts\python.exe
        goto py_install_deps
    ) else (
        echo [WARN] Failed to set up virtual environment. Using standard Python installation.
        SET PY_INTERPRETER=python3
        goto py_install_deps
    )
)

goto py_not_found

REM -- INSTALL PYTHON DEPENDENCIES --
:py_install_deps
echo [INFO] Installing dependencies...
if exist .\requirements.txt (
    echo [INFO] Installing required python packages %PY_INTERPRETER%...
    "%PY_INTERPRETER%" -m pip install -r .\requirements.txt
    if %ERRORLEVEL%==0 (
        echo [INFO] Installed packages successfully.
    ) else (
        goto py_package_fail
    )
) else (
    echo [WARN] Found no requirements.txt, skipping installation. This might cause errors later on, if dependencies were not installed manually.
)
goto ot_install_deps

REM -- INSTALL DATASETS --
:ot_install_deps
echo [INFO] Installing datasets...
if not exist ".\datasets" mkdir ".\datasets"

REM BG/L Dataset
curl --output ".\datasets\BGL.zip" "https://zenodo.org/records/8196385/files/BGL.zip?download=1" --fail
if %ERRORLEVEL%==0 (
    REM unzip using powershell
    powershell -command "Expand-Archive -Path '.\datasets\BGL.zip' -DestinationPath '.\datasets\BGL'"
    if %ERRORLEVEL%==0 (
        echo [INFO] Installed "Blue Gene/L supercomputer log" Dataset
    ) else (
        echo [WARN] Failed to unzip ".\datasets\BGL.zip". Please extract to ".\datasets\BGL" manually.
    )
) else (
    echo [WARN] Failed to install "Blue Gene/L supercomputer log". Please install and unzip the dataset manually from the following repository: https://github.com/logpai/loghub/tree/master
)

REM 
curl --output ".\datasets\HDFS_v1.zip" "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1" --fail
if %ERRORLEVEL%==0 (
    REM unzip using powershell
    powershell -command "Expand-Archive -Path '.\datasets\HDFS_v1.zip' -DestinationPath '.\datasets\HDFS_v1'"
    if %ERRORLEVEL%==0 (
        echo [INFO] Installed "Hadoop distributed file system log" Dataset
    ) else (
        echo [WARN] Failed to unzip ".\datasets\HDFS_v1.zip". Please extract to ".\datasets\HDFS_v1" manually.
    )
) else (
    echo [WARN] Failed to install "Hadoop distributed file system log". Please install and unzip the dataset manually from the following repository: https://github.com/logpai/loghub/tree/master
)

goto quit

:py_not_found
echo [ERROR] No Python Interpreter was found. Quitting.
goto quit

:py_package_fail
echo [ERROR] Failed to install python packages using PIP. Please check your internet connection and try again.
goto quit

:quit
if defined VIRTUAL_ENV (
    call "%VIRTUAL_ENV%\Scripts\deactivate"
)

endlocal