@echo off
REM ================================
REM Drone Audio Detection - Windows Setup
REM ================================

echo === Setting up Drone Audio Detection environment ===

REM 1️⃣ Create virtual environment
python -m venv venv
echo Virtual environment created at .\venv

REM 2️⃣ Activate virtual environment
call venv\Scripts\activate
echo Virtual environment activated

REM 3️⃣ Upgrade pip
python -m pip install --upgrade pip

REM 4️⃣ Install core dependencies
pip install numpy>=1.24 scipy>=1.11 matplotlib>=3.7 seaborn>=0.12 tqdm>=4.66

REM 5️⃣ Install audio processing
pip install librosa>=0.10 soundfile>=0.12

REM 6️⃣ Install PyTorch CPU version (change for GPU if needed)
pip install torch>=2.2 torchaudio>=2.2 torchvision>=0.18 --index-url https://download.pytorch.org/whl/cpu

REM 7️⃣ Install TensorBoard and Jupyter
pip install tensorboard>=2.16 ipython>=8.15 ipywidgets>=8.2

REM 8️⃣ Optional utilities
pip install requests>=2.32

echo === All dependencies installed successfully! ===
echo Activate environment with: call venv\Scripts\activate
pause
