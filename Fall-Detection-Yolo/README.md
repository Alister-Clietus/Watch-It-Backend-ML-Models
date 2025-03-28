# YOLO Fall Detection Project - Setup Guide

This guide will help you set up and run the YOLO-based fall detection project.

## 1️⃣ Open the Project in Visual Studio Code
1. Open **Visual Studio Code**.
2. Navigate to the **fall detection project folder**.
3. Open the terminal by pressing **Ctrl + ` (backtick)**.

---

## 2️⃣ Create and Activate a Virtual Environment
To keep dependencies isolated, create a **virtual environment**:

### **Create a Virtual Environment**
```sh
python -m venv venv
```

### **Activate the Virtual Environment**
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **Linux/Mac:**
  ```sh
  source venv/bin/activate
  ```

You should see `(venv)` appear in your terminal, indicating that the virtual environment is active.

---

## 3️⃣ Install Required Dependencies
Once the virtual environment is activated, install all required dependencies:
```sh
pip install -r requirements.txt
```


# PyTorch Installation Guide (CUDA 11.7)

This guide will help you install PyTorch with **CUDA 11.7** and verify that it is correctly configured on your system.

## 1️⃣ Check Your NVIDIA GPU and CUDA Version
Before installing PyTorch, ensure that your system has an **NVIDIA GPU** and a compatible **CUDA version** installed.

### Check GPU and CUDA Version
Run the following command in the terminal:
```sh
nvidia-smi
```
This should output details about your GPU, including the **CUDA version**. Example output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 516.40       Driver Version: 516.40       CUDA Version: 11.7     |
+-----------------------------------------------------------------------------+
```
In this example, **CUDA 11.7** is installed.

---

## 2️⃣ Install PyTorch with CUDA 11.7
To install PyTorch with CUDA 11.7, run:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
This ensures PyTorch is installed with the correct CUDA version.

---

## 3️⃣ Verify PyTorch Installation
To confirm that PyTorch is correctly installed and using the GPU, run the following script.

### **Create a verification script**
Save the following code as **gpu.py**:
```python
import torch

print("PyTorch CUDA available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
```

### **Run the verification script**
Execute the script using:
```sh
python gpu.py
```

### **Expected Output (if CUDA is working)**
```
PyTorch CUDA available: True
CUDA Version: 11.7
GPU Name: NVIDIA GeForce ...
```
If **CUDA is not detected**, ensure you have installed the correct drivers and CUDA version.

---

## 4️⃣ Troubleshooting
### **1. PyTorch does not detect CUDA**
- Ensure NVIDIA drivers are installed:
  ```sh
  nvidia-smi
  ```
- Reinstall CUDA 11.7 from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).
- Reinstall PyTorch with the correct CUDA version.

### **2. Check CUDA Installation**
To verify CUDA is installed, run:
```sh
nvcc --version
```
This should return the installed CUDA version.

---

## ✅ PyTorch with CUDA 11.7 is Now Installed!
You are now ready to use PyTorch with GPU acceleration. 🚀


## ✅ The setup is complete! 🚀
You can now run the project and start detecting falls using YOLO.

If you encounter issues, ensure:
- Your Python version is **3.7 or later**.
- CUDA is properly installed (for GPU acceleration).
- The virtual environment is activated before running any command.

Let me know if you need any modifications! 🎯

