# RF-DETR TensorRT-EP Inference (Windows / Visual Studio)

A Windows (Visual Studio) C++ implementation for **RF-DETR** inference accelerated by **TensorRT-EP** (via ONNX / ONNX Runtime TensorRT EP).  
Tested on **RTX 4070 Ti Super** with `rf-detr-base`, improving inference latency from **9–13 ms** to **2–5 ms**.

> ✅ Visual Studio project solution.  
> ✅ C++17 + CUDA + cuDNN + OpenCV + ONNX Runtime GPU + TensorRT.

---

## 1. Features

- RF-DETR ONNX inference on Windows (C++17)
- Acceleration using **TensorRT 10.x**
- ONNX Runtime + TensorRT EP
- OpenCV-based image preprocessing and visualization
- Performance tested on RTX 4070 Ti Super

---

## 2. Environment

**OS**
- Windows 10/11 (x64)

**Compiler**
- Visual Studio 2022 (MSVC) + C++17

**GPU Stack**
- CUDA **12.6**
- cuDNN **9.8.0**

**Dependencies**
- OpenCV **4.10.0**
- onnxruntime-gpu **1.22.0**
- TensorRT **10.9.0.34**

---

## 3. Performance

Device: **RTX 4070 Ti Super**  
Model: **rf-detr-base**

| Backend | Latency (ms) |
|---|---:|
| ONNX Runtime (CUDA EP) | 9–13 |
| TensorRT (TensorRT EP) | **2–5** |

> Notes:
> - Timing depends on input size, batch size, warmup, and post-processing.

---

## 5. Build (Visual Studio, No CMake)

### 5.1 Create Visual Studio Project

1. Create a new **Console App (C++)**
2. Set **C++ Language Standard** = **ISO C++17**
   - Project Properties → C/C++ → Language → C++ Language Standard → `/std:c++17`

### 5.2 Include Directories

Project Properties → C/C++ → General → Additional Include Directories:

- OpenCV include
- ONNX Runtime include
- TensorRT include
- CUDA include
- cuDNN include


### 5.3 Library Directories

Project Properties → Linker → General → Additional Library Directories:

- OpenCV lib
- ONNX Runtime lib
- TensorRT lib
- CUDA lib
- cuDNN lib

### 5.4 Linker Inputs (Additional Dependencies)

Project Properties → Linker → Input → Additional Dependencies:

**OpenCV**
- `opencv_world4100.lib`

**ONNX Runtime**
- `onnxruntime.lib`
- `onnxruntime_providers_cuda.lib`
- `onnxruntime_providers_shared.lib`
- `onnxruntime_providers_tensorrt.lib`

**Windows system libs**
- `kernel32.lib`
- `user32.lib`
- `gdi32.lib`
- `winspool.lib`
- `comdlg32.lib`
- `advapi32.lib`
- `shell32.lib`
- `ole32.lib`
- `oleaut32.lib`
- `uuid.lib`
- `odbc32.lib`
- `odbccp32.lib`

**TensorRT 10**
- `nvinfer_10.lib`
- `nvinfer_plugin_10.lib`
- `nvonnxparser_10.lib`
- `nvinfer_dispatch_10.lib`
- `nvinfer_lean_10.lib`
- `nvinfer_vc_plugin_10.lib`

**CUDA/cuDNN**
- `cudnn.lib`
- `cublas.lib`
- `cudart.lib`

---

## 6. Runtime Setup (DLLs / PATH)

To run successfully, ensure DLLs can be found (choose one method):

### Option A: Add to PATH
Add these directories into Windows `PATH`:

- OpenCV `bin`
- ONNX Runtime `bin` (or the folder containing `onnxruntime.dll`)
- TensorRT `lib`
- CUDA `bin`
- cuDNN `bin`

### Option B: Copy DLLs next to exe (recommended)
Copy required `.dll` files into:

> If you see errors like missing `cudnn*.dll`, `zlibwapi.dll`, `nvinfer*.dll`,
> it means DLL search path is not set correctly.

## 7. ONNX Runtime TensorRT EP Notes

This repo uses ONNX Runtime GPU with TensorRT Execution Provider.
The typical provider priority is:

- TensorRT EP
- CUDA EP
- CPU EP

> If TensorRT EP fails to load, it will fallback to CUDA or CPU.

## 8. Troubleshooting
8.1 TensorRT INT64 warning

If you see warnings about INT64 weights, TensorRT will cast to INT32.
You can re-export ONNX with proper dtypes or run simplification.

8.2 cuDNN version mismatch

Example:

TensorRT was linked against cuDNN X but loaded cuDNN Y

Fix:

Ensure the cuDNN DLL in PATH matches the version TensorRT expects.

8.3 OpenCV parallel plugin missing

If you see missing opencv_core_parallel_*.dll messages:

Ensure OpenCV bin is in PATH

Or use opencv_world static/monolithic build configuration consistently

9. Acknowledgements

[RF-DETR](https://rfdetr.roboflow.com/)

[ONNX Runtime](https://onnxruntime.ai/docs/get-started)

[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

[OpenCV](https://opencv.org/)
