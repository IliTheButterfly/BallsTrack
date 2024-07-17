# Installation steps for use with cmake cuda and opencv

These steps have been tested on a machine with these specs:


| Category | Value                     |
| ---      | ---                       |
| OS Name  | Microsoft Windows 10 Home |
| Version  |	10.0.19045 Build 19045 |
| RAM      | 64 GB                     |
| CPU      | Intel Core i7-9700K       |
| GPU      | RTX 4090                  |

## Install a compiler

Go to https://jmeubank.github.io/tdm-gcc/ and install the latest gcc release. At the moment of writing this, TDM-GCC 10.3.0 was used.

## Install CUDA

Go to https://developer.nvidia.com/cuda-downloads to install CUDA. You will need to make an account before getting the download link. CUDA 11.0 was used for this step.

## Install OpenCV

Go to https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html#decreasing-the-build-time-with-ninja and follow the steps for building with cmake.

## Setup vscode

Go to https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-vscode?pivots=shell-cmd and follow the instructions.

## Setup environment variables

After installing the libraries, you need to ensure your environment variables are setup correctly.

Here is a list of the environment variables that were needed:
- CMAKE_MAKE_PROGRAM: `path\to\ninja.exe`
- OpenCV_DIR: `path\to\opencv\build\x64\vc16\`
- VCPKG_ROOT: `path\to\vcpkg\`
- Add to PATH the following: `;path\to\GCC\bin\;path\to\ninja\;path\to\vcpkg\;%OpenCV_DIR%\bin`
