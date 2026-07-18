\## llama.cpp Build Setup (Sakshi)



\- Cloned separately at D:\\Projects\\llama.cpp (outside repo, avoid bloating git)

\- CUDA backend: build-cuda/ — built with CUDA 12.8, MSVC v143 toolset, arch 89 (RTX 4050 Ada)

\- Vulkan backend: build-vulkan/ — built with Vulkan SDK 1.4.350.0, runs on both RTX 4050 (discrete) and Radeon 740M (iGPU)

\- Required: VS2026 Build Tools + v143 toolset, CUDA Toolkit 12.8, Vulkan SDK

\- Note: had to manually copy CUDA MSBuildExtensions to VC/v170 and v180 BuildCustomizations folders

\- Note: disabled Windows Smart App Control (was blocking nvcc's fatbinary.exe)

