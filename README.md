# vulkan-nn-lib

Platform-bağımsız GPGPU kütüphanesi — Vulkan compute shader'ları ve ECS mimarisi ile yapay sinir ağı inference.

**Öğrenciler:** Ahmet Korkmaz
**Danışman:** Prof. Dr. Turgay Tugay Bilgin
**Dönem:** 2025-2026 Bahar

## Bağımlılıklar

| Bağımlılık | Sürüm | Notlar |
|------------|-------|--------|
| CMake | ≥ 3.20 | Build sistemi |
| Vulkan | ≥ 1.2 | Loader + headers |
| glslc | herhangi | SPIR-V shader derleyici |

### Ubuntu / Debian / Raspberry Pi OS kurulumu

```bash
sudo apt update
sudo apt install cmake libvulkan-dev vulkan-tools mesa-vulkan-drivers glslang-tools
```

> **Raspberry Pi 5 notu:**
> VideoCore VII, Vulkan 1.3 destekler. Mesa 23.1+ ile tam compute shader desteği
> gelir. Kernel 6.1+ ve `firmware-misc-nonfree` paketi gereklidir.
> ```bash
> sudo apt install firmware-misc-nonfree
> sudo rpi-update   # isteğe bağlı, güncel firmware için
> ```

### LunarG Vulkan SDK (alternatif)

Tam SDK (validation layer'lar dahil) için:
```bash
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
sudo apt update && sudo apt install vulkan-sdk
```

## Derleme

```bash
cmake -B build
cmake --build build -j$(nproc)
```

### Test çalıştırma

```bash
cd build && ctest --output-on-failure
```

### Benchmark

```bash
./build/bench_cpu_vs_gpu
```

## Proje yapısı

```
vulkan-nn-lib/
├── CMakeLists.txt          # Ana build tanımı
├── shaders/
│   ├── relu.comp           # ReLU aktivasyon shader'ı
│   ├── matmul.comp         # Tiled GEMM shader'ı
│   └── CMakeLists.txt      # glslc derleme kuralları
├── src/
│   ├── vk_context.{h,cpp}  # VkInstance/Device/Queue yönetimi
│   ├── vk_buffer.{h,cpp}   # GPU bellek yönetimi
│   ├── inference.{h,cpp}   # VulkanMLP sınıfı
│   └── main.cpp            # Smoke test
├── tests/
│   ├── test_relu.cpp
│   └── test_matmul.cpp
├── benchmark/
│   └── bench_cpu_vs_gpu.cpp
└── models/                 # Ağırlık dosyaları (.bin, git-ignored)
```

## Haftalık ilerleme

Bkz. [HAFTALIK_ILERLEME.md](HAFTALIK_ILERLEME.md)
