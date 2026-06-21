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
> VideoCore VII GPU, Mesa V3DV sürücüsüyle Vulkan compute destekler (Mesa 23.1+).
> Sürücü **64-bit (arm64) Raspberry Pi OS Bookworm** ile `mesa-vulkan-drivers`
> paketinden gelir; ayrı firmware kurulumu gerekmez. Detaylı kurulum ve test
> adımları için bkz. [RASPBERRY_PI_TEST.md](RASPBERRY_PI_TEST.md).

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

### Benchmark'lar

> Benchmark'lar için **Release** derleme şarttır (CMake artık tip belirtilmezse
> otomatik Release seçer). Adil CPU kıyası için `libopenblas-dev` önerilir.

```bash
./build/bench_cpu_vs_gpu    # Adil GEMM kıyası: naif / çok-thread / OpenBLAS / GPU
./build/bench_soa_vs_aos    # DOP etkisi: SoA vs AoS bellek düzeni (Vulkan gerektirmez)
./build/bench_inference     # MLP inference hızı (pre-record optimizasyonu)
./build/bench_platform      # Tüm Vulkan cihazlarında karşılaştırma
```

### MNIST uçtan uca doğruluk (gerçek model)

Kütüphanenin yalnızca hızlı değil **doğru** da çalıştığını gösterir:

```bash
python3 tools/train_mnist.py   # MNIST indirir, 784→128→10 MLP eğitir, models/*.bin üretir
./build/mnist_infer            # aynı ağırlıkları Vulkan'da koşar, test doğruluğu raporlar
```

## Proje yapısı

```
vulkan-nn-lib/
├── CMakeLists.txt          # Ana build tanımı
├── shaders/
│   ├── relu.comp           # ReLU aktivasyon shader'ı
│   ├── matmul.comp         # Tiled GEMM shader'ı
│   ├── fc_layer.comp       # Tam bağlı katman (GEMV + bias + ReLU)
│   └── CMakeLists.txt      # glslc derleme kuralları
├── src/
│   ├── vk_context.{h,cpp}  # VkInstance/Device/Queue yönetimi
│   ├── vk_buffer.{h,cpp}   # GPU bellek yönetimi
│   ├── inference.{h,cpp}   # VulkanMLP sınıfı (EnTT/ECS tabanlı)
│   └── main.cpp            # Smoke test
├── tests/                  # test_relu, test_matmul, test_buffer, test_mlp
├── benchmark/
│   ├── bench_cpu_vs_gpu.cpp # Adil baseline: naif/çok-thread/OpenBLAS/GPU
│   ├── bench_soa_vs_aos.cpp # DOP (veri temelli) etkisi ölçümü
│   ├── bench_inference.cpp
│   └── bench_platform.cpp
├── examples/
│   └── mnist_infer.cpp     # MNIST uçtan uca doğruluk
├── tools/
│   └── train_mnist.py      # MNIST eğitimi + ağırlık dışa aktarma (numpy)
└── models/                 # Ağırlık/veri dosyaları (.bin, git-ignored)
```

## Bulgular ve Sınırlılıklar

Bu bölüm projenin iddialarını ölçümle dürüstçe konumlandırır.

### 1. Veri Temelli Programlama (DOP) etkisi — ölçüldü

`bench_soa_vs_aos`, aynı MLP forward hesabını **aynı ağırlıklarla** iki bellek
düzeniyle koşar: SoA (bitişik diziler — kütüphanenin GPU buffer ve EnTT bileşen
havuzu düzeni) vs AoS (dağınık nesneler — klasik OOP). Tek değişken bellek
yerleşimidir. Örnek ölçüm (12 çekirdek, Release):

| Katman (inSize × nöron) | AoS (ms) | SoA (ms) | Hızlanma |
|-------------------------|---------:|---------:|---------:|
| 784 × 5000 (geniş)      | 2.21 | 1.92 | 1.15× |
| 256 × 16000             | 2.55 | 1.68 | 1.52× |
| 64 × 64000              | 3.67 | 1.24 | 2.96× |
| 16 × 256000             | 6.81 | 1.19 | 5.71× |
| 4 × 512000 (uç nokta)   | 8.06 | 0.72 | 11.27× |

**Bulgu:** DOP'un kazancı nöron başına iş miktarına bağlıdır. Geniş katmanlarda
fark küçüktür (ağırlık satırı her iki düzende de bitişik okunur); ÇOK SAYIDA
KÜÇÜK bileşende (ECS'nin tipik senaryosu) cache-miss sayısı arttıkça SoA 11×'e
kadar hızlanır. Bu, "veri temelli bellek düzeni verimliliği artırır" iddiasını
sayısallaştırır.

### 2. Sinir ağı kapsamı — MLP çıkarımı, uçtan uca doğrulandı

Kapsam dürüstçe **MLP çıkarımı (inference)** ile sınırlıdır (tam bağlı katman +
ReLU). `mnist_infer`, numpy ile eğitilen 784→128→10 modelin ağırlıklarını
Vulkan'da koşar:

- **Test doğruluğu: %97.41** (10000 örnek) — eğitilen modelle birebir.
- **GPU vs CPU referans:** max |fark| = 2.9e-06 → sayısal olarak doğru.

Böylece kütüphane yalnızca hızlı değil, gerçek bir görevde **doğru** da çalışıyor.
**Gelecek çalışma:** CNN/evrişim, diğer aktivasyonlar, batch inference.

### 3. Adil kıyas tabanı — çok-çekirdek + OpenBLAS eklendi

`bench_cpu_vs_gpu` artık GPU'yu yalnızca optimize edilmemiş tek-thread CPU ile
değil; çok-thread ve **OpenBLAS** ile de kıyaslar. Örnek (Intel iGPU, 12 çekirdek):

| Boyut | Naif (ms) | Çok-thread (ms) | OpenBLAS (ms) | GPU (ms) |
|------:|----------:|----------------:|--------------:|---------:|
| 128 | 1.72 | 0.84 | 0.15 | 0.73 |
| 256 | 13.08 | 5.52 | 0.38 | 1.83 |
| 512 | 96.91 | 29.89 | 1.48 | 11.94 |

**Dürüst bulgu:** Bu iş yükü ve donanımda optimize CPU (özellikle OpenBLAS)
GPU'yu geçiyor. Eski "GPU 11× hızlı" sonucu, yalnızca **naif** CPU'ya görelidir
ve yanıltıcıdır. GPU'nun değeri taşınabilirlik ve daha büyük/batch iş yüklerinde
ortaya çıkar. **Gelecek çalışma:** Kompute / ncnn-Vulkan gibi taşınabilir bir
Vulkan ML kütüphanesiyle kıyas, batch dispatch birleştirme.

## Haftalık ilerleme

Bkz. [HAFTALIK_ILERLEME.md](HAFTALIK_ILERLEME.md)
