# Raspberry Pi 5 Test Hazırlık Rehberi

Bu belge, `vulkan-nn-lib` kütüphanesinin **gerçek hedef donanımda — Raspberry Pi 5
(VideoCore VII GPU, Mesa V3DV Vulkan sürücüsü)** çalıştırılması ve doğrulanması
için gereken tüm adımları içerir.

> **Neden önemli?** Projenin tezi "platform bağımsız GPGPU kütüphanesi". Hafta 9'daki
> platform bağımsızlık testi yalnızca masaüstü ortamda (Intel iGPU + `llvmpipe` yazılım
> renderer) yapılmıştı. Asıl hedef platform olan RPi5 üzerinde test **2026-06-19'da
> tamamlandı ve başarılı oldu** — sonuçlar [§6](#6-sonuçlar-2026-06-19--rpi5-gerçek-donanım-koşumu)
> bölümünde. Bu test projenin ana iddiasını gerçek ARM/VideoCore donanımında kanıtlar.

---

## 1. Donanım ve İşletim Sistemi Ön Koşulları

| Gereksinim | Değer | Not |
|------------|-------|-----|
| Kart | Raspberry Pi 5 | VideoCore VII GPU |
| İşletim Sistemi | **Raspberry Pi OS (64-bit / arm64) Bookworm** | ⚠️ V3DV **yalnızca 64-bit** OS'te çalışır, 32-bit'te Vulkan yoktur |
| Mesa sürümü | ≥ 23.1 (Bookworm ile gelir) | V3DV compute shader desteği için |
| Güç | Resmî 27W USB-C adaptör | GPU yük altında düşük voltajda throttle olmasın |
| Soğutma | Aktif soğutucu (fan) önerilir | Benchmark sırasında thermal throttle'ı önler |

OS'i kontrol et:

```bash
uname -m              # aarch64 görmeli (arm64), armv7l ise yanlış OS!
cat /etc/os-release   # bookworm olmalı
```

---

## 2. Kurulması Gereken Paketler

Raspberry Pi üzerinde sırayla:

```bash
# 1) Sistem güncel olsun
sudo apt update && sudo apt full-upgrade -y

# 2) Build araçları
sudo apt install -y build-essential cmake git

# 3) Vulkan loader + headers + V3DV sürücüsü + araçlar
sudo apt install -y libvulkan-dev vulkan-tools mesa-vulkan-drivers

# 4) Shader derleyici (glslc veya glslangValidator)
sudo apt install -y glslang-tools
# İsteğe bağlı (glslc'yi de sağlar):
# sudo apt install -y shaderc
```

**Paketlerin görevi:**
- `mesa-vulkan-drivers` → V3DV ICD'sini sağlar (`libvulkan_broadcom.so`, VideoCore VII Vulkan sürücüsü)
- `libvulkan-dev` → Vulkan loader + başlık dosyaları (CMake `find_package(Vulkan)` için)
- `vulkan-tools` → `vulkaninfo`, `vkcube` (doğrulama için)
- `glslang-tools` → `glslangValidator` (CMake bunu otomatik bulur; `glslc` yoksa fallback yapar)
- `cmake`, `git`, `build-essential` → derleme + EnTT'nin `FetchContent` ile çekilmesi

> **Not (firmware):** README'de eskiden geçen `firmware-misc-nonfree` / `rpi-update`
> adımları V3DV için **gerekli değildir**. VideoCore VII Vulkan sürücüsü Mesa kullanıcı
> alanında (`mesa-vulkan-drivers`) ve V3D çekirdek DRM sürücüsü RPi çekirdeğinde
> hâlihazırda gelir. Bookworm güncel ise ek firmware kurulumu gerekmez.

---

## 3. Vulkan Kurulumunu Doğrula (derlemeden ÖNCE)

```bash
# GPU çekirdek sürücüsü yüklü mü?
ls /dev/dri            # card0/card1 ve renderD128 görmeli
lsmod | grep v3d       # v3d / gpu_sched modülleri

# V3DV ICD dosyası var mı?
ls /usr/share/vulkan/icd.d/   # broadcom_icd.aarch64.json görmeli

# En önemlisi: Vulkan cihazı görünüyor mu?
vulkaninfo --summary
```

`vulkaninfo --summary` çıktısında şunları ara ve **not al** (rapora gidecek):
- `deviceName` → büyük olasılıkla **"V3D 7.1"** veya benzeri
- `deviceType` → `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`
- `apiVersion` → 1.2.x veya 1.3.x
- `driverName` → `V3DV (Mesa)`

Compute limitlerini doğrula (shader uyumluluğu için kritik):

```bash
vulkaninfo | grep -E "maxComputeWorkGroupInvocations|maxComputeSharedMemorySize|maxComputeWorkGroupSize"
```

Beklenen / gereken değerler:

| Limit | Bizim ihtiyacımız | Beklenen V3DV değeri | Durum |
|-------|-------------------|----------------------|-------|
| `maxComputeWorkGroupInvocations` | 256 (matmul 16×16) | 256 | ⚠️ tam sınırda — doğrula |
| `maxComputeSharedMemorySize` | 2048 byte (2× 16×16 float) | 16384 | ✅ rahat |
| `maxComputeWorkGroupSize[0]` | 64 (fc_layer) | ≥ 256 | ✅ |

> ⚠️ Eğer `maxComputeWorkGroupInvocations` 256'dan küçük çıkarsa `shaders/matmul.comp`
> içindeki `TILE` değerini 16 → 8'e düşür (64 invocation/workgroup) ve yeniden derle.
> 256 ise dokunmaya gerek yok.

Hızlı görsel doğrulama (opsiyonel, masaüstü bağlıysa):
```bash
vkcube   # dönen küp görünmeli — GPU pipeline çalışıyor demektir
```

---

## 4. Projeyi Derle

```bash
git clone <repo-url> vulkan-nn-lib
cd vulkan-nn-lib

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

> ⚠️ **`-DCMAKE_BUILD_TYPE=Release` şart.** Benchmark sayılarının anlamlı olması için
> CPU baseline'ın optimize derlenmesi gerekir; Debug'da CPU yapay olarak yavaş çıkar.

> ⚠️ **İnternet gerekir:** İlk `cmake -B build` çağrısı EnTT'yi `FetchContent` ile
> GitHub'dan indirir. Pi internete bağlı olmalı (veya `build/_deps` önceden doldurulmalı).
> RPi5'te ilk derleme birkaç dakika sürebilir.

---

## 5. Testleri ve Benchmark'ları Çalıştır

```bash
# Doğruluk testleri — hepsi GPU'da (V3DV) çalışıp geçmeli
cd build && ctest --output-on-failure
cd ..

# Tek cihaz benchmark
./build/bench_cpu_vs_gpu     # GEMM CPU vs GPU (128/256/512)
./build/bench_inference      # 784→128→10 MLP, 1000 tekrar

# Platform tablosu — RPi'de V3D cihazını otomatik enumerate eder
./build/bench_platform
```

`ctest` çıktısında **4/4 test geçmeli** (test_relu, test_matmul, test_buffer, test_mlp).
Bir test V3DV'de fail olursa → bu değerli bir platform bağımsızlık bulgusudur, rapora yaz.

---

## 6. Sonuçlar (2026-06-19 — RPi5 gerçek donanım koşumu)

> ✅ **Durum: BAŞARILI.** Kütüphane RPi5 / VideoCore VII (V3D) üzerinde derlendi,
> 4/4 doğruluk testi geçti, üç benchmark da hatasız sayı üretti. Platform
> bağımsızlık iddiası gerçek ARM/VideoCore donanımında doğrulandı.

**Çalışma ortamı (rehberdeki varsayımdan farkı not edildi):**

| Alan | Değer |
|------|-------|
| deviceName | **V3D 7.1.10.2** |
| deviceType | Integrated GPU |
| apiVersion | 1.3.318 |
| Mesa / driverVersion | **25.2.8** |
| vendorID | 0x14e4 (Broadcom) |
| OS / kernel | **Ubuntu 24.04.4 LTS**, `6.8.0-1057-raspi`, aarch64 |
| Sıcaklık (koşum boyunca) | 52.7 → 58.2 °C — throttle **yok** |

> ℹ️ Rehber başta Raspberry Pi OS Bookworm (Mesa ≥ 23.1) varsayıyordu; gerçek test
> **Ubuntu 24.04 (Mesa 25.2.8)** üzerinde yapıldı. V3DV her iki dağıtımda da
> `mesa-vulkan-drivers` ile gelir; sonuç değişmedi. (Ubuntu'da kurulum sırasında
> `noble-updates` deposunun eksikliğinden kaynaklı `bzip2`/`libbz2` bağımlılık
> çakışması yaşandı — depo geri eklenince çözüldü.)

**Doğruluk testleri:** `test_buffer`, `test_relu`, `test_matmul` (3 vaka),
`test_mlp` (freeze/unfreeze dahil 4 vaka) → **hepsi PASSED** (GPU/V3DV dispatch ile).
Bu, `maxComputeWorkGroupInvocations` ≥ 256 olduğunu da kanıtlar (matmul TILE=16 sorunsuz çalıştı).

**GEMM (`bench_cpu_vs_gpu`):**

| Boyut | CPU ort (ms) | GPU ort (ms) | Hızlanma |
|-------|-------------|-------------|---------|
| 128×128 | 12.63 | 1.63 | 7.7× |
| 256×256 | 85.64 | 17.50 | 4.9× |
| 512×512 | 838.81 | 125.03 | 6.7× |

> Bağımsız ikinci koşumda (SSH) GPU süreleri tutarlı (~1.66 / 18.1 / 106 ms) çıktı;
> CPU baseline'ı koşumlar arası dalgalanıyor (512×512'de 839–1142 ms), bu nedenle
> mutlak hızlanma 5–11× bandında değişebilir. CPU referansı tek-thread naif GEMM'dir.

**MLP inference (`bench_inference`, 784→128→10, 1000 tekrar):**

| Ölçüm | Süre |
|-------|------|
| İlk çağrı (buildCommandBuffer dahil) | 0.730 ms |
| Tekrar başına | 0.683 ms |
| Throughput | 1464 inf/sec |

**Platform karşılaştırması (`bench_platform`, V3D GPU vs llvmpipe CPU):**

| Cihaz | Tip | Ağ | İlk çağrı (ms) | Çağrı başına (ms) | inf/sec |
|-------|-----|----|---------------|------------------|---------|
| V3D 7.1.10.2 | Integrated GPU | 784→128→10 | 0.78 | 0.66 | 1511 |
| llvmpipe (LLVM 20.1.2) | CPU | 784→128→10 | 134.7 | 0.37 | 2720 |
| V3D 7.1.10.2 | Integrated GPU | 784→512→256→10 | 2.15 | 3.20 | 312 |
| llvmpipe (LLVM 20.1.2) | CPU | 784→512→256→10 | 3.82 | 1.74 | 573 |

> **Önemli bulgu:** Saf yoğun GEMM'de (yukarıdaki tablo) V3D, naif CPU'yu 5–11×
> geçiyor; ama gerçek MLP inference'ta vektörize **llvmpipe (CPU) V3D'den hızlı**.
> Sebep: küçük tek-örnek inference'ta katman başına dispatch/submit overhead'i V3D'de
> baskın. Yani kazanç iş yüküne bağlı — büyük yoğun matmul'da GPU, küçük seri
> inference'ta CPU yolu avantajlı. Batch'leme / dispatch birleştirme net bir gelecek-çalışma yönü.

**Enerji ölçümü (`vcgencmd pmic_read_adc`, dahili PMIC — board gücü = Σ rail V×I):**

| Durum | Board gücü (W) | Dinamik (boşta üstü) |
|-------|---------------|----------------------|
| Boşta (idle) | **2.10** | — |
| MLP inference yükü (784→128→10) | **2.56** | +0.46 W |
| Yoğun GEMM yükü (128/256/512) | **3.47** | +1.37 W |

| Çıkarım başına enerji | Değer |
|-----------------------|-------|
| Throughput | 1468 inf/sec |
| Enerji/çıkarım (toplam board) | **≈ 1.74 mJ** |
| Enerji/çıkarım (dinamik, boşta düşülmüş) | **≈ 0.31 mJ** |

Sıcaklık ölçüm boyunca 48.3 → 56.5 °C, **throttle yok**. Tüm kart yük altında bile
**< 3.5 W** çekti — bu, "düşük güçlü / enerji verimli platform" iddiasını ölçümle destekler.

> **Yöntem notu (dürüstlük):** Güç, yük altında 20 sn boyunca PMIC'ten ~0.2 sn aralıkla
> örneklenip ortalandı; boşta değer aynı şekilde 10 sn ölçüldü. Inference yük değeri
> **muhafazakârdır (gerçeğin altında)**: sürdürülebilir yük için benchmark süreç döngüsünde
> tekrar tekrar başlatıldığından duvar-saati süresinin önemli kısmı Vulkan init/teardown'a
> gidiyor ve aktif compute gücünü seyreltiyor. Sürekli GPU compute'un baskın olduğu GEMM
> döngüsü (+1.37 W) gerçek yük çekişini daha iyi temsil eder. Sayılar tek koşumdur;
> mutlak watt ölçüm yöntemine (PMIC vs. inline USB-C güç ölçer) göre ±%10 oynayabilir.

---

## 7. Olası Sorunlar ve Çözümler

| Belirti | Olası neden | Çözüm |
|---------|-------------|-------|
| `vulkaninfo` "No Vulkan ICDs" / cihaz yok | 32-bit OS veya `mesa-vulkan-drivers` eksik | 64-bit Bookworm'a geç; paketi kur |
| `find_package(Vulkan)` bulamıyor | `libvulkan-dev` eksik | `sudo apt install libvulkan-dev` |
| Shader derlenmiyor (`.spv` yok) | `glslc`/`glslangValidator` yok | `sudo apt install glslang-tools` |
| Test fail: workgroup size hatası | `maxComputeWorkGroupInvocations` < 256 | `matmul.comp` `TILE` 16→8, yeniden derle |
| `VK_ERROR_DEVICE_LOST` / takılma | Termal throttle veya zayıf güç | Aktif soğutma + resmî 27W adaptör |
| EnTT indirilemiyor | İnternet yok | Pi'yi bağla veya `_deps` önceden taşı |
| CPU benchmark anlamsız hızlı/yavaş | Debug build | `-DCMAKE_BUILD_TYPE=Release` ile yeniden derle |

---

## 8. Beklentiler

- VideoCore VII masaüstü/entegre GPU'lara göre düşük performanslıdır; mutlak süreler
  masaüstünden yavaş olacaktır — **amaç hız rekoru değil, aynı kodun farklı Vulkan
  backend'inde değişiklik yapmadan çalıştığını kanıtlamaktır.**
- Küçük ağda (784→128→10) upload/submit/wait overhead'i baskın olabilir; bu masaüstünde
  de gözlendi (Hafta 9 notu). Büyük ağda GPU avantajı belirginleşir.
- Hedef: `ctest` 4/4 geçer + benchmark'lar hata vermeden sayı üretir → platform
  bağımsızlık iddiası gerçek ARM/VideoCore donanımında doğrulanmış olur.
