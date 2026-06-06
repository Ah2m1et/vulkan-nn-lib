# Bitirme Projesi Haftalık İlerleme Raporu

## Proje Bilgileri

| Alan | Bilgi |
|------|-------|
| **Öğrenci Adı Soyadı** | Ahmet Korkmaz |
| **Öğrenci No** | 21360859072 |
| **Proje Başlığı** | Yapay Sinir Ağları için Vulkan API ve Veri Temelli Programlama Tabanlı Platform Bağımsız GPGPU Kütüphanesi Geliştirilmesi |
| **Danışman** | Prof. Dr. Turgay Tugay Bilgin |
| **Dönem** | 2025-2026 Bahar |

---

## İş Planı

| Hafta | Tarih Aralığı | Planlanan İş | Tahmini Tamamlanma (%) | Durum |
|-------|---------------|--------------|------------------------|-------|
| 1  | 07.04 - 13.04 | Proje dizini, GitHub reposu, geliştirme ortamı kurulumu, CMake yapılandırması | %10 | ✅ Tamamlandı |
| 2  | 14.04 - 20.04 | Vulkan context kurulumu (VkInstance, VkDevice, compute queue), GPU buffer yönetimi | %20 | ✅ Tamamlandı |
| 3  | 28.04 - 03.05 | İlk compute shader (ReLU), SPIR-V derleme pipeline'ı, shader test altyapısı | %30 | ✅ Tamamlandı |
| 4  | 05.05 - 11.05 | Tiled matrix multiplication (GEMM) shader, workgroup optimizasyonu | %42 | ✅ Tamamlandı |
| 5  | 12.05 - 18.05 | CPU baseline implementasyonu, ilk CPU vs GPU benchmark ölçümleri | %54 | ✅ Tamamlandı |
| 6  | 19.05 - 25.05 | ECS mimarisiyle katman yönetimi, VulkanMLP sınıfı iskelet kodu | %64 | ✅ Tamamlandı |
| 7  | 26.05 - 01.06 | MNIST ağırlık yükleme, 784→128→10 MLP inference pipeline | %75 | ✅ Tamamlandı |
| 8  | 02.06 - 08.06 | Command buffer önceden kaydetme optimizasyonu, inference hız testleri | %84 | ✅ Tamamlandı |
| 9  | 09.06 - 15.06 | Platform bağımsızlık testi (farklı Vulkan cihazda çalıştırma), benchmark tablosu | %92 | ✅ Tamamlandı |
| 10 | 16.06 - 22.06 | Kod temizliği, dokümantasyon, final rapor, GitHub release | %100 | ⬜ Başlamadı |

**Durum simgeleri:** ⬜ Başlamadı | 🔄 Devam Ediyor | ✅ Tamamlandı | ⚠️ Gecikti

---

## Haftalık İlerleme Kayıtları

---

### Hafta 5 *(Tarih: 12.05.2025 - 24.05.2025)*

**Plandaki hedef:**
- CPU baseline GEMM implementasyonu, ilk CPU vs GPU benchmark ölçümleri (chrono + bellek transferi dahil toplam süre)

**Bu hafta yaptıklarım:**
- `benchmark/bench_cpu_vs_gpu.cpp` tamamlandı: GPU TODO bloğu gerçek dispatch ile dolduruldu
  - CPU: saf üçlü döngü (optimizasyon yok), 5 tekrar ortalaması
  - GPU: upload + tiled GEMM dispatch + download toplam duvar saati süresi, 1 warm-up + 5 ölçüm
  - 3 matris boyutu test edildi: 128, 256, 512
- `CMakeLists.txt`: `bench_cpu_vs_gpu` hedefine `SHADER_DIR` tanımı eklendi (matmul.spv yolu)
- `ctest` 3/3 test hâlâ geçti (test_relu, test_matmul, test_buffer)

**Benchmark sonuçları (Release build, `bench_cpu_vs_gpu`):**

| Boyut | CPU ort (ms) | GPU ort (ms) | Hızlanma |
|-------|-------------|-------------|---------|
| 128×128 | 3.290 | 1.023 | 3.22× |
| 256×256 | 19.661 | 2.360 | 8.33× |
| 512×512 | 206.276 | 18.483 | 11.16× |

GPU süresi upload + dispatch + download'ı kapsar; matris büyüdükçe hızlanma oranı artar (compute süresi bellek transfer maliyetini bastırıyor).

**Plana göre durumum:**
- Hafta 5 hedeflerine ulaşıldı; not: hafta 1 hafta gecikmeli tamamlandı (24.05.2025) ⚠️

**Karşılaştığım sorunlar / zorluklar:**
- Gecikmeden dolayı sorun yaşanmadı; mevcut altyapı (VkContext, GpuBuffer, dispatchPipeline) benchmark için yeterliydi

**Gelecek hafta hedefim:**
- Platform bağımsızlık testi (farklı Vulkan cihazda çalıştırma), benchmark tablosu

---

### Hafta 8 *(Tarih: 06.06.2026)*

**Plandaki hedef:**
- Command buffer önceden kaydetme (pre-record) optimizasyonu, inference hız testleri

**Bu hafta yaptıklarım:**
- `VulkanMLP::buildCommandBuffer()` (private, lazy): tüm katmanları tek bir reusable command buffer'a kaydeder
  - Her katman için persistent descriptor pool + descriptor set tahsis edildi
  - Katmanlar arası `VkBufferMemoryBarrier` (compute→compute, SHADER_WRITE→SHADER_READ)
  - `ONE_TIME_SUBMIT` yerine sıfır-flag; buffer N kez submit edilebilir
- `VulkanMLP::forward()` güncellendi: `addLayer()` sonrası `prepared_=false` → ilk `forward()` `buildCommandBuffer()` tetikler; sonraki çağrılarda yalnızca upload + single `vkQueueSubmit` + wait + download
- `loadWeights()` pre-record'u geçersiz **kılmıyor**: buffer handle değişmediği için mevcut cmd geçerli
- `benchmark/bench_inference.cpp` yazıldı: 784→128→10, 1000 tekrar

**Benchmark sonuçları (Release build, `bench_inference`):**

| Ölçüm | Süre |
|-------|------|
| İlk çağrı (buildCommandBuffer dahil) | 1.711 ms |
| 1000 tekrar toplam | 826.58 ms |
| Tekrar başına | 0.827 ms |
| Throughput | 1209 inf/sec |

Naif yaklaşımda her `forward()` çağrısı için N katman × (descriptor pool oluştur + cmd alloc + kayıt + submit + bekleme + temizlik) yapılırdı. Pre-record ile sadece tek bir submit + wait var; kayıt maliyeti sıfıra iner.

**Plana göre durumum:**
- Haftanın tüm hedeflerine ulaşıldı ✅

**Karşılaştığım sorunlar / zorluklar:**
- Yok

**Gelecek hafta hedefim:**
- Kod temizliği, dokümantasyon, final rapor, GitHub release

---

### Hafta 9 *(Tarih: 06.06.2026)*

**Plandaki hedef:**
- Platform bağımsızlık testi (farklı Vulkan cihazda çalıştırma), benchmark tablosu

**Bu hafta yaptıklarım:**
- `src/vk_context.h/cpp` genişletildi:
  - `DeviceInfo` struct: cihaz adı, tipi, Vulkan API versiyonu
  - `enumerateDevices()`: compute queue'su olan tüm fiziksel cihazları listeler
  - `createContextForDevice(uint32_t index)`: belirtilen cihazı seçerek context oluşturur
  - `createContext()` → `createContextForDevice(0)` wrapper'ına dönüştürüldü (geriye dönük uyumlu)
- `benchmark/bench_platform.cpp` yazıldı: tüm cihazları enumerate et, her birinde iki ağ boyutunda benchmark yap, tablo yazdır
- `ctest` 4/4 test hâlâ geçiyor

**Benchmark tablosu (Release build, `bench_platform`):**

| Cihaz | Tip | Ağ | 1. çağrı (ms) | Tekrar başına (ms) | inf/sec |
|-------|-----|-----|---------------|-------------------|---------|
| Intel(R) Graphics (RPL-U) | Integrated GPU | 784→128→10 | 1.813 | 1.083 | 923 |
| llvmpipe (LLVM 15.0.7) | CPU | 784→128→10 | 21.334 | 0.660 | 1515 |
| Intel(R) Graphics (RPL-U) | Integrated GPU | 784→512→256→10 | 2.442 | 1.675 | 597 |
| llvmpipe (LLVM 15.0.7) | CPU | 784→512→256→10 | 2.561 | 1.111 | 899 |

**Gözlem:** Küçük ağda (784→128→10) CPU software renderer (llvmpipe) iGPU'yu geçiyor. Bunun nedeni: upload + submit + wait overhead'i küçük ağ compute süresini bastırıyor. Büyük ağda (784→512→256→10) fark kapanıyor. Platform bağımsızlık hedefi başarıyla doğrulandı: aynı kod, farklı Vulkan backend'lerinde çalışıyor.

**Plana göre durumum:**
- Haftanın tüm hedeflerine ulaşıldı ✅

**Karşılaştığım sorunlar / zorluklar:**
- Yok

**Gelecek hafta hedefim:**
- Kod temizliği, dokümantasyon, final rapor, GitHub release

---

### Hafta 6 + 7 *(Tarih: 02.06.2026 - 06.06.2026 — gecikme telafisi)*

**Plandaki hedef:**
- Hafta 6: VulkanMLP katman yönetimi, `addLayer()` altyapısı
- Hafta 7: Ağırlık yükleme (`loadWeights`), `forward()` GPU dispatch zinciri

**Bu hafta yaptıklarım:**
- `shaders/fc_layer.comp` yazıldı: GEMV + bias + opsiyonel ReLU shader
  - 4 binding (input, weights, bias, output), push constant `{inSize, outSize, useReLU}`
  - `local_size_x = 64`; her invocation bir çıkış nöronunu hesaplıyor
- `src/inference.h` yeniden tasarlandı: `Layer` struct (inSize, outSize, ComputePipeline, 3× GpuBuffer), `loadWeights()` API
- `src/inference.cpp` tam implementasyon:
  - `addLayer()`: fc_layer.spv'den pipeline yarat, weight/bias/output bufferları tahsis et, sıfırla
  - `loadWeights(idx, weights, biases)`: GPU buffer'lara upload et
  - `forward(input)`: inputBuf'a yükle → her katman için `dispatchPipeline` → son katmandan download
  - Son katmanda ReLU yok, aradakilerde var
- `tests/test_mlp.cpp`: 4→3→2 ağ, CPU referans + GPU karşılaştırma (hata toleransı 1e-4)
- `shaders/CMakeLists.txt` ve `CMakeLists.txt` güncellendi (`test_mlp` hedefi eklendi)
- `ctest` 4/4 test geçti (test_relu, test_matmul, test_buffer, test_mlp)

**Plana göre durumum:**
- Hafta 6+7 birlikte kapatıldı (2 hafta gecikme telafisi) ✅ ⚠️

**Karşılaştığım sorunlar / zorluklar:**
- `VulkanMLP` destructor'ı `destroyContext` çağrısından sonra çalışıyordu (dangling device handle); test'te scope ile çözüldü
- Hafta 6+7 gecikmeli tamamlandı (06.06.2026)

**Gelecek hafta hedefim:**
- Command buffer önceden kaydetme (pre-record) optimizasyonu, inference hız testleri

---

### Hafta 4 *(Tarih: 05.05.2025 - 11.05.2025)*

**Plandaki hedef:**
- Tiled matrix multiplication (GEMM) shader, workgroup optimizasyonu, `test_matmul` GPU dispatch testi

**Bu hafta yaptıklarım:**
- `matmul.comp`: 16×16 tiled GEMM shader (shared memory, push constant M/N/K, sınır kontrolü) — hafta 3'ten devredilen taslak tamamlandı
- `tests/test_matmul.cpp` yeniden yazıldı: CPU referans + 3 GPU dispatch testi (2×3×2, 64×64, 17×33×19 non-power-of-2)
- `shaders/CMakeLists.txt` hata düzeltmesi: `GLSLC_IS_GLSLANG_VALIDATOR` cache kaybolunca `-V` bayrağı düşüyordu; binary adı üzerinden algılama ile sabitlendi
- `CMakeLists.txt`: `test_matmul`'a `SHADER_DIR` tanımı eklendi
- `ctest` 3/3 test geçti (test_relu, test_matmul, test_buffer)

**Plana göre durumum:**
- Haftanın tüm hedeflerine ulaşıldı, hafta kapatıldı ✅

**Karşılaştığım sorunlar / zorluklar:**
- CMake cache'te `GLSLC=glslangValidator` saklı iken `GLSLC_IS_GLSLANG_VALIDATOR` değişkeni cache'lenmediği için `-V` bayrağı düşüp shader derlenemiyordu; binary adı kontrolüyle çözüldü

**Gelecek hafta hedefim:**
- CPU baseline GEMM implementasyonu, ilk CPU vs GPU benchmark ölçümleri (chrono + bellek bant genişliği)

---

### Hafta 3 *(Tarih: 28.04.2025 - 03.05.2025)*

**Plandaki hedef:**
- İlk compute shader (ReLU), SPIR-V derleme pipeline'ı, shader test altyapısı

**Bu hafta yaptıklarım:**
- `src/vk_pipeline.h` + `src/vk_pipeline.cpp` yazıldı: `loadSPIRV`, `createComputePipeline`, `destroyComputePipeline`, `dispatchPipeline` fonksiyonları
- `ComputePipeline` struct: descriptor set layout, pipeline layout, compute pipeline tek çatı altında
- `dispatchPipeline`: geçici descriptor pool/set oluşturma, buffer bağlama, command buffer kaydetme, submit + wait döngüsü
- `relu.comp` → `relu.spv` SPIR-V derleme pipeline'ı CMake'e entegre edildi (`glslangValidator` fallback desteği eklendi)
- `tests/test_relu.cpp` güncellendi: CPU referans + gerçek GPU dispatch + sonuç karşılaştırması
- `ctest` 3/3 test geçti (test_relu, test_matmul, test_buffer)

**Plana göre durumum:**
- Haftanın tüm hedeflerine ulaşıldı, hafta kapatıldı ✅
- Not: Vize sınavı nedeniyle hafta 2 bitti sonrası ~2 hafta ara verildi; plan tarihleri kaydırıldı.

**Karşılaştığım sorunlar / zorluklar:**
- `glslc` sistemde yüklü değil, `glslangValidator` ile çözüldü (CMake fallback eklendi)

**Gelecek hafta hedefim:**
- Tiled GEMM compute shader, workgroup optimizasyonu, `test_matmul` GPU dispatch testi

---

### Hafta 2 *(Tarih: 14.04.2025 - 20.04.2025)*

**Plandaki hedef:**
- Vulkan context kurulumu (VkInstance, VkDevice, compute queue), GPU buffer yönetimi

**Bu hafta yaptıklarım:**
- `VkCommandPool` `VkContext`'e eklendi (hafta 3 compute dispatch için hazırlık)
- `VkContext`: instance → physical device → logical device → compute queue → command pool akışı tamamlandı
- `GpuBuffer`: host-visible + host-coherent bellek, `uploadData` / `downloadData` çalışır hale getirildi
- `tests/test_buffer.cpp` yazıldı: float roundtrip, zero-fill ve command pool geçerlilik testleri

**Plana göre durumum:**
- Haftanın tüm hedeflerine ulaşıldı, hafta kapatıldı ✅

**Karşılaştığım sorunlar / zorluklar:**
- Yok

**Gelecek hafta hedefim:**
- İlk compute shader (ReLU) SPIR-V derleme pipeline'ı, shader dispatch altyapısı (descriptor set, pipeline layout, compute pipeline)

---

### Hafta 1 *(Tarih: 07.04.2025 - 13.04.2025)*

**Plandaki hedef:**
- Proje dizini, GitHub reposu, geliştirme ortamı kurulumu, CMake yapılandırması

**Bu hafta yaptıklarım:**
- Proje dizini oluşturuldu, CMake yapılandırması tamamlandı, GitHub reposu açıldı, geliştirme ortamı (Vulkan SDK, glslc, mesa-vulkan-drivers) kuruldu

**Plana göre durumum:**
- Haftanın hedeflerine ulaşıldı

**Karşılaştığım sorunlar / zorluklar:**
- Yok

**Gelecek hafta hedefim:**
- Vulkan context kurulumu (VkInstance, VkDevice, compute queue) ve GPU buffer yönetimi

---

<!--
ŞABLON: Yeni hafta eklemek için aşağıdaki bloğu kopyalayıp üste yapıştırın.

### Hafta X *(Tarih: GG.AA.YYYY - GG.AA.YYYY)*

**Plandaki hedef:**
-

**Bu hafta yaptıklarım:**
-

**Plana göre durumum:**
-

**Karşılaştığım sorunlar / zorluklar:**
-

**Gelecek hafta hedefim:**
-

---
-->
