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
| 2  | 14.04 - 20.04 | Vulkan context kurulumu (VkInstance, VkDevice, compute queue), GPU buffer yönetimi | %20 | 🔄 Devam Ediyor |
| 3  | 21.04 - 27.04 | İlk compute shader (ReLU), SPIR-V derleme pipeline'ı, shader test altyapısı | %30 | ⬜ Başlamadı |
| 4  | 28.04 - 04.05 | Tiled matrix multiplication (GEMM) shader, workgroup optimizasyonu | %42 | ⬜ Başlamadı |
| 5  | 05.05 - 11.05 | CPU baseline implementasyonu, ilk CPU vs GPU benchmark ölçümleri | %54 | ⬜ Başlamadı |
| 6  | 12.05 - 18.05 | ECS mimarisiyle katman yönetimi, VulkanMLP sınıfı iskelet kodu | %64 | ⬜ Başlamadı |
| 7  | 19.05 - 25.05 | MNIST ağırlıklarını yükleme, 784→128→10 MLP inference pipeline | %75 | ⬜ Başlamadı |
| 8  | 26.05 - 01.06 | Command buffer önceden kaydetme optimizasyonu, inference hız testleri | %84 | ⬜ Başlamadı |
| 9  | 02.06 - 08.06 | Platform bağımsızlık testi (farklı Vulkan cihazda çalıştırma), benchmark tablosu | %92 | ⬜ Başlamadı |
| 10 | 09.06 - 15.06 | Kod temizliği, dokümantasyon, final rapor, GitHub release | %100 | ⬜ Başlamadı |

**Durum simgeleri:** ⬜ Başlamadı | 🔄 Devam Ediyor | ✅ Tamamlandı | ⚠️ Gecikti

---

## Haftalık İlerleme Kayıtları

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
- Haftanın hedeflerine ulaşıldı

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
