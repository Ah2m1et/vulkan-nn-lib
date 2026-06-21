// ---------------------------------------------------------------------------
// mnist_infer.cpp
//
// Danışman geri bildirimi #2: kütüphanenin gerçek bir görevde DOĞRU çalıştığını
// uçtan uca gösterir. tools/train_mnist.py ile eğitilen 784->128->10 MLP'nin
// ağırlıklarını yükler, MNIST test setinin tamamında Vulkan üzerinde inference
// yapar ve SINIFLANDIRMA DOĞRULUĞUNU raporlar.
//
// Ayrıca ilk N örnekte GPU çıktısını CPU referansıyla karşılaştırarak
// kütüphanenin sayısal doğruluğunu da doğrular (yalnızca hızlı değil, doğru).
//
//   1) python3 tools/train_mnist.py      # models/*.bin uretir
//   2) ./build-release/mnist_infer
// ---------------------------------------------------------------------------

#include "vk_context.h"
#include "inference.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MODELS_DIR
#define MODELS_DIR "models"
#endif

struct LayerW {
    int32_t inSize, outSize;
    std::vector<float> weights;  // outSize*inSize, satir-major
    std::vector<float> bias;     // outSize
};

template <typename T>
static T readPod(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("dosya beklenenden kisa");
    return v;
}

static std::vector<LayerW> loadWeights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("agirlik dosyasi acilamadi: " + path +
                                     "\n  Once: python3 tools/train_mnist.py");
    if (readPod<int32_t>(f) != 0x314D4C50)
        throw std::runtime_error("kotu magic (MLP1 bekleniyordu)");
    int32_t n = readPod<int32_t>(f);
    std::vector<LayerW> layers(n);
    for (auto& L : layers) {
        L.inSize  = readPod<int32_t>(f);
        L.outSize = readPod<int32_t>(f);
        L.weights.resize((size_t)L.inSize * L.outSize);
        L.bias.resize(L.outSize);
        f.read(reinterpret_cast<char*>(L.weights.data()),
               L.weights.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(L.bias.data()),
               L.bias.size() * sizeof(float));
        if (!f) throw std::runtime_error("agirlik verisi eksik");
    }
    return layers;
}

struct TestSet {
    int32_t count, dim;
    std::vector<float>   images;  // count*dim
    std::vector<int32_t> labels;  // count
};

static TestSet loadTestSet(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("test dosyasi acilamadi: " + path +
                                     "\n  Once: python3 tools/train_mnist.py");
    if (readPod<int32_t>(f) != 0x54534E4D)
        throw std::runtime_error("kotu magic (MNST bekleniyordu)");
    TestSet ts;
    ts.count = readPod<int32_t>(f);
    ts.dim   = readPod<int32_t>(f);
    ts.images.resize((size_t)ts.count * ts.dim);
    ts.labels.resize(ts.count);
    f.read(reinterpret_cast<char*>(ts.images.data()),
           ts.images.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(ts.labels.data()),
           ts.labels.size() * sizeof(int32_t));
    if (!f) throw std::runtime_error("test verisi eksik");
    return ts;
}

// CPU referans forward (test_mlp ile ayni semantik: son katman haric ReLU).
static std::vector<float> cpuForward(const std::vector<float>& in,
                                     const std::vector<LayerW>& layers) {
    std::vector<float> act = in;
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& L = layers[l];
        std::vector<float> out(L.outSize);
        bool relu = (l + 1 < layers.size());
        for (int i = 0; i < L.outSize; ++i) {
            float acc = L.bias[i];
            for (int j = 0; j < L.inSize; ++j)
                acc += L.weights[(size_t)i * L.inSize + j] * act[j];
            out[i] = relu ? (acc > 0.f ? acc : 0.f) : acc;
        }
        act = std::move(out);
    }
    return act;
}

static int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

int main() {
    const std::string wpath = std::string(MODELS_DIR) + "/mnist_mlp.bin";
    const std::string tpath = std::string(MODELS_DIR) + "/mnist_test.bin";

    auto layers = loadWeights(wpath);
    auto ts     = loadTestSet(tpath);

    printf("=== MNIST uctan uca dogrulama (Vulkan inference) ===\n");
    printf("Model: %d", layers.front().inSize);
    for (const auto& L : layers) printf("->%d", L.outSize);
    printf("  | test ornegi: %d\n\n", ts.count);

    VkContext ctx = createContext();
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physDevice, &props);
    printf("Cihaz: %s\n\n", props.deviceName);

    // mlp, destroyContext'ten ÖNCE yok edilmeli (aksi halde dangling device
    // handle → vkFreeCommandBuffers segfault). Bu yüzden ayrı bir scope içinde.
    float  maxAbsDiff = 0.0f;
    double acc        = 0.0;
    {
    VulkanMLP mlp(ctx, SHADER_DIR);
    for (auto& L : layers) mlp.addLayer(L.inSize, L.outSize);
    for (size_t i = 0; i < layers.size(); ++i)
        mlp.loadWeights(i, layers[i].weights, layers[i].bias);

    // --- 1) Sayisal dogruluk: GPU vs CPU referans (ilk N ornek) ---
    const int N_CHECK = 50;
    for (int s = 0; s < N_CHECK && s < ts.count; ++s) {
        std::vector<float> img(ts.images.begin() + (size_t)s * ts.dim,
                               ts.images.begin() + (size_t)(s + 1) * ts.dim);
        auto g = mlp.forward(img);
        auto c = cpuForward(img, layers);
        for (size_t i = 0; i < g.size(); ++i)
            maxAbsDiff = std::max(maxAbsDiff, std::abs(g[i] - c[i]));
    }
    printf("GPU vs CPU referans (ilk %d ornek): max |fark| = %.2e  -> %s\n",
           N_CHECK, maxAbsDiff,
           maxAbsDiff < 1e-3f ? "TUTARLI (dogru)" : "TUTARSIZ (!!)");

    // --- 2) Tam test setinde siniflandirma dogrulugu ---
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    int correct = 0;
    for (int s = 0; s < ts.count; ++s) {
        std::vector<float> img(ts.images.begin() + (size_t)s * ts.dim,
                               ts.images.begin() + (size_t)(s + 1) * ts.dim);
        if (argmax(mlp.forward(img)) == ts.labels[s]) ++correct;
    }
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();

    acc = 100.0 * correct / ts.count;
    printf("\nSiniflandirma dogrulugu : %d / %d = %.2f%%\n", correct, ts.count, acc);
    printf("Toplam inference suresi  : %.2f s  (%.0f ornek/sn)\n",
           secs, ts.count / secs);
    printf("\nSonuc: kutuphane gercek bir modeli uctan uca DOGRU calistiriyor.\n");
    } // mlp burada yok edilir (context hâlâ geçerli)

    destroyContext(ctx);
    return (maxAbsDiff < 1e-3f && acc > 90.0) ? 0 : 1;
}
