// ---------------------------------------------------------------------------
// bench_soa_vs_aos.cpp
//
// Veri Temelli Programlama (DOP) katkısının NİCEL ölçümü (danışman geri
// bildirimi #1). Tezin başlığındaki "veri temelli bellek düzeni verimliliği
// artırır" iddiasını somut sayıya çevirir.
//
// AYNI MLP forward hesabı (GEMV + bias + ReLU), AYNI ağırlıklarla, iki farklı
// bellek düzeniyle yapılır; tek fark veri yerleşimi/erişim biçimidir:
//
//   * AoS (Array of Structs / nesne-temelli, klasik OOP):
//       her nöron ayrı heap'te yaşayan bir nesnedir ve nöronlar bellekte
//       DAĞINIK sırada gezilir (gerçek bir OOP programında nesnelerin zamanla
//       saçılmasının taklidi). Her nörona erişim bir işaretçi kovalama +
//       muhtemel cache-miss demektir.
//
//   * SoA (Struct of Arrays / veri-temelli):
//       tüm ağırlıklar tek bir BİTİŞİK matriste, bias/çıkış bitişik dizilerde.
//       Bu, kütüphanenin GPU storage-buffer düzeniyle ve EnTT'nin bileşenleri
//       bitişik havuzlarda tutan ECS felsefesiyle BİREBİR aynıdır.
//
// ÖNEMLİ GÖZLEM: Etki, nöron başına iş miktarına bağlıdır. Geniş katmanlarda
// (büyük inSize) tek bir cache-miss yüzlerce FLOP'a bölündüğü için fark küçüktür;
// ÇOK SAYIDA KÜÇÜK nöronda (ECS'nin tipik senaryosu) DOP'un kazancı belirginleşir.
// Bu yüzden inSize küçüldükçe hızlanmanın nasıl arttığı taranır.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
using ms_d  = std::chrono::duration<double, std::milli>;

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// ---------------------------------------------------------------------------
// AoS — klasik nesne-temelli temsil
// ---------------------------------------------------------------------------
struct Neuron {
    std::vector<float> weights;  // inSize ağırlık — ayrı heap tahsisi
    float              bias;
    uint32_t           id;       // mantıksal çıkış indeksi
};

struct LayerAoS {
    std::vector<std::unique_ptr<Neuron>> storage;  // sahiplik
    std::vector<Neuron*>                 order;     // GEZME sırası (karıştırılmış)
    uint32_t inSize;
};

struct LayerSoA {
    std::vector<float> weights;  // outSize * inSize, satır-major, BİTİŞİK
    std::vector<float> bias;     // outSize, bitişik
    uint32_t inSize, outSize;
};

// AoS katmanını üret; gezme sırasını karıştır ki erişim bellekte dağınık olsun.
static LayerAoS makeLayerAoS(uint32_t inSize, uint32_t outSize, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.1f);
    LayerAoS layer;
    layer.inSize = inSize;
    layer.storage.resize(outSize);
    layer.order.resize(outSize);
    for (uint32_t i = 0; i < outSize; ++i) {
        auto n = std::make_unique<Neuron>();
        n->weights.resize(inSize);
        for (auto& w : n->weights) w = dist(rng);
        n->bias = dist(rng);
        n->id   = i;
        layer.order[i] = n.get();
        layer.storage[i] = std::move(n);
    }
    std::shuffle(layer.order.begin(), layer.order.end(), rng);  // dağınık erişim
    return layer;
}

// SoA'yı AoS'tan KOPYALAYARAK üret → iki düzen birebir aynı ağırlığa sahip olur.
static LayerSoA makeLayerSoAfrom(const LayerAoS& a) {
    LayerSoA s;
    s.inSize  = a.inSize;
    s.outSize = (uint32_t)a.storage.size();
    s.weights.resize((size_t)s.inSize * s.outSize);
    s.bias.resize(s.outSize);
    for (uint32_t i = 0; i < s.outSize; ++i) {
        const Neuron* n = a.storage[i].get();
        std::copy(n->weights.begin(), n->weights.end(),
                  s.weights.begin() + (size_t)i * s.inSize);
        s.bias[i] = n->bias;
    }
    return s;
}

static void forwardAoS(const LayerAoS& layer, const std::vector<float>& in,
                       std::vector<float>& out) {
    for (const Neuron* n : layer.order) {       // dağınık işaretçi kovalama
        float acc = n->bias;
        const float* w = n->weights.data();
        for (uint32_t j = 0; j < layer.inSize; ++j)
            acc += w[j] * in[j];
        out[n->id] = relu(acc);
    }
}

static void forwardSoA(const LayerSoA& layer, const std::vector<float>& in,
                       std::vector<float>& out) {
    const float* W = layer.weights.data();
    const float* I = in.data();
    for (uint32_t i = 0; i < layer.outSize; ++i) {
        const float* w = W + (size_t)i * layer.inSize;  // bitişik satır
        float acc = layer.bias[i];
        for (uint32_t j = 0; j < layer.inSize; ++j)
            acc += w[j] * I[j];
        out[i] = relu(acc);
    }
}

// ---------------------------------------------------------------------------
struct Case { uint32_t inSize, outSize; int reps; const char* label; };

int main() {
    // Her durumda toplam ağırlık sayısı ~ sabit (~4M); inSize küçüldükçe nöron
    // sayısı (= cache-miss sayısı) artar, nöron başına iş azalır.
    const Case cases[] = {
        {784,  5000,  200, "784  x 5000   (genis kat.)"},
        {256, 16000,  200, "256  x 16000"},
        { 64, 64000,  200, "64   x 64000"},
        { 16, 256000, 200, "16   x 256000 (cok kucuk)"},
        {  4, 512000, 200, "4    x 512000 (uc nokta)"},
    };

    printf("\nVeri Temelli Programlama (DOP) etkisi — SoA vs AoS bellek duzeni\n");
    printf("Ayni aritmetik + ayni agirliklar; tek fark veri yerlesimi/erisimi.\n");
    printf("inSize kuculdukce noron (=cache-miss) sayisi artar → DOP kazanci buyur.\n\n");
    printf("%-28s  %-11s  %-11s  %-10s\n",
           "Katman (inSize x noron)", "AoS (ms)", "SoA (ms)", "Hizlanma");
    printf("%-28s  %-11s  %-11s  %-10s\n",
           "----------------------------", "-----------", "-----------",
           "----------");

    std::mt19937 rng(42);
    for (const auto& c : cases) {
        std::vector<float> in(c.inSize), outA(c.outSize), outS(c.outSize);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : in) x = dist(rng);

        LayerAoS la = makeLayerAoS(c.inSize, c.outSize, rng);
        LayerSoA ls = makeLayerSoAfrom(la);

        // --- Doğruluk: iki düzen aynı sonucu vermeli ---
        forwardAoS(la, in, outA);
        forwardSoA(ls, in, outS);
        float maxDiff = 0.0f;
        for (uint32_t i = 0; i < c.outSize; ++i)
            maxDiff = std::max(maxDiff, std::abs(outA[i] - outS[i]));

        // --- AoS ölçüm ---
        volatile float sinkA = 0.0f;
        auto t0 = Clock::now();
        for (int r = 0; r < c.reps; ++r) { forwardAoS(la, in, outA); sinkA += outA[0]; }
        double aosMs = ms_d(Clock::now() - t0).count() / c.reps;

        // --- SoA ölçüm ---
        volatile float sinkS = 0.0f;
        t0 = Clock::now();
        for (int r = 0; r < c.reps; ++r) { forwardSoA(ls, in, outS); sinkS += outS[0]; }
        double soaMs = ms_d(Clock::now() - t0).count() / c.reps;

        printf("%-28s  %-11.3f  %-11.3f  %-8.2fx %s\n",
               c.label, aosMs, soaMs, aosMs / soaMs,
               maxDiff < 1e-3f ? "" : " [!! sonuc farkli]");
        (void)sinkA; (void)sinkS;
    }

    printf("\nNot: SoA (bitisik bilesen dizileri) kutuphanenin GPU buffer ve EnTT\n");
    printf("bilesen havuzu duzeniyle aynidir; AoS klasik dagilmis nesne modelidir.\n");
    return 0;
}
