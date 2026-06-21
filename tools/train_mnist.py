#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# train_mnist.py
#
# Danışman geri bildirimi #2: kütüphanenin yalnızca HIZLI değil DOĞRU da
# çalıştığını uçtan uca göstermek için gerçek bir görev (MNIST sınıflandırma).
#
# Bu script:
#   1. MNIST veri setini indirir (PyTorch S3 aynası, IDX formatı),
#   2. 784 -> 128 -> 10 bir MLP'yi (ReLU + softmax) numpy ile sıfırdan eğitir,
#   3. ağırlıkları models/mnist_mlp.bin olarak (kütüphanenin loadWeights
#      düzeninde: W[i*inSize+j], ardından bias) yazar,
#   4. test setini models/mnist_test.bin olarak (görüntü + etiket) yazar.
#
# Ardından `examples/mnist_infer` çalıştırılınca AYNI ağırlıklarla Vulkan
# üzerinde inference yapılır ve test doğruluğu raporlanır. Böylece kütüphanenin
# doğruluğu gerçek bir modelde kanıtlanır.
#
# Bağımlılık: yalnızca numpy.
#   python3 tools/train_mnist.py
# ---------------------------------------------------------------------------

import gzip
import os
import struct
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
CACHE = os.path.join(MODELS, "mnist_raw")

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

IN, H, OUT = 784, 128, 10


def download(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, FILES[name])
    if not os.path.exists(path):
        url = MIRROR + FILES[name]
        print(f"  indiriliyor: {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_images(name):
    with gzip.open(download(name), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"kotu magic {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float32) / 255.0


def load_labels(name):
    with gzip.open(download(name), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"kotu magic {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def one_hot(y, k=10):
    o = np.zeros((y.size, k), dtype=np.float32)
    o[np.arange(y.size), y] = 1.0
    return o


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    print("MNIST yukleniyor...")
    Xtr, Ytr = load_images("train_images"), load_labels("train_labels")
    Xte, Yte = load_images("test_images"), load_labels("test_labels")
    print(f"  train={Xtr.shape}  test={Xte.shape}")

    rng = np.random.default_rng(42)
    # He init (ReLU icin)
    W0 = (rng.standard_normal((H, IN)).astype(np.float32) * np.sqrt(2.0 / IN))
    b0 = np.zeros(H, dtype=np.float32)
    W1 = (rng.standard_normal((OUT, H)).astype(np.float32) * np.sqrt(2.0 / H))
    b1 = np.zeros(OUT, dtype=np.float32)

    lr, epochs, bs = 0.1, 15, 128
    Ytr_oh = one_hot(Ytr)
    n = Xtr.shape[0]

    print("egitim basliyor...")
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            x, t = Xtr[b], Ytr_oh[b]
            # forward (shader ile ayni: z = x @ W.T + b, ReLU, sonra softmax)
            z0 = x @ W0.T + b0
            a0 = np.maximum(0.0, z0)
            z1 = a0 @ W1.T + b1
            p = softmax(z1)
            # backward
            m = x.shape[0]
            dz1 = (p - t) / m
            dW1 = dz1.T @ a0
            db1 = dz1.sum(axis=0)
            da0 = dz1 @ W1
            dz0 = da0 * (z0 > 0)
            dW0 = dz0.T @ x
            db0 = dz0.sum(axis=0)
            W1 -= lr * dW1; b1 -= lr * db1
            W0 -= lr * dW0; b0 -= lr * db0

        # epoch sonu test dogrulugu
        a0 = np.maximum(0.0, Xte @ W0.T + b0)
        pred = (a0 @ W1.T + b1).argmax(axis=1)
        acc = (pred == Yte).mean()
        print(f"  epoch {ep+1:2d}/{epochs}  test_acc={acc*100:.2f}%")

    os.makedirs(MODELS, exist_ok=True)

    # --- agirliklari yaz: "MLP1" formati ---
    wpath = os.path.join(MODELS, "mnist_mlp.bin")
    with open(wpath, "wb") as f:
        f.write(struct.pack("<i", 0x314D4C50))  # 'MLP1' (little-endian)
        f.write(struct.pack("<i", 2))           # katman sayisi
        for (W, b) in [(W0, b0), (W1, b1)]:
            outS, inS = W.shape
            f.write(struct.pack("<ii", inS, outS))
            f.write(W.astype("<f4").tobytes())   # row-major W[i*inS+j]
            f.write(b.astype("<f4").tobytes())
    print(f"yazildi: {wpath}")

    # --- test setini yaz: "MNST" formati ---
    tpath = os.path.join(MODELS, "mnist_test.bin")
    with open(tpath, "wb") as f:
        f.write(struct.pack("<i", 0x54534E4D))  # 'MNST'
        f.write(struct.pack("<ii", Xte.shape[0], IN))
        f.write(Xte.astype("<f4").tobytes())
        f.write(Yte.astype("<i4").tobytes())
    print(f"yazildi: {tpath}  ({Xte.shape[0]} ornek)")
    print("\nSimdi:  ./build-release/mnist_infer")


if __name__ == "__main__":
    sys.exit(main())
