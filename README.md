# U-Net ile Gastrointestinal Polip Segmentasyonu

Kvasir-SEG veri seti üzerinde U-Net mimarisi kullanılarak kolonoskopi görüntülerinde polip segmentasyonu.

## Proje Özeti
Bu projede, 1000 kolonoskopi görüntüsü ve gastroenterolog onaylı maskelerden oluşan Kvasir-SEG veri seti kullanılarak piksel düzeyinde polip segmentasyonu gerçekleştirilmiştir. U-Net mimarisi sıfırdan implement edilmiş, iteratif bir geliştirme süreciyle hiperparametre optimizasyonu yapılmıştır.

## Sonuçlar

| Metrik | v1 | v2 (Final) | İyileşme |
|--------|-----|------------|----------|
| Dice Coefficient | 0.6119 | **0.8113** | +%33 |
| IoU (Jaccard) | 0.5010 | **0.7211** | +%44 |
| Pixel Accuracy | 0.9091 | **0.9455** | +%4 |


## Metodoloji

### Adım 1: Veri Keşfi
- Kvasir-SEG veri setinin yapısal analizi (1000 görüntü, 333 farklı çözünürlük)
- Polip/arka plan oran analizi (ortalama %15.4 — sınıf dengesizliği tespiti)
- Piksel değer dağılımı ve maske kalite kontrolü

### Adım 2: Veri Ön İşleme
- Resize (256x256), normalizasyon (/255), threshold (binary maske)
- Train/Val/Test: %70/%15/%15 (700/150/150)
- Albumentations augmentation: Flip, Rotation(±30°), ElasticTransform, BrightnessContrast, HueSaturationValue, GaussianBlur
- tf.data pipeline (shuffle + augment + batch + prefetch)

### Adım 3: Model ve Eğitim
- **Mimari:** U-Net (Encoder-Decoder + Skip Connections), 31M parametre
- **Loss:** BCE + Dice Loss
- **Optimizer:** Adam + Cosine Decay (lr: 1e-4 → 1e-6)
- **Callbacks:** ModelCheckpoint, EarlyStopping (patience=15)

### v1 → v2 İyileştirmeler
| Değişiklik | v1 | v2 |
|-----------|-----|-----|
| Learning Rate | 1e-3 | 1e-4 |
| Epoch | 50 | 100 |
| LR Scheduler | ReduceLROnPlateau | Cosine Decay |
| Augmentation | tf.image (4 tür) | Albumentations (7 tür) |

## Teknolojiler

- Python 3.12
- TensorFlow / Keras
- OpenCV
- Albumentations
- NumPy, Matplotlib, scikit-learn
- Google Colab Pro (T4 GPU)

## Veri Seti

[Kvasir-SEG](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) — Jha et al. (2020)

## Kullanım

1. Herhangi bir notebook'u Google Colab'da açın
2. Runtime → Change runtime type → T4 GPU
3. Kaggle API key'inizi yükleyin (`kaggle.json`)
4. Hücreleri sırayla çalıştırın


Danışman: Doç. Dr. Murat Canayaz
