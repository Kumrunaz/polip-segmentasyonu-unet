# =============================================================
# ADIM 2: VERİ ÖN İŞLEME, AUGMENTATION VE PİPELINE
# U-Net ile Polip Segmentasyonu - Final Projesi
# =============================================================

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

# ===== 2.1 Sabitler =====
IMG_SIZE = 256
BATCH_SIZE = 16
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_DIR = 'kvasir_seg/Kvasir-SEG/Kvasir-SEG/images/'
MASK_DIR = 'kvasir_seg/Kvasir-SEG/Kvasir-SEG/masks/'
image_paths = sorted(glob(IMAGE_DIR + '*.jpg') + glob(IMAGE_DIR + '*.png'))
mask_paths = sorted(glob(MASK_DIR + '*.jpg') + glob(MASK_DIR + '*.png'))

print(f"Goruntu boyutu: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch boyutu:   {BATCH_SIZE}")


# ===== 2.2 On Isleme Fonksiyonu =====
def load_and_preprocess(image_path, mask_path):
    """
    Tek bir goruntu-maske ciftini oku ve isle.
    - BGR -> RGB donusumu
    - 256x256 resize (goruntu: INTER_AREA, maske: INTER_NEAREST)
    - /255.0 ile [0,1] araligina normalize
    - Maske icin >0.5 threshold (binary: 0 veya 1)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)

    return img, mask


# ===== 2.3 Tek Ornek Testi =====
test_img, test_mask = load_and_preprocess(image_paths[0], mask_paths[0])

print(f"\nOn isleme testi:")
print(f"  Goruntu shape: {test_img.shape}")
print(f"  Goruntu min/max: [{test_img.min():.3f}, {test_img.max():.3f}]")
print(f"  Maske shape:   {test_mask.shape}")
print(f"  Maske benzersiz degerler: {np.unique(test_mask)}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('On Isleme Sonucu (Tek Ornek)', fontsize=14, fontweight='bold')
axes[0].imshow(test_img)
axes[0].set_title(f'Goruntu {test_img.shape}')
axes[0].axis('off')
axes[1].imshow(test_mask, cmap='gray')
axes[1].set_title(f'Maske (binary) {test_mask.shape}')
axes[1].axis('off')
overlay = test_img.copy()
overlay[test_mask > 0.5, 1] = np.clip(overlay[test_mask > 0.5, 1] + 0.4, 0, 1)
axes[2].imshow(overlay)
axes[2].set_title('Overlay')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('on_isleme_testi.png', dpi=150, bbox_inches='tight')
plt.show()


# ===== 2.4 Tum Veriyi Yukle =====
print("\nTum veriler yukleniyor...")
X_data, Y_data = [], []
for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
    img, mask = load_and_preprocess(img_path, mask_path)
    X_data.append(img)
    Y_data.append(mask)

X_data = np.array(X_data)                    # (1000, 256, 256, 3)
Y_data = np.array(Y_data)[..., np.newaxis]   # (1000, 256, 256, 1)

print(f"\nGoruntuler: {X_data.shape} | dtype: {X_data.dtype}")
print(f"Maskeler:   {Y_data.shape} | dtype: {Y_data.dtype}")
print(f"Bellekte kaplanan alan: {(X_data.nbytes + Y_data.nbytes) / 1e9:.2f} GB")


# ===== 2.5 Train / Validation / Test Split =====
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_data, Y_data, test_size=0.30, random_state=SEED
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.50, random_state=SEED
)

print(f"\n{'='*50}")
print(f"VERI BOLME SONUCU")
print(f"{'='*50}")
print(f"Egitim seti:    {X_train.shape[0]} goruntu ({X_train.shape[0]/len(X_data)*100:.0f}%)")
print(f"Dogrulama seti: {X_val.shape[0]} goruntu ({X_val.shape[0]/len(X_data)*100:.0f}%)")
print(f"Test seti:      {X_test.shape[0]} goruntu ({X_test.shape[0]/len(X_data)*100:.0f}%)")
print(f"{'='*50}")


# ===== 2.6 Set Dengesi Kontrolu =====
def calc_polyp_ratio(masks):
    return np.array([masks[i].mean() for i in range(len(masks))])

train_ratios = calc_polyp_ratio(Y_train)
val_ratios = calc_polyp_ratio(Y_val)
test_ratios = calc_polyp_ratio(Y_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Setler Arasi Polip Orani Dagilimi (Denge Kontrolu)', fontsize=14, fontweight='bold')

for ax, ratios, name, color in zip(axes,
    [train_ratios, val_ratios, test_ratios],
    ['Egitim', 'Dogrulama', 'Test'],
    ['#378ADD', '#1D9E75', '#D85A30']):
    ax.hist(ratios * 100, bins=20, color=color, edgecolor='white', alpha=0.8)
    ax.axvline(ratios.mean() * 100, color='red', linestyle='--',
               label=f'Ort: %{ratios.mean()*100:.1f}')
    ax.set_title(f'{name} ({len(ratios)} goruntu)')
    ax.set_xlabel('Polip Orani (%)')
    ax.set_ylabel('Frekans')
    ax.legend()

plt.tight_layout()
plt.savefig('set_denge_kontrolu.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Ortalama polip orani -> Egitim: %{train_ratios.mean()*100:.1f} | "
      f"Dogrulama: %{val_ratios.mean()*100:.1f} | Test: %{test_ratios.mean()*100:.1f}")


# ===== 2.7 Augmentation (Albumentations) =====
aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT),
    A.ElasticTransform(alpha=50, sigma=10, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
])


def augment_albumentations(image, mask):
    augmented = aug_transform(
        image=(image * 255).astype(np.uint8),
        mask=(mask[:, :, 0] * 255).astype(np.uint8)
    )
    aug_img = augmented['image'].astype(np.float32) / 255.0
    aug_mask = (augmented['mask'] > 127).astype(np.float32)
    return aug_img, aug_mask[..., np.newaxis]


def tf_augment(image, mask):
    def _augment(img, msk):
        aug_img, aug_msk = augment_albumentations(img.numpy(), msk.numpy())
        return aug_img, aug_msk
    image, mask = tf.py_function(_augment, [image, mask], [tf.float32, tf.float32])
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return image, mask


# Augmentation gorsel kontrolu
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('v2 Augmentation Ornekleri', fontsize=14, fontweight='bold')

sample_img = X_train[0]
sample_mask = Y_train[0]

axes[0, 0].imshow(sample_img)
axes[0, 0].set_title('Orijinal', fontweight='bold')
axes[0, 0].axis('off')
axes[0, 1].imshow(sample_mask[:,:,0], cmap='gray')
axes[0, 1].set_title('Orijinal Maske', fontweight='bold')
axes[0, 1].axis('off')

positions = [(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]
for k in range(5):
    aug_img, aug_mask = augment_albumentations(sample_img, sample_mask)
    r, c = positions[k*2], positions[k*2+1]
    axes[r[0], r[1]].imshow(aug_img)
    axes[r[0], r[1]].set_title(f'Aug #{k+1}')
    axes[r[0], r[1]].axis('off')
    axes[c[0], c[1]].imshow(aug_mask[:,:,0], cmap='gray')
    axes[c[0], c[1]].set_title(f'Maske #{k+1}')
    axes[c[0], c[1]].axis('off')

plt.tight_layout()
plt.savefig('augmentation_ornekleri.png', dpi=150, bbox_inches='tight')
plt.show()


# ===== 2.8 tf.data Pipeline =====
train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_ds = (train_ds.shuffle(500, seed=SEED)
            .map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"\nPipeline hazir!")
print(f"  Egitim:    {tf.data.experimental.cardinality(train_ds).numpy()} batch")
print(f"  Dogrulama: {tf.data.experimental.cardinality(val_ds).numpy()} batch")
print(f"  Test:      {tf.data.experimental.cardinality(test_ds).numpy()} batch")


# ===== 2.9 Ozet =====
print(f"\n{'='*60}")
print(f"  ADIM 2 OZETI - VERI ON ISLEME")
print(f"{'='*60}")
print(f"  Resize:          {IMG_SIZE}x{IMG_SIZE}")
print(f"  Normalize:       /255.0 -> [0, 1]")
print(f"  Maske threshold: >0.5 -> binary (0 veya 1)")
print(f"  Maske resize:    INTER_NEAREST (kenar koruma)")
print(f"  Egitim:          {X_train.shape[0]} goruntu (augmented)")
print(f"  Dogrulama:       {X_val.shape[0]} goruntu")
print(f"  Test:            {X_test.shape[0]} goruntu")
print(f"  Augmentation:    Albumentations (7 tur)")
print(f"  Pipeline:        tf.data (shuffle + prefetch)")
print(f"  Batch size:      {BATCH_SIZE}")
print(f"{'='*60}")
