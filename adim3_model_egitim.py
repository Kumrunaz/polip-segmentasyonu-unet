# =============================================================
# ADIM 3: U-NET MODELİ, EĞİTİM VE DEĞERLENDİRME (v2 Final)
# U-Net ile Polip Segmentasyonu - Final Projesi
# =============================================================

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

# ===== SABITLER =====
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
INIT_LR = 1e-4
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TF: {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")


# =============================================================
# VERİ HAZIRLAMA (Adim 1-2 tekrar)
# =============================================================

IMAGE_DIR = 'kvasir_seg/Kvasir-SEG/Kvasir-SEG/images/'
MASK_DIR = 'kvasir_seg/Kvasir-SEG/Kvasir-SEG/masks/'
image_paths = sorted(glob(IMAGE_DIR + '*.jpg') + glob(IMAGE_DIR + '*.png'))
mask_paths = sorted(glob(MASK_DIR + '*.jpg') + glob(MASK_DIR + '*.png'))


def load_and_preprocess(image_path, mask_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    return img, mask


print("Veriler yukleniyor...")
X_data, Y_data = [], []
for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
    img, mask = load_and_preprocess(img_path, mask_path)
    X_data.append(img)
    Y_data.append(mask)
X_data = np.array(X_data)
Y_data = np.array(Y_data)[..., np.newaxis]

X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=0.30, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=SEED)
print(f"Egitim: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")


# Augmentation
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


# Pipeline
train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_ds = (train_ds.shuffle(500, seed=SEED)
            .map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# =============================================================
# U-NET MİMARİSİ
# =============================================================

def conv_block(inputs, num_filters):
    """
    Temel konvolusyon blogu.
    Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    """
    x = layers.Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def encoder_block(inputs, num_filters):
    """
    Encoder blogu: conv_block + MaxPooling + Dropout
    skip: Decoder'a gonderilecek ozellik haritasi
    pool: Bir sonraki encoder katmanina gidecek kucultulmus veri
    """
    skip = conv_block(inputs, num_filters)
    pool = layers.MaxPooling2D(2)(skip)
    pool = layers.Dropout(0.2)(pool)
    return skip, pool


def decoder_block(inputs, skip_connection, num_filters):
    """
    Decoder blogu: UpSampling + Skip Connection + conv_block
    Conv2DTranspose ile boyut 2x buyutulur, encoder'dan gelen
    skip connection ile birlestirilir.
    """
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_connection])
    x = layers.Dropout(0.2)(x)
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape=(256, 256, 3)):
    """
    Tam U-Net modeli.
    Encoder:    64 -> 128 -> 256 -> 512
    Bottleneck: 1024
    Decoder:    512 -> 256 -> 128 -> 64
    Cikis:      1 kanal (sigmoid)
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    skip1, pool1 = encoder_block(inputs, 64)    # 256->128
    skip2, pool2 = encoder_block(pool1, 128)     # 128->64
    skip3, pool3 = encoder_block(pool2, 256)     # 64->32
    skip4, pool4 = encoder_block(pool3, 512)     # 32->16

    # Bottleneck
    bottleneck = conv_block(pool4, 1024)          # 16x16x1024

    # Decoder
    d1 = decoder_block(bottleneck, skip4, 512)   # 16->32
    d2 = decoder_block(d1, skip3, 256)            # 32->64
    d3 = decoder_block(d2, skip2, 128)            # 64->128
    d4 = decoder_block(d3, skip1, 64)             # 128->256

    # Cikis
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='U-Net-v2')
    return model


model = build_unet()
print(f"Model: {model.name} | Parametre: {model.count_params():,}")
model.summary()


# =============================================================
# LOSS FONKSİYONLARI VE METRİKLER
# =============================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice = 2 * |A kesisim B| / (|A| + |B|)"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """Dice Loss = 1 - Dice Coefficient"""
    return 1 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Kombine Loss = BCE + Dice Loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def iou_metric(y_true, y_pred, smooth=1e-6):
    """IoU = |A kesisim B| / |A birlesim B|"""
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred_bin)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)


# =============================================================
# EĞİTİM YAPILANDIRMASI (v2)
# =============================================================

# Cosine Decay LR Scheduler
steps_per_epoch = len(X_train) // BATCH_SIZE
total_steps = steps_per_epoch * EPOCHS

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=total_steps,
    alpha=1e-6
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=bce_dice_loss,
    metrics=[dice_coefficient, iou_metric, 'accuracy']
)

callbacks = [
    ModelCheckpoint(
        'best_unet_polyp_v2.keras',
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_dice_coefficient',
        mode='max',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

print(f"\nv2 Egitim Yapilandirmasi:")
print(f"  LR: {INIT_LR} (Cosine Decay)")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch: {BATCH_SIZE}")


# =============================================================
# EĞİTİM
# =============================================================

print("\n" + "=" * 60)
print("  EGITIM BASLIYOR (v2)")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nEgitim tamamlandi!")


# =============================================================
# EĞİTİM GRAFİKLERİ
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('v2 Egitim Sureci Analizi', fontsize=16, fontweight='bold')

axes[0,0].plot(history.history['loss'], label='Egitim', linewidth=2, color='#378ADD')
axes[0,0].plot(history.history['val_loss'], label='Dogrulama', linewidth=2, color='#D85A30')
axes[0,0].set_title('Loss (BCE + Dice)')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(history.history['dice_coefficient'], label='Egitim', linewidth=2, color='#378ADD')
axes[0,1].plot(history.history['val_dice_coefficient'], label='Dogrulama', linewidth=2, color='#D85A30')
axes[0,1].set_title('Dice Coefficient')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Dice')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(history.history['iou_metric'], label='Egitim', linewidth=2, color='#378ADD')
axes[1,0].plot(history.history['val_iou_metric'], label='Dogrulama', linewidth=2, color='#D85A30')
axes[1,0].set_title('IoU')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('IoU')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(history.history['accuracy'], label='Egitim', linewidth=2, color='#378ADD')
axes[1,1].plot(history.history['val_accuracy'], label='Dogrulama', linewidth=2, color='#D85A30')
axes[1,1].set_title('Piksel Dogrulugu')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Accuracy')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v2_egitim_grafikleri.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================
# TEST DEĞERLENDİRME
# =============================================================

print("Test seti degerlendiriliyor...")
results = model.evaluate(test_ds, verbose=1)

print(f"\n{'='*50}")
print(f"  v2 TEST SONUCLARI")
print(f"{'='*50}")
print(f"  Loss (BCE + Dice):    {results[0]:.4f}")
print(f"  Dice Coefficient:     {results[1]:.4f}")
print(f"  IoU (Jaccard Index):  {results[2]:.4f}")
print(f"  Pixel Accuracy:       {results[3]:.4f}")
print(f"{'='*50}")


# =============================================================
# TAHMİN GÖRSELLEŞTİRME
# =============================================================

predictions = model.predict(X_test)
pred_masks = (predictions > 0.5).astype(np.float32)

fig, axes = plt.subplots(6, 4, figsize=(16, 24))
fig.suptitle('v2 Test Seti Tahmin Sonuclari', fontsize=18, fontweight='bold', y=1.01)

column_titles = ['Kolonoskopi', 'Ground Truth', 'Model Tahmini', 'Karsilastirma']
for j, title in enumerate(column_titles):
    axes[0, j].set_title(title, fontsize=13, fontweight='bold')

np.random.seed(123)
indices = np.random.choice(len(X_test), 6, replace=False)

for i, idx in enumerate(indices):
    axes[i, 0].imshow(X_test[idx])
    axes[i, 0].axis('off')

    axes[i, 1].imshow(Y_test[idx, :, :, 0], cmap='gray')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(pred_masks[idx, :, :, 0], cmap='gray')
    axes[i, 2].axis('off')

    gt = Y_test[idx, :, :, 0]
    pred = pred_masks[idx, :, :, 0]
    comparison = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    comparison[:, :, 1] = gt * pred              # Yesil: TP
    comparison[:, :, 0] = (1 - gt) * pred        # Kirmizi: FP
    comparison[:, :, 2] = gt * (1 - pred)        # Mavi: FN
    axes[i, 3].imshow(comparison)
    axes[i, 3].axis('off')

    inter = np.sum(gt * pred)
    dice_val = (2. * inter + 1e-6) / (np.sum(gt) + np.sum(pred) + 1e-6)
    axes[i, 3].text(5, 20, f'Dice: {dice_val:.4f}', color='white', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('v2_tahmin_sonuclari.png', dpi=150, bbox_inches='tight')
plt.show()

print("Yesil=TP | Kirmizi=FP | Mavi=FN")


# =============================================================
# DETAYLI METRİK ANALİZİ
# =============================================================

dice_scores = []
iou_scores = []

for i in range(len(X_test)):
    gt = Y_test[i, :, :, 0].flatten()
    pred = pred_masks[i, :, :, 0].flatten()
    inter = np.sum(gt * pred)
    dice = (2. * inter + 1e-6) / (np.sum(gt) + np.sum(pred) + 1e-6)
    dice_scores.append(dice)
    union = np.sum(gt) + np.sum(pred) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    iou_scores.append(iou)

dice_scores = np.array(dice_scores)
iou_scores = np.array(iou_scores)

print(f"\n{'='*55}")
print(f"  v2 DETAYLI TEST METRIKLERI")
print(f"{'='*55}")
print(f"  Dice  -> Ort: {dice_scores.mean():.4f} +/- {dice_scores.std():.4f}")
print(f"           Min: {dice_scores.min():.4f} | Max: {dice_scores.max():.4f}")
print(f"  IoU   -> Ort: {iou_scores.mean():.4f} +/- {iou_scores.std():.4f}")
print(f"           Min: {iou_scores.min():.4f} | Max: {iou_scores.max():.4f}")
print(f"\n  Dice > 0.8: {np.sum(dice_scores > 0.8)} / {len(dice_scores)}")
print(f"  Dice < 0.3: {np.sum(dice_scores < 0.3)} / {len(dice_scores)}")
print(f"{'='*55}")

# Skor dagilimi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('v2 Test Seti Skor Dagilimi', fontsize=14, fontweight='bold')

axes[0].hist(dice_scores, bins=25, color='#378ADD', edgecolor='white', alpha=0.8)
axes[0].axvline(dice_scores.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Ortalama: {dice_scores.mean():.4f}')
axes[0].set_title('Dice Score')
axes[0].set_xlabel('Dice')
axes[0].set_ylabel('Frekans')
axes[0].legend()

axes[1].hist(iou_scores, bins=25, color='#1D9E75', edgecolor='white', alpha=0.8)
axes[1].axvline(iou_scores.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Ortalama: {iou_scores.mean():.4f}')
axes[1].set_title('IoU Score')
axes[1].set_xlabel('IoU')
axes[1].set_ylabel('Frekans')
axes[1].legend()

plt.tight_layout()
plt.savefig('v2_skor_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()

# En iyi 3 vs En kotu 3
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
fig.suptitle('v2 En Iyi 3 vs En Kotu 3', fontsize=16, fontweight='bold')

best_idx = np.argsort(dice_scores)[-3:][::-1]
for i, idx in enumerate(best_idx):
    axes[0, i*2].imshow(X_test[idx])
    axes[0, i*2].set_title(f'Dice: {dice_scores[idx]:.3f}', color='green', fontsize=11)
    axes[0, i*2].axis('off')
    axes[0, i*2+1].imshow(pred_masks[idx, :, :, 0], cmap='gray')
    axes[0, i*2+1].set_title('Tahmin', fontsize=10)
    axes[0, i*2+1].axis('off')

worst_idx = np.argsort(dice_scores)[:3]
for i, idx in enumerate(worst_idx):
    axes[1, i*2].imshow(X_test[idx])
    axes[1, i*2].set_title(f'Dice: {dice_scores[idx]:.3f}', color='red', fontsize=11)
    axes[1, i*2].axis('off')
    axes[1, i*2+1].imshow(pred_masks[idx, :, :, 0], cmap='gray')
    axes[1, i*2+1].set_title('Tahmin', fontsize=10)
    axes[1, i*2+1].axis('off')

axes[0, 0].set_ylabel('EN IYI', fontsize=14, fontweight='bold', color='green')
axes[1, 0].set_ylabel('EN KOTU', fontsize=14, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('v2_en_iyi_en_kotu.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================
# FİNAL RAPOR
# =============================================================

best_epoch = np.argmax(history.history['val_dice_coefficient']) + 1
best_val_dice = max(history.history['val_dice_coefficient'])
total_epochs = len(history.history['loss'])

print(f"\n{'='*65}")
print(f"  FINAL PROJE RAPORU")
print(f"  U-Net ile Polip Segmentasyonu (Kvasir-SEG)")
print(f"{'='*65}")
print(f"  Model:             U-Net (31M parametre)")
print(f"  Dataset:           Kvasir-SEG (1000 goruntu)")
print(f"  Split:             700 / 150 / 150")
print(f"  Loss:              BCE + Dice Loss")
print(f"  Optimizer:         Adam + Cosine Decay (lr={INIT_LR})")
print(f"  En iyi epoch:      {best_epoch} (val_dice: {best_val_dice:.4f})")
print(f"  Test Dice:         {dice_scores.mean():.4f} +/- {dice_scores.std():.4f}")
print(f"  Test IoU:          {iou_scores.mean():.4f} +/- {iou_scores.std():.4f}")
print(f"  Pixel Accuracy:    {results[3]:.4f}")
print(f"{'='*65}")

# Model kaydet
model.save('unet_polyp_v2_final.keras')
print("Model 'unet_polyp_v2_final.keras' olarak kaydedildi.")
