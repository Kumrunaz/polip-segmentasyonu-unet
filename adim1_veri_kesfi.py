# =============================================================
# ADIM 1: KVASIR-SEG VERİ SETİ KEŞFİ
# U-Net ile Polip Segmentasyonu - Final Projesi
# =============================================================

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# ===== 1.1 Dataset Yolları =====
IMAGE_DIR = 'kvasir_seg/Kvasir-SEG/Kvasir-SEG/images/'
MASK_DIR = 'kvasir_seg/Kvasir-SEG/Kvasir-SEG/masks/'

image_paths = sorted(glob(IMAGE_DIR + '*.jpg') + glob(IMAGE_DIR + '*.png'))
mask_paths = sorted(glob(MASK_DIR + '*.jpg') + glob(MASK_DIR + '*.png'))

print(f"Toplam goruntu: {len(image_paths)}")
print(f"Toplam maske:   {len(mask_paths)}")

# Eslesme kontrolu
img_names = set([os.path.basename(p).split('.')[0] for p in image_paths])
mask_names = set([os.path.basename(p).split('.')[0] for p in mask_paths])
matched = img_names & mask_names
print(f"Eslesen cift: {len(matched)}")


# ===== 1.2 Ornek Gorsellestirme =====
fig, axes = plt.subplots(5, 3, figsize=(14, 20))
fig.suptitle('Kvasir-SEG Veri Seti Ornekleri', fontsize=16, fontweight='bold')

axes[0, 0].set_title('Kolonoskopi Goruntusu', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Ground Truth Maske', fontsize=12, fontweight='bold')
axes[0, 2].set_title('Overlay', fontsize=12, fontweight='bold')

np.random.seed(42)
indices = np.random.choice(len(image_paths), 5, replace=False)

for i, idx in enumerate(indices):
    img = cv2.imread(image_paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)

    axes[i, 0].imshow(img)
    axes[i, 0].set_ylabel(f'#{idx}', fontsize=11)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(mask, cmap='gray')
    axes[i, 1].axis('off')

    overlay = img.copy()
    mask_bool = mask > 127
    overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1].astype(int) + 100, 0, 255)
    axes[i, 2].imshow(overlay)
    axes[i, 2].axis('off')

    axes[i, 0].text(5, 25, f'{img.shape[1]}x{img.shape[0]}',
                     color='white', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('ornekler.png', dpi=150, bbox_inches='tight')
plt.show()


# ===== 1.3 Boyut Dagilimi Analizi =====
widths = []
heights = []

for p in tqdm(image_paths, desc='Boyutlar okunuyor'):
    img = cv2.imread(p)
    h, w = img.shape[:2]
    widths.append(w)
    heights.append(h)

widths = np.array(widths)
heights = np.array(heights)

print("\n" + "=" * 50)
print("GORUNTU BOYUT ISTATISTIKLERI")
print("=" * 50)
print(f"Genislik  -> Min: {widths.min()}, Max: {widths.max()}, Ort: {widths.mean():.0f}")
print(f"Yukseklik -> Min: {heights.min()}, Max: {heights.max()}, Ort: {heights.mean():.0f}")
print(f"Benzersiz boyutlar: {len(set(zip(widths, heights)))}")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Goruntu Boyut Dagilimi', fontsize=14, fontweight='bold')

axes[0].hist(widths, bins=30, color='#378ADD', edgecolor='white', alpha=0.8)
axes[0].set_title('Genislik Dagilimi')
axes[0].set_xlabel('Piksel')
axes[0].set_ylabel('Frekans')
axes[0].axvline(widths.mean(), color='red', linestyle='--', label=f'Ort: {widths.mean():.0f}')
axes[0].legend()

axes[1].hist(heights, bins=30, color='#1D9E75', edgecolor='white', alpha=0.8)
axes[1].set_title('Yukseklik Dagilimi')
axes[1].set_xlabel('Piksel')
axes[1].set_ylabel('Frekans')
axes[1].axvline(heights.mean(), color='red', linestyle='--', label=f'Ort: {heights.mean():.0f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('boyut_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()


# ===== 1.4 Polip/Arka Plan Oran Analizi =====
polyp_ratios = []

for p in tqdm(mask_paths, desc='Maske oranlari hesaplaniyor'):
    mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    total_pixels = mask.shape[0] * mask.shape[1]
    polyp_pixels = np.sum(mask > 127)
    ratio = polyp_pixels / total_pixels
    polyp_ratios.append(ratio)

polyp_ratios = np.array(polyp_ratios)

print("\n" + "=" * 50)
print("POLIP / ARKA PLAN ORAN ANALIZI")
print("=" * 50)
print(f"Ortalama polip orani: %{polyp_ratios.mean()*100:.1f}")
print(f"Minimum polip orani:  %{polyp_ratios.min()*100:.1f}")
print(f"Maksimum polip orani: %{polyp_ratios.max()*100:.1f}")
print(f"Medyan polip orani:   %{np.median(polyp_ratios)*100:.1f}")
print(f"\nCok kucuk polipler (<%5):  {np.sum(polyp_ratios < 0.05)} goruntu")
print(f"Orta polipler (%5-%30):    {np.sum((polyp_ratios >= 0.05) & (polyp_ratios < 0.30))} goruntu")
print(f"Buyuk polipler (>%30):     {np.sum(polyp_ratios >= 0.30)} goruntu")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Polip Alan Orani Analizi', fontsize=14, fontweight='bold')

axes[0].hist(polyp_ratios * 100, bins=30, color='#D85A30', edgecolor='white', alpha=0.8)
axes[0].axvline(polyp_ratios.mean() * 100, color='red', linestyle='--',
                label=f'Ortalama: %{polyp_ratios.mean()*100:.1f}')
axes[0].set_title('Polip Orani Dagilimi')
axes[0].set_xlabel('Polip Orani (%)')
axes[0].set_ylabel('Frekans')
axes[0].legend()

axes[1].boxplot(polyp_ratios * 100, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#D85A30', alpha=0.6),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Polip Orani Box Plot')
axes[1].set_ylabel('Polip Orani (%)')

plt.tight_layout()
plt.savefig('polip_oran_analizi.png', dpi=150, bbox_inches='tight')
plt.show()


# ===== 1.5 Piksel Deger Dagilimi =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Piksel Deger Dagilimi (50 Ornek)', fontsize=14, fontweight='bold')

sample_indices = np.random.choice(len(image_paths), 50, replace=False)
all_r, all_g, all_b, all_mask_vals = [], [], [], []

for idx in sample_indices:
    img = cv2.imread(image_paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
    all_r.extend(img[:,:,0].flatten()[::10])
    all_g.extend(img[:,:,1].flatten()[::10])
    all_b.extend(img[:,:,2].flatten()[::10])
    all_mask_vals.extend(mask.flatten()[::10])

axes[0].hist(all_r, bins=50, alpha=0.5, color='red', label='R', density=True)
axes[0].hist(all_g, bins=50, alpha=0.5, color='green', label='G', density=True)
axes[0].hist(all_b, bins=50, alpha=0.5, color='blue', label='B', density=True)
axes[0].set_title('RGB Kanal Dagilimi')
axes[0].set_xlabel('Piksel Degeri (0-255)')
axes[0].legend()

axes[1].hist(all_mask_vals, bins=50, color='gray', edgecolor='white', alpha=0.8)
axes[1].set_title('Maske Piksel Dagilimi')
axes[1].set_xlabel('Piksel Degeri (0-255)')
axes[1].set_ylabel('Frekans')

plt.tight_layout()
plt.savefig('piksel_dagilimi.png', dpi=150, bbox_inches='tight')
plt.show()

unique_vals = np.unique(all_mask_vals)
print(f"\nMaskelerdeki benzersiz deger sayisi: {len(unique_vals)}")


# ===== 1.6 Ozet =====
print("\n" + "=" * 60)
print("  KVASIR-SEG VERI SETI OZETI")
print("=" * 60)
print(f"  Toplam goruntu-maske cifti: {len(matched)}")
print(f"  Goruntu tipi:              RGB (3 kanal)")
print(f"  Boyut araligi:             {widths.min()}x{heights.min()} - {widths.max()}x{heights.max()}")
print(f"  Ortalama polip orani:      %{polyp_ratios.mean()*100:.1f}")
print(f"  Maske tipi:                {'Binary' if len(unique_vals) <= 5 else 'Threshold gerekli'}")
print("=" * 60)
