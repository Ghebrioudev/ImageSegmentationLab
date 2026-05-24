import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1) Chargement image
# ==========================================
image = cv2.imread("Scene_3.png")
gt_color = cv2.imread("GT3.png")

if image is None or gt_color is None:
    print("Image ou GT non trouvé")
    exit()

# BGR -> RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gt_rgb = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB)

# ==========================================
# 2) Prétraitement
# ==========================================
blur = cv2.GaussianBlur(image_rgb, (5, 5), 1)

# Conversion LAB
lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)

# ==========================================
# 3) K-means
# ==========================================
pixels = lab.reshape((-1, 3))
pixels = np.float32(pixels)

K = 3

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    20,
    1.0
)

_, labels, centers = cv2.kmeans(
    pixels,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

labels = labels.flatten()
segmented = labels.reshape(image.shape[:2])

# ==========================================
# 4) Choisir cluster route
# ==========================================
L_channel = centers[:, 0]
road_cluster = np.argmax(L_channel)

mask = np.zeros(segmented.shape, dtype=np.uint8)
mask[segmented == road_cluster] = 255

# ==========================================
# 5) Post-traitement morphologique
# ==========================================
kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# ==========================================
# 6) Adapter GT à la taille de l'image
# ==========================================
gt_rgb = cv2.resize(
    gt_rgb,
    (mask.shape[1], mask.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# 7) Extraire routes rouges du GT
# ==========================================
gt_bin = np.zeros(mask.shape, dtype=np.uint8)

red = gt_rgb[:, :, 0]
green = gt_rgb[:, :, 1]
blue = gt_rgb[:, :, 2]

gt_bin[(red > 80) & (green < 60) & (blue < 60)] = 255

# ==========================================
# 8) Métriques
# ==========================================
pred = mask > 0
truth = gt_bin > 0

TP = np.logical_and(pred, truth).sum()
FP = np.logical_and(pred, np.logical_not(truth)).sum()
FN = np.logical_and(np.logical_not(pred), truth).sum()

iou = TP / (TP + FP + FN + 1e-8)
dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)

print("===== RESULTATS SCENE 3 =====")
print("IoU       :", round(iou, 4))
print("Dice      :", round(dice, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))

# ==========================================
# 9) Affichage
# ==========================================
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Image originale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(segmented, cmap="nipy_spectral")
plt.title("Clusters K-means")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(mask, cmap="gray")
plt.title("Route segmentée")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(gt_bin, cmap="gray")
plt.title("Ground truth")
plt.axis("off")

plt.tight_layout()
plt.show()