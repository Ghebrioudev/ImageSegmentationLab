"""
=============================================================
PROJET : Segmentation d'images - Scène 1
Module  : TAI & FD1
Auteur  : Aymen
Scène   : Chat / Ciel / Sol / Arbres
Méthode : Prétraitement (Filtre Gaussien + LAB) + K-Means (k=4)
=============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 0. CHARGEMENT DES IMAGES
# ============================================================
print("=" * 50)
print("SCÈNE 1 : Segmentation Chat/Ciel/Sol/Arbres")
print("=" * 50)

# Charger l'image originale (BGR -> RGB)
image_bgr = cv2.imread("Scene_1.png", cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Charger le masque Ground Truth et le redimensionner à la taille de l'image
gt_bgr = cv2.imread("GT1.png", cv2.IMREAD_COLOR)
gt_bgr = cv2.resize(gt_bgr, (image_rgb.shape[1], image_rgb.shape[0]))
gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

print(f"Image originale : {image_rgb.shape}")
print(f"Masque GT       : {gt_rgb.shape} (redimensionné)")

# ============================================================
# 1. PRÉTRAITEMENT (techniques vues en TP TAI)
# ============================================================
print("\n[Étape 1] Prétraitement...")

# 1.1 Filtre Gaussien pour réduire le bruit (vu en TP4)
image_filtre = cv2.GaussianBlur(image_rgb, (5, 5), sigmaX=1)

# 1.2 Conversion RGB -> LAB (meilleure séparation des couleurs, vu en TP3)
image_lab = cv2.cvtColor(image_filtre, cv2.COLOR_RGB2LAB)

# 1.3 Préparer les pixels pour le clustering
h, w, c = image_lab.shape
pixels = image_lab.reshape(-1, 3).astype(np.float64)

# 1.4 Normalisation Min-Max (vu en TP1 FD)
scaler = MinMaxScaler()
pixels_normalises = scaler.fit_transform(pixels)

print(f"  - Filtre Gaussien (5x5, sigma=1) appliqué")
print(f"  - Conversion RGB -> LAB effectuée")
print(f"  - Normalisation Min-Max effectuée")
print(f"  - Nombre de pixels à clusteriser : {len(pixels_normalises)}")

# ============================================================
# 2. SEGMENTATION PAR K-MEANS (k=4) (vu en TP2/3 FD + TP3 TAI)
# ============================================================
print("\n[Étape 2] Application K-Means (k=4)...")

kmeans = KMeans(
    n_clusters=4,
    init='k-means++',   # meilleure initialisation
    n_init=10,           # 10 initialisations pour stabilité
    max_iter=300,
    random_state=42
)
kmeans.fit(pixels_normalises)

labels = kmeans.labels_
centres = kmeans.cluster_centers_

print(f"  - K-Means convergé en {kmeans.n_iter_} itérations")
print(f"  - Inertie finale : {kmeans.inertia_:.2f}")

# Reconstruire l'image segmentée
image_segmentee_labels = labels.reshape(h, w)

# Attribuer une couleur représentative à chaque cluster
# (la couleur moyenne des pixels du cluster dans l'espace RGB)
image_segmentee = np.zeros((h, w, 3), dtype=np.uint8)
couleurs_clusters = []

for k in range(4):
    masque_cluster = (image_segmentee_labels == k)
    couleur_moy = image_rgb[masque_cluster].mean(axis=0).astype(np.uint8)
    couleurs_clusters.append(couleur_moy)
    image_segmentee[masque_cluster] = couleur_moy

print(f"  - Image segmentée reconstruite")

# ============================================================
# 3. IDENTIFICATION DES CLUSTERS
# ============================================================
print("\n[Étape 3] Identification des clusters...")

# Couleurs GT connues (analysées depuis GT1.png) en RGB
GT_COULEURS = {
    "chat":   np.array([239, 216, 146]),  # Jaune clair
    "arbres": np.array([166, 134, 154]),  # Violet/Rose
    "sol":    np.array([138, 160, 108]),  # Vert
    "ciel":   np.array([106, 136, 210]),  # Bleu
}

# Calculer la couleur moyenne GT pour chaque cluster prédit
# en cherchant quelle couleur GT domine dans la zone du cluster
labels_noms = {}
for k in range(4):
    masque_cluster = (image_segmentee_labels == k)
    pixels_gt_zone = gt_rgb[masque_cluster]  # pixels GT dans cette zone

    # Pour chaque couleur GT, compter combien de pixels GT sont proches
    scores = {}
    for nom, gt_c in GT_COULEURS.items():
        diff = np.abs(pixels_gt_zone.astype(int) - gt_c.astype(int))
        proches = (diff.max(axis=1) < 50).sum()
        scores[nom] = proches

    nom_proche = max(scores, key=scores.get)
    labels_noms[k] = nom_proche
    print(f"  - Cluster {k} -> {nom_proche} (couleur moy RGB: {couleurs_clusters[k]}, score: {scores[nom_proche]})")

# ============================================================
# 4. CALCUL DES MÉTRIQUES (IoU, Dice, Précision, Rappel)
# ============================================================
print("\n[Étape 4] Calcul des métriques...")

def calculer_metriques(pred_mask, gt_mask):
    """Calcule IoU, Dice, Précision et Rappel entre deux masques binaires."""
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()

    iou       = TP / (TP + FP + FN + 1e-8)
    dice      = 2 * TP / (2 * TP + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    rappel    = TP / (TP + FN + 1e-8)

    return {
        "IoU"      : round(float(iou), 4),
        "Dice"     : round(float(dice), 4),
        "Précision": round(float(precision), 4),
        "Rappel"   : round(float(rappel), 4),
    }

# Seuil de tolérance pour extraire le masque GT par couleur
TOLERANCE = 40

metriques_totales = {}

for k in range(4):
    nom = labels_noms[k]
    gt_couleur = GT_COULEURS[nom]

    # Masque prédit : pixels appartenant au cluster k
    masque_pred = (image_segmentee_labels == k)

    # Masque GT : pixels proches de la couleur GT du cluster
    diff = np.abs(gt_rgb.astype(int) - gt_couleur.astype(int))
    masque_gt = (diff.max(axis=2) < TOLERANCE)

    metriques = calculer_metriques(masque_pred, masque_gt)
    metriques_totales[nom] = metriques

    print(f"\n  [{nom.upper()}]")
    for m, v in metriques.items():
        print(f"    {m:12s}: {v:.4f}")

# Moyennes globales
print("\n  [MOYENNE GLOBALE]")
for m in ["IoU", "Dice", "Précision", "Rappel"]:
    vals = [metriques_totales[n][m] for n in metriques_totales]
    print(f"    {m:12s}: {np.mean(vals):.4f}")

# ============================================================
# 5. VISUALISATION
# ============================================================
print("\n[Étape 5] Affichage des résultats...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Scène 1 : Segmentation K-Means (k=4)\nChat / Ciel / Sol / Arbres",
             fontsize=14, fontweight='bold')

# Image originale
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Image Originale")
axes[0, 0].axis('off')

# Image filtrée
axes[0, 1].imshow(image_filtre)
axes[0, 1].set_title("Après Filtre Gaussien")
axes[0, 1].axis('off')

# Image segmentée
axes[0, 2].imshow(image_segmentee)
axes[0, 2].set_title("Segmentation K-Means (k=4)")
axes[0, 2].axis('off')

# Masque Ground Truth
axes[1, 0].imshow(gt_rgb)
axes[1, 0].set_title("Masque Ground Truth (GT1)")
axes[1, 0].axis('off')

# Comparaison côte à côte (segmenté vs GT)
comparaison = np.hstack([
    cv2.resize(image_segmentee, (gt_rgb.shape[1], gt_rgb.shape[0])),
    gt_rgb
])
axes[1, 1].imshow(comparaison)
axes[1, 1].set_title("Segmenté (gauche) vs GT (droite)")
axes[1, 1].axis('off')

# Tableau des métriques
axes[1, 2].axis('off')
noms_clusters = list(metriques_totales.keys())
colonnes = ["IoU", "Dice", "Précision", "Rappel"]
donnees_table = [[metriques_totales[n][m] for m in colonnes] for n in noms_clusters]

table = axes[1, 2].table(
    cellText=donnees_table,
    rowLabels=[n.capitalize() for n in noms_clusters],
    colLabels=colonnes,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
axes[1, 2].set_title("Métriques par Cluster", fontweight='bold')

plt.tight_layout()
plt.savefig("resultats_scene1.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Résultats sauvegardés dans 'resultats_scene1.png'")
print("=" * 50)
print("FIN DU SCRIPT - SCÈNE 1")
print("=" * 50)