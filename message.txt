"""
=============================================================
  Projet BioInfo - Segmentation d'Images
  Module : TAI & FD1
  Scène 4 : Extraction de la personne debout
  Méthode : Combinaison image couleur + image supplémentaire
            + K-Means clustering multi-canal
  Auteur   : [Votre Nom]
  Année    : 2025/2026
=============================================================

  PIPELINE :
  ┌─────────────────────────────────────────────────────┐
  │  1. Chargement des images (couleur + profondeur + GT)│
  │  2. Prétraitement (débruitage + normalisation)       │
  │  3. Construction features multi-canal [R,G,B,D]     │
  │  4. Segmentation K-Means (K=3 clusters)             │
  │  5. Identification du cluster "personne"            │
  │  6. Post-traitement morphologique                   │
  │  7. Calcul des métriques (IoU, F1, Accuracy...)     │
  │  8. Visualisation et sauvegarde                     │
  └─────────────────────────────────────────────────────┘

  UTILISATION :
      python mapartie.py Scene_4_RGB_1.png Scene_4_D_2.png GT4.png
=============================================================
"""

# ── Bibliothèques utilisées ────────────────────────────────
import cv2                         # Traitement d'images (OpenCV)
import numpy as np                 # Calcul matriciel
import matplotlib.pyplot as plt    # Visualisation des résultats
import os                          # Gestion des fichiers/dossiers
import sys                         # Arguments en ligne de commande

# Algorithme de clustering K-Means (Fouille de Données)
from sklearn.cluster import KMeans

# Métriques d'évaluation de segmentation
from sklearn.metrics import (
    accuracy_score,    # Précision globale pixel
    f1_score,          # F1-Score (Dice)
    jaccard_score,     # IoU (Intersection over Union)
    confusion_matrix   # Matrice de confusion TP/TN/FP/FN
)


# ══════════════════════════════════════════════════════════
#  ÉTAPE 1 : CHARGEMENT DES IMAGES
# ══════════════════════════════════════════════════════════

def load_images(color_path, extra_path, gt_path=None):
    """
    Charge les trois images nécessaires au pipeline :

    Paramètres :
    ------------
    color_path : str  → chemin vers l'image couleur RGB (Scene_4_RGB_1.png)
    extra_path : str  → chemin vers l'image supplémentaire/profondeur (Scene_4_D_2.png)
    gt_path    : str  → chemin vers le masque Ground Truth (GT4.png), optionnel

    Retourne :
    ----------
    img_color : ndarray (H, W, 3)  → image couleur BGR
    img_extra : ndarray (H, W, 3)  → image supplémentaire
    gt_mask   : ndarray (H, W)     → masque binaire GT (0=fond, 1=personne)
    """

    # Charger l'image couleur en BGR (format par défaut d'OpenCV)
    img_color = cv2.imread(color_path)

    # Charger l'image supplémentaire (carte de profondeur ou image IR)
    img_extra = cv2.imread(extra_path)

    # Vérifier que les fichiers existent et sont lisibles
    if img_color is None:
        raise FileNotFoundError(f"Image couleur introuvable : {color_path}")
    if img_extra is None:
        raise FileNotFoundError(f"Image supplémentaire introuvable : {extra_path}")

    # Si les deux images n'ont pas la même taille → redimensionner l'image extra
    # (important pour pouvoir fusionner pixel à pixel)
    if img_color.shape[:2] != img_extra.shape[:2]:
        img_extra = cv2.resize(
            img_extra,
            (img_color.shape[1], img_color.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        print("[INFO] Image supplémentaire redimensionnée.")

    # Chargement optionnel du masque Ground Truth (pour évaluation)
    gt_mask = None
    if gt_path and os.path.exists(gt_path):
        # Charger en niveaux de gris (1 seul canal)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            # Redimensionner si nécessaire
            gt_mask = cv2.resize(
                gt_mask,
                (img_color.shape[1], img_color.shape[0]),
                interpolation=cv2.INTER_NEAREST  # NEAREST pour masque binaire
            )
            # Binariser : pixels > 127 → 1 (personne), sinon → 0 (fond)
            _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

    print(f"[OK] Images chargées. Taille : {img_color.shape[:2]}")
    return img_color, img_extra, gt_mask


# ══════════════════════════════════════════════════════════
#  ÉTAPE 2 : PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════

def preprocess(img_color, img_extra):
    """
    Prépare les images avant la segmentation :
    - Réduction du bruit par filtre gaussien
    - Conversion de l'image extra en niveaux de gris

    Paramètres :
    ------------
    img_color : ndarray (H, W, 3)  → image couleur BGR brute
    img_extra : ndarray            → image supplémentaire brute

    Retourne :
    ----------
    img_color_blur : image couleur débruitée
    img_extra_blur : image supplémentaire en niveaux de gris débruitée
    """

    # ── Filtre Gaussien sur l'image couleur ──────────────────
    # Noyau 5x5, sigma calculé automatiquement
    # But : atténuer le bruit haute fréquence avant clustering
    img_color_blur = cv2.GaussianBlur(img_color, (5, 5), 0)

    # ── Conversion image extra en niveaux de gris ────────────
    # K-Means a besoin d'un seul canal pour l'image depth
    if len(img_extra.shape) == 3:
        # Image couleur → convertir en grayscale
        img_extra_gray = cv2.cvtColor(img_extra, cv2.COLOR_BGR2GRAY)
    else:
        # Déjà en niveaux de gris
        img_extra_gray = img_extra.copy()

    # ── Filtre Gaussien sur l'image profondeur ───────────────
    img_extra_blur = cv2.GaussianBlur(img_extra_gray, (5, 5), 0)

    print("[OK] Prétraitement terminé.")
    return img_color_blur, img_extra_blur


# ══════════════════════════════════════════════════════════
#  ÉTAPE 3 : CONSTRUCTION DES FEATURES MULTI-CANAL
# ══════════════════════════════════════════════════════════

def build_feature_matrix(img_color, img_extra_gray):
    """
    Fusionne l'image couleur et l'image profondeur en une seule
    matrice de features pour le clustering.

    Chaque pixel est représenté par un vecteur à 4 dimensions :
        pixel(i,j) = [ R, G, B, D ]
    où D est la valeur de profondeur normalisée.

    Paramètres :
    ------------
    img_color      : ndarray (H, W, 3)  → image couleur BGR prétraitée
    img_extra_gray : ndarray (H, W)     → image profondeur grayscale prétraitée

    Retourne :
    ----------
    features : ndarray (H*W, 4)  → matrice de features aplatie
    h, w     : int               → dimensions de l'image
    """

    h, w = img_color.shape[:2]  # Hauteur et largeur

    # ── Conversion BGR → RGB (pour cohérence des canaux) ─────
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # ── Normalisation dans [0, 1] ────────────────────────────
    # Permet au K-Means de traiter tous les canaux à la même échelle
    rgb_norm   = img_rgb.astype(np.float32) / 255.0       # (H, W, 3)
    extra_norm = img_extra_gray.astype(np.float32) / 255.0 # (H, W)

    # ── Aplatissement : image 2D → vecteur 1D ───────────────
    # Chaque pixel devient une ligne dans la matrice
    rgb_flat   = rgb_norm.reshape(-1, 3)    # (H*W, 3)
    extra_flat = extra_norm.reshape(-1, 1)  # (H*W, 1)

    # ── Concaténation : [R, G, B] + [D] = [R, G, B, D] ──────
    # La matrice finale a H*W lignes et 4 colonnes
    features = np.concatenate([rgb_flat, extra_flat], axis=1)  # (H*W, 4)

    print(f"[OK] Matrice de features construite : {features.shape}")
    return features, h, w


# ══════════════════════════════════════════════════════════
#  ÉTAPE 4 : SEGMENTATION PAR K-MEANS
# ══════════════════════════════════════════════════════════

def kmeans_segmentation(features, h, w, n_clusters=3, seed=42):
    """
    Applique l'algorithme K-Means sur la matrice de features.

    K-Means partitionne les pixels en K groupes (clusters) en
    minimisant la distance intra-cluster :
        J = Σ_k Σ_{x ∈ Ck} ||x - μk||²

    Paramètres :
    ------------
    features   : ndarray (H*W, 4)  → matrice de features
    h, w       : int               → dimensions de l'image
    n_clusters : int               → nombre de clusters (K=3 ici)
    seed       : int               → graine aléatoire pour reproductibilité

    Retourne :
    ----------
    labels_2d : ndarray (H, W)     → label de cluster pour chaque pixel
    centers   : ndarray (K, 4)     → centroïdes des K clusters
    """

    print(f"[...] K-Means avec {n_clusters} clusters en cours...")

    # Initialisation K-Means++ : meilleure convergence que random
    # n_init=10 : répéter 10 fois, garder le meilleur résultat
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        init='k-means++'  # Initialisation intelligente
    )

    # Assigner chaque pixel (ligne de features) à un cluster
    labels = kmeans.fit_predict(features)  # (H*W,)

    # Remettre en forme 2D pour correspondre à l'image
    labels_2d = labels.reshape(h, w)       # (H, W)

    print("[OK] K-Means terminé.")
    return labels_2d, kmeans.cluster_centers_


# ══════════════════════════════════════════════════════════
#  ÉTAPE 5 : IDENTIFICATION DU CLUSTER "PERSONNE"
# ══════════════════════════════════════════════════════════

def identify_person_cluster(labels_2d, cluster_centers, img_extra_gray):
    """
    Détermine quel cluster correspond à la personne.

    Deux stratégies combinées :
    ───────────────────────────
    Stratégie 1 (poids 70%) : Position centrale
        → La personne est au centre de l'image
        → On compte quel cluster est dominant dans la zone centrale

    Stratégie 2 (poids 30%) : Valeur de profondeur
        → Dans l'image depth, la personne apparaît plus sombre
        → On cherche le cluster avec la valeur D la plus basse

    Paramètres :
    ------------
    labels_2d       : ndarray (H, W)  → labels K-Means
    cluster_centers : ndarray (K, 4)  → centroïdes des clusters
    img_extra_gray  : ndarray (H, W)  → image profondeur (non utilisée directement)

    Retourne :
    ----------
    person_cluster : int → index du cluster correspondant à la personne
    """

    h, w = labels_2d.shape
    n_clusters = cluster_centers.shape[0]

    # ── Stratégie 1 : Cluster dominant au CENTRE de l'image ──
    # On prend la zone centrale : de H/4 à 3H/4 et W/4 à 3W/4
    cx1, cx2 = w // 4, 3 * w // 4
    cy1, cy2 = h // 4, 3 * h // 4
    center_region = labels_2d[cy1:cy2, cx1:cx2]  # Zone centrale

    # Compter le nombre de pixels de chaque cluster dans cette zone
    center_counts = np.bincount(center_region.flatten(), minlength=n_clusters)

    # Normaliser pour obtenir un score entre 0 et 1
    center_score = center_counts / center_counts.sum()

    # ── Stratégie 2 : Cluster avec valeur D la plus BASSE ────
    # La personne est plus sombre dans l'image depth → valeur faible
    extra_values = cluster_centers[:, 3]  # Canal D des centroïdes

    # Inverser : petit D → grand score (on veut le plus sombre)
    depth_score = 1.0 - (extra_values / (extra_values.max() + 1e-8))

    # ── Score combiné ─────────────────────────────────────────
    # 70% basé sur la position centrale + 30% basé sur la profondeur
    combined = 0.7 * center_score + 0.3 * depth_score

    # Le cluster avec le score le plus élevé = personne
    person_cluster = int(np.argmax(combined))

    # Affichage des scores pour débogage
    print(f"[OK] Cluster personne identifié : {person_cluster}")
    print(f"     Score centre : {center_score.round(3)}")
    print(f"     Score depth  : {depth_score.round(3)}")
    print(f"     Score final  : {combined.round(3)}")

    return person_cluster


# ══════════════════════════════════════════════════════════
#  ÉTAPE 6 : POST-TRAITEMENT MORPHOLOGIQUE
# ══════════════════════════════════════════════════════════

def postprocess_mask(labels_2d, person_cluster):
    """
    Affine le masque binaire par des opérations morphologiques.

    Séquence :
    ──────────
    1. Extraction du masque binaire (0/1)
    2. Ouverture morphologique → supprime les petits artefacts isolés
    3. Fermeture morphologique → remplit les trous dans la silhouette
    4. Composante connexe la plus grande → conserve uniquement la personne

    Paramètres :
    ------------
    labels_2d      : ndarray (H, W)  → labels K-Means
    person_cluster : int             → index du cluster personne

    Retourne :
    ----------
    final_mask : ndarray (H, W) binaire → masque final de la personne
    """

    # ── Masque binaire brut ───────────────────────────────────
    # 1 = pixels appartenant au cluster personne, 0 = reste
    binary_mask = (labels_2d == person_cluster).astype(np.uint8)

    # ── Élément structurant elliptique 7x7 ───────────────────
    # Forme elliptique pour respecter les contours arrondis du corps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ── Ouverture morphologique (érosion puis dilatation) ─────
    # Supprime les petits groupes de pixels isolés (bruit)
    mask_open = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # ── Fermeture morphologique (dilatation puis érosion) ─────
    # Remplit les trous à l'intérieur de la silhouette
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    # ── Sélection de la plus grande composante connexe ────────
    # Élimine les régions parasites restantes en ne gardant que
    # la région la plus grande (= la personne principale)
    num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask_close)

    if num_labels > 1:
        # stats[0] = fond (ignoré), stats[1:] = composantes réelles
        # CC_STAT_AREA = superficie en pixels de chaque composante
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (comp_labels == largest).astype(np.uint8)
    else:
        # Cas rare : une seule composante → garder telle quelle
        final_mask = mask_close

    print("[OK] Post-traitement terminé.")
    return final_mask


# ══════════════════════════════════════════════════════════
#  ÉTAPE 7 : CALCUL DES MÉTRIQUES D'ÉVALUATION
# ══════════════════════════════════════════════════════════

def compute_metrics(pred_mask, gt_mask):
    """
    Compare le masque prédit au masque Ground Truth et calcule
    les métriques standards de segmentation binaire.

    Métriques calculées :
    ─────────────────────
    • Accuracy  = (TP+TN) / (TP+TN+FP+FN)   → % pixels bien classés
    • IoU       = TP / (TP+FP+FN)            → chevauchement prédit/GT
    • F1-Score  = 2*TP / (2*TP+FP+FN)        → équilibre précision/rappel
    • Précision = TP / (TP+FP)               → fiabilité des détections
    • Rappel    = TP / (TP+FN)               → complétude des détections

    où : TP=Vrai Positif, TN=Vrai Négatif, FP=Faux Positif, FN=Faux Négatif

    Paramètres :
    ------------
    pred_mask : ndarray (H, W) binaire → masque prédit par notre algorithme
    gt_mask   : ndarray (H, W) binaire → masque Ground Truth de référence

    Retourne :
    ----------
    metrics : dict → dictionnaire des métriques calculées
    """

    # Aplatir en vecteurs 1D pour les fonctions sklearn
    pred_flat = pred_mask.flatten()
    gt_flat   = gt_mask.flatten()

    # ── Calcul des métriques ──────────────────────────────────
    accuracy = accuracy_score(gt_flat, pred_flat)
    iou      = jaccard_score(gt_flat, pred_flat, zero_division=0)
    f1       = f1_score(gt_flat, pred_flat, zero_division=0)

    # Matrice de confusion pour précision et rappel manuels
    cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ── Regrouper dans un dictionnaire ───────────────────────
    metrics = {
        "Accuracy"  : round(accuracy,  4),
        "IoU"       : round(iou,       4),
        "F1-Score"  : round(f1,        4),
        "Précision" : round(precision, 4),
        "Rappel"    : round(recall,    4),
    }

    # ── Affichage formaté ─────────────────────────────────────
    print("\n══════════════ MÉTRIQUES ══════════════")
    for k, v in metrics.items():
        # Barre visuelle proportionnelle à la valeur
        barre = "█" * int(v * 20)
        print(f"  {k:<12} : {v:.4f}  {barre}")
    print("═══════════════════════════════════════\n")

    return metrics


# ══════════════════════════════════════════════════════════
#  ÉTAPE 8 : VISUALISATION DES RÉSULTATS
# ══════════════════════════════════════════════════════════

def visualize_results(img_color, img_extra_gray, labels_2d,
                      final_mask, gt_mask=None, save_path=None):
    """
    Affiche côte à côte toutes les étapes du pipeline :
    image couleur | image depth | clusters | masque prédit | ground truth

    Paramètres :
    ------------
    img_color      : image couleur originale (BGR)
    img_extra_gray : image profondeur (grayscale)
    labels_2d      : carte des clusters K-Means
    final_mask     : masque binaire final prédit
    gt_mask        : masque Ground Truth (optionnel)
    save_path      : chemin pour sauvegarder la figure (optionnel)
    """

    # Convertir BGR → RGB pour affichage correct avec matplotlib
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # Nombre de sous-figures selon disponibilité du GT
    n_cols = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # ── Colonne 1 : Image couleur originale ──────────────────
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image Couleur", fontsize=10)
    axes[0].axis("off")

    # ── Colonne 2 : Image profondeur ─────────────────────────
    axes[1].imshow(img_extra_gray, cmap="gray")
    axes[1].set_title("Image Supplémentaire", fontsize=10)
    axes[1].axis("off")

    # ── Colonne 3 : Carte des clusters K-Means ───────────────
    axes[2].imshow(labels_2d, cmap="tab10")
    axes[2].set_title("K-Means Clusters", fontsize=10)
    axes[2].axis("off")

    # ── Colonne 4 : Masque binaire prédit ────────────────────
    axes[3].imshow(final_mask, cmap="gray")
    axes[3].set_title("Masque Prédit", fontsize=10)
    axes[3].axis("off")

    # ── Colonne 5 : Ground Truth (si disponible) ─────────────
    if gt_mask is not None:
        axes[4].imshow(gt_mask, cmap="gray")
        axes[4].set_title("Ground Truth", fontsize=10)
        axes[4].axis("off")

    # Titre principal de la figure
    plt.suptitle("Scène 4 – Extraction de la Personne",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ── Sauvegarde de la figure ───────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Figure sauvegardée : {save_path}")

    plt.show()


# ══════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL : ENCHAÎNEMENT DE TOUTES LES ÉTAPES
# ══════════════════════════════════════════════════════════

def run_pipeline(color_path, extra_path, gt_path=None,
                 n_clusters=3, output_dir="."):
    """
    Exécute le pipeline complet de segmentation de la Scène 4.

    Paramètres :
    ------------
    color_path  : str  → chemin image couleur
    extra_path  : str  → chemin image supplémentaire (profondeur)
    gt_path     : str  → chemin masque Ground Truth (optionnel)
    n_clusters  : int  → nombre de clusters K-Means (défaut=3)
    output_dir  : str  → dossier de sauvegarde des résultats

    Retourne :
    ----------
    final_mask : ndarray → masque binaire prédit
    metrics    : dict    → métriques d'évaluation (None si pas de GT)
    """

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("  PIPELINE SCÈNE 4 - DÉMARRAGE")
    print("="*50)

    # ── Étape 1 : Chargement ─────────────────────────────────
    img_color, img_extra, gt_mask = load_images(color_path, extra_path, gt_path)

    # ── Étape 2 : Prétraitement ──────────────────────────────
    img_color_p, img_extra_p = preprocess(img_color, img_extra)

    # ── Étape 3 : Construction features [R, G, B, D] ─────────
    features, h, w = build_feature_matrix(img_color_p, img_extra_p)

    # ── Étape 4 : Segmentation K-Means ──────────────────────
    labels_2d, centers = kmeans_segmentation(features, h, w, n_clusters)

    # ── Étape 5 : Identification cluster personne ────────────
    person_cluster = identify_person_cluster(labels_2d, centers, img_extra_p)

    # ── Étape 6 : Post-traitement morphologique ──────────────
    final_mask = postprocess_mask(labels_2d, person_cluster)

    # ── Étape 7 : Calcul des métriques (si GT disponible) ────
    metrics = None
    if gt_mask is not None:
        metrics = compute_metrics(final_mask, gt_mask)
    else:
        print("[INFO] Pas de Ground Truth → métriques non calculées.")

    # ── Étape 8 : Visualisation et sauvegarde ────────────────
    save_path = os.path.join(output_dir, "scene4_result.png")
    visualize_results(img_color, img_extra_p, labels_2d,
                      final_mask, gt_mask, save_path)

    # Sauvegarder le masque prédit en image PNG (0 ou 255)
    mask_save = (final_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "scene4_mask_predicted.png"), mask_save)
    print("[OK] Masque prédit sauvegardé.")

    print("\n" + "="*50)
    print("  PIPELINE TERMINÉ")
    print(f"  Résultats dans : {output_dir}/")
    print("="*50 + "\n")

    return final_mask, metrics


# ══════════════════════════════════════════════════════════
#  POINT D'ENTRÉE DU PROGRAMME
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Lancement depuis le terminal.

    Usage :
        python mapartie.py <image_couleur> <image_extra> [masque_gt]

    Exemple :
        python mapartie.py Scene_4_RGB_1.png Scene_4_D_2.png GT4.png
    """

    # Vérifier que les arguments obligatoires sont fournis
    if len(sys.argv) < 3:
        print("=" * 55)
        print("  USAGE :")
        print("  python mapartie.py <couleur> <extra> [gt]")
        print("")
        print("  EXEMPLE :")
        print("  python mapartie.py Scene_4_RGB_1.png Scene_4_D_2.png GT4.png")
        print("=" * 55)
        sys.exit(1)

    # Récupérer les arguments de la ligne de commande
    color_path = sys.argv[1]               # Image couleur (obligatoire)
    extra_path = sys.argv[2]               # Image profondeur (obligatoire)
    gt_path    = sys.argv[3] if len(sys.argv) >= 4 else None  # GT (optionnel)

    # Lancer le pipeline complet
    run_pipeline(
        color_path  = color_path,
        extra_path  = extra_path,
        gt_path     = gt_path,
        n_clusters  = 3,                   # K=3 clusters
        output_dir  = "resultats_scene4"   # Dossier de sortie
    )