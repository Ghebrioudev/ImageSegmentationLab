"""
=============================================================
PROJET : Segmentation d'images - Scènes Multiples Unifiées
=============================================================

PIPELINE UNIFIÉ :
  Scène 1 : Chat / Ciel / Sol / Arbres (K-Means k=4, RGB→LAB)
  Scène 2 : Disque Rétinien Lumineux (Otsu + morphologie)
  Scène 3 : Routes Aériennes (K-Means k=3, RGB→LAB)
  Scène 4 : Extraction de Personne (K-Means k=3, Multi-canal [R,G,B,D])

UTILISATION :
  python image_segmentation_unified.py [scene_number]

  Exemples :
    python image_segmentation_unified.py 1        # Exécuter scène 1 uniquement
    python image_segmentation_unified.py all      # Exécuter toutes les scènes
    python image_segmentation_unified.py 1 2 3 4  # Exécuter scènes spécifiques
=============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix
from skimage import exposure


# ============================================================
# PARTIE 0 : MODULE DE MÉTRIQUES PARTAGÉ
# ============================================================

class MetricsComputer:
    """Module unifié pour le calcul des métriques de segmentation."""

    @staticmethod
    def compute_metrics_basic(pred_mask, gt_mask):
        """
        Calcule IoU, Dice, Précision et Rappel entre deux masques binaires.
        (Utilisé par Scènes 1, 2, 3)
        """
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)

        TP = np.logical_and(pred, gt).sum()
        FP = np.logical_and(pred, ~gt).sum()
        FN = np.logical_and(~pred, gt).sum()
        TN = np.logical_and(~pred, ~gt).sum()

        iou = TP / (TP + FP + FN + 1e-8)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        return {
            "IoU": round(float(iou), 4),
            "Dice": round(float(dice), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
        }

    @staticmethod
    def compute_metrics_extended(pred_mask, gt_mask):
        """
        Calcule Accuracy, IoU, F1-Score, Précision et Rappel.
        (Utilisé par Scène 4)
        """
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        accuracy = accuracy_score(gt_flat, pred_flat)
        iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
        f1 = f1_score(gt_flat, pred_flat, zero_division=0)

        cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "Accuracy": round(accuracy, 4),
            "IoU": round(iou, 4),
            "F1-Score": round(f1, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
        }

    @staticmethod
    def print_metrics(metrics, scene_name, cluster_name=None):
        """Affiche les métriques de manière formatée."""
        print(f"\n{'='*50}")
        if cluster_name:
            print(f"  [{cluster_name.upper()}] - {scene_name}")
        else:
            print(f"  {scene_name.upper()}")
        print(f"{'='*50}")
        for k, v in metrics.items():
            barre = "█" * int(v * 20) if v > 0 else ""
            print(f"  {k:<15}: {v:.4f}  {barre}")
        print()


# ============================================================
# SCÈNE 1 : Chat / Ciel / Sol / Arbres (K-Means k=4)
# ============================================================

class Scene1:
    """Segmentation Chat/Ciel/Sol/Arbres avec K-Means (k=4)."""

    def __init__(self, image_path, gt_path):
        self.image_path = image_path
        self.gt_path = gt_path
        self.metrics_computer = MetricsComputer()

        # Couleurs GT connues (RGB)
        self.GT_COULEURS = {
            "chat": np.array([239, 216, 146]),
            "arbres": np.array([166, 134, 154]),
            "sol": np.array([138, 160, 108]),
            "ciel": np.array([106, 136, 210]),
        }
        self.TOLERANCE = 40

    def load_images(self):
        """Charge les images et les redimensionne si nécessaire."""
        image_bgr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        gt_bgr = cv2.imread(self.gt_path, cv2.IMREAD_COLOR)

        if image_bgr is None or gt_bgr is None:
            raise FileNotFoundError(f"Impossible de charger les images Scene 1")

        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_bgr = cv2.resize(gt_bgr, (self.image_rgb.shape[1], self.image_rgb.shape[0]))
        self.gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

        print(f"[Scene 1] Image: {self.image_rgb.shape}, GT: {self.gt_rgb.shape}")

    def preprocess(self):
        """Prétraitement: Filtre Gaussien + RGB→LAB + Normalisation."""
        image_filtre = cv2.GaussianBlur(self.image_rgb, (5, 5), sigmaX=1)
        image_lab = cv2.cvtColor(image_filtre, cv2.COLOR_RGB2LAB)

        h, w, c = image_lab.shape
        pixels = image_lab.reshape(-1, 3).astype(np.float64)

        scaler = MinMaxScaler()
        pixels_normalises = scaler.fit_transform(pixels)

        self.image_filtre = image_filtre
        self.pixels_normalises = pixels_normalises
        self.h, self.w = h, w
        print("[Scene 1] Prétraitement terminé")

    def segment_kmeans(self, k=4):
        """Segmentation par K-Means avec k clusters."""
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(self.pixels_normalises)

        labels = kmeans.labels_
        self.image_segmentee_labels = labels.reshape(self.h, self.w)

        # Reconstruire l'image segmentée
        self.image_segmentee = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.couleurs_clusters = []

        for cluster_id in range(k):
            masque = (self.image_segmentee_labels == cluster_id)
            couleur_moy = self.image_rgb[masque].mean(axis=0).astype(np.uint8)
            self.couleurs_clusters.append(couleur_moy)
            self.image_segmentee[masque] = couleur_moy

        print(f"[Scene 1] K-Means (k={k}) convergé en {kmeans.n_iter_} itérations")

    def identify_clusters(self):
        """Identifie quel cluster correspond à quel objet."""
        self.labels_noms = {}

        for k in range(len(self.couleurs_clusters)):
            masque_cluster = (self.image_segmentee_labels == k)
            pixels_gt_zone = self.gt_rgb[masque_cluster]

            scores = {}
            for nom, gt_c in self.GT_COULEURS.items():
                diff = np.abs(pixels_gt_zone.astype(int) - gt_c.astype(int))
                proches = (diff.max(axis=1) < 50).sum()
                scores[nom] = proches

            nom_proche = max(scores, key=scores.get)
            self.labels_noms[k] = nom_proche
            print(f"  Cluster {k} → {nom_proche}")

    def compute_metrics(self):
        """Calcule les métriques pour chaque cluster."""
        self.metriques_totales = {}

        for k in range(len(self.couleurs_clusters)):
            nom = self.labels_noms[k]
            gt_couleur = self.GT_COULEURS[nom]

            masque_pred = (self.image_segmentee_labels == k)
            diff = np.abs(self.gt_rgb.astype(int) - gt_couleur.astype(int))
            masque_gt = (diff.max(axis=2) < self.TOLERANCE)

            metriques = self.metrics_computer.compute_metrics_basic(masque_pred, masque_gt)
            self.metriques_totales[nom] = metriques

            self.metrics_computer.print_metrics(metriques, "Scène 1", nom)

        # Moyennes globales
        print("[Scene 1] MOYENNES GLOBALES")
        for m in ["IoU", "Dice", "Precision", "Recall"]:
            vals = [self.metriques_totales[n][m] for n in self.metriques_totales]
            print(f"  {m}: {np.mean(vals):.4f}")

    def visualize(self, output_dir="."):
        """Crée et affiche les résultats."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        fig.suptitle("Scène 1 : Segmentation K-Means (k=4)\nChat / Ciel / Sol / Arbres",
                     fontsize=14, fontweight='bold', y=0.98)

        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title("Image Originale", fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.image_filtre)
        axes[0, 1].set_title("Après Filtre Gaussien", fontsize=11)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(self.image_segmentee)
        axes[0, 2].set_title("Segmentation K-Means (k=4)", fontsize=11)
        axes[0, 2].axis('off')

        axes[1, 0].imshow(self.gt_rgb)
        axes[1, 0].set_title("Masque Ground Truth", fontsize=11)
        axes[1, 0].axis('off')

        comparaison = np.hstack([
            cv2.resize(self.image_segmentee, (self.gt_rgb.shape[1], self.gt_rgb.shape[0])),
            self.gt_rgb
        ])
        axes[1, 1].imshow(comparaison)
        axes[1, 1].set_title("Segmenté (gauche) vs GT (droite)", fontsize=11)
        axes[1, 1].axis('off')

        axes[1, 2].axis('off')
        noms_clusters = list(self.metriques_totales.keys())
        colonnes = ["IoU", "Dice", "Precision", "Recall"]
        donnees_table = [[self.metriques_totales[n][m] for m in colonnes] for n in noms_clusters]

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
        axes[1, 2].set_title("Métriques par Cluster", fontweight='bold', fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(output_dir, "scene1_results.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Scene 1] Résultats sauvegardés: {output_path}")
        plt.show()

    def run(self, output_dir="resultats_scene1"):
        """Exécute le pipeline complet de la Scène 1."""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "="*60)
        print("SCÈNE 1 : Chat / Ciel / Sol / Arbres (K-Means k=4)")
        print("="*60)

        self.load_images()
        self.preprocess()
        self.segment_kmeans(k=4)
        self.identify_clusters()
        self.compute_metrics()
        self.visualize(output_dir)


# ============================================================
# SCÈNE 2 : Disque Rétinien Lumineux (Otsu + Morphologie)
# ============================================================

class Scene2:
    """Segmentation du disque rétinien lumineux avec Otsu et morphologie."""

    def __init__(self, image_path, gt_path):
        self.image_path = image_path
        self.gt_path = gt_path
        self.metrics_computer = MetricsComputer()

    def load_images(self):
        """Charge les images."""
        img = cv2.imread(self.image_path)
        gt = cv2.imread(self.gt_path, cv2.IMREAD_GRAYSCALE)

        if img is None or gt is None:
            raise FileNotFoundError(f"Impossible de charger les images Scene 2")

        self.img = img
        self.gt = gt
        print(f"[Scene 2] Image: {img.shape}, GT: {gt.shape}")

    def preprocess(self):
        """Prétraitement: canal vert + CLAHE + médian + top-hat."""
        # Extraction canal vert
        self.green = self.img[:, :, 1]

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        enhanced = clahe.apply(self.green)

        # Filtre médian
        self.blur = cv2.medianBlur(enhanced, 5)

        print("[Scene 2] Prétraitement terminé")

    def segment_tophat_otsu(self):
        """Segmentation: Top-hat + Otsu thresholding."""
        # Top-hat pour isoler les zones très brillantes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        tophat = cv2.morphologyEx(self.blur, cv2.MORPH_TOPHAT, kernel)

        # Seuillage Otsu
        _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphologie
        kernel_small = np.ones((8, 8), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
        self.clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_small)

        print("[Scene 2] Segmentation Otsu + morphologie terminée")

    def extract_bright_disk(self):
        """Extrait la plus grande composante connexe (le disque)."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.clean)

        self.mask = np.zeros_like(self.clean)

        best_score = 0
        best_label = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < 500:
                continue

            x, y, w, h, _ = stats[i]
            ratio = w / h if h != 0 else 0
            score = area * (1 - abs(1 - ratio))

            if score > best_score:
                best_score = score
                best_label = i

        self.mask[labels == best_label] = 255
        print(f"[Scene 2] Disque extrait (aire: {best_score:.0f})")

    def compute_metrics(self):
        """Calcule les métriques."""
        # Adapter les dimensions
        if self.gt.shape != self.mask.shape:
            gt_resized = cv2.resize(self.gt, (self.mask.shape[1], self.mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            gt_resized = self.gt

        metriques = self.metrics_computer.compute_metrics_basic(self.mask, gt_resized)
        self.metrics_computer.print_metrics(metriques, "Scène 2")

    def visualize(self, output_dir="."):
        """Affiche les résultats."""
        fig, axes = plt.subplots(1, 4, figsize=(17, 5))
        fig.suptitle("Scène 2 : Disque Rétinien Lumineux (Otsu + Morphologie)",
                     fontsize=14, fontweight='bold', y=0.98)

        axes[0].imshow(self.green, cmap='gray')
        axes[0].set_title("Canal Vert", fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(self.blur, cmap='gray')
        axes[1].set_title("CLAHE + Médian", fontsize=11)
        axes[1].axis('off')

        axes[2].imshow(self.mask, cmap='gray')
        axes[2].set_title("Segmentation", fontsize=11)
        axes[2].axis('off')

        axes[3].imshow(self.gt, cmap='gray')
        axes[3].set_title("Ground Truth", fontsize=11)
        axes[3].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(output_dir, "scene2_results.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Scene 2] Résultats sauvegardés: {output_path}")
        plt.show()

    def run(self, output_dir="resultats_scene2"):
        """Exécute le pipeline complet de la Scène 2."""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "="*60)
        print("SCÈNE 2 : Disque Rétinien Lumineux (Otsu + Morphologie)")
        print("="*60)

        self.load_images()
        self.preprocess()
        self.segment_tophat_otsu()
        self.extract_bright_disk()
        self.compute_metrics()
        self.visualize(output_dir)


# ============================================================
# SCÈNE 3 : Routes Aériennes (K-Means k=3)
# ============================================================

class Scene3:
    """Segmentation des routes aériennes avec K-Means (k=3)."""

    def __init__(self, image_path, gt_path):
        self.image_path = image_path
        self.gt_path = gt_path
        self.metrics_computer = MetricsComputer()

    def load_images(self):
        """Charge les images."""
        image = cv2.imread(self.image_path)
        gt_color = cv2.imread(self.gt_path)

        if image is None or gt_color is None:
            raise FileNotFoundError(f"Impossible de charger les images Scene 3")

        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.gt_rgb = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB)
        print(f"[Scene 3] Image: {self.image_rgb.shape}, GT: {self.gt_rgb.shape}")

    def preprocess(self):
        """Prétraitement: Filtre Gaussien + RGB→LAB."""
        blur = cv2.GaussianBlur(self.image_rgb, (5, 5), 1)
        self.lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)
        print("[Scene 3] Prétraitement terminé")

    def segment_kmeans(self, k=3):
        """Segmentation par K-Means."""
        pixels = self.lab.reshape((-1, 3))
        pixels = np.float32(pixels)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0
        )

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        labels = labels.flatten()
        self.segmented = labels.reshape(self.image_rgb.shape[:2])
        self.centers = centers

        print(f"[Scene 3] K-Means (k={k}) terminé")

    def identify_road_cluster(self):
        """Identifie le cluster des routes (plus lumineux en canal L)."""
        L_channel = self.centers[:, 0]
        self.road_cluster = np.argmax(L_channel)

        self.mask = np.zeros(self.segmented.shape, dtype=np.uint8)
        self.mask[self.segmented == self.road_cluster] = 255

        print(f"[Scene 3] Cluster route identifié: {self.road_cluster}")

    def postprocess_morphology(self):
        """Post-traitement morphologique."""
        kernel = np.ones((3, 3), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        print("[Scene 3] Post-traitement morphologique terminé")

    def extract_gt_roads(self):
        """Extrait les routes du masque GT (pixels rouges)."""
        # Adapter GT à la taille du masque
        self.gt_rgb = cv2.resize(
            self.gt_rgb,
            (self.mask.shape[1], self.mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        gt_bin = np.zeros(self.mask.shape, dtype=np.uint8)

        red = self.gt_rgb[:, :, 0]
        green = self.gt_rgb[:, :, 1]
        blue = self.gt_rgb[:, :, 2]

        gt_bin[(red > 80) & (green < 60) & (blue < 60)] = 255
        self.gt_bin = gt_bin
        print("[Scene 3] Routes GT extraites")

    def compute_metrics(self):
        """Calcule les métriques."""
        metriques = self.metrics_computer.compute_metrics_basic(self.mask, self.gt_bin)
        self.metrics_computer.print_metrics(metriques, "Scène 3")

    def visualize(self, output_dir="."):
        """Affiche les résultats."""
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        fig.suptitle("Scène 3 : Routes Aériennes (K-Means k=3)",
                     fontsize=14, fontweight='bold', y=0.98)

        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title("Image Originale", fontsize=11)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(self.segmented, cmap="nipy_spectral")
        axes[0, 1].set_title("Clusters K-Means", fontsize=11)
        axes[0, 1].axis("off")

        axes[1, 0].imshow(self.mask, cmap="gray")
        axes[1, 0].set_title("Routes Segmentées", fontsize=11)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(self.gt_bin, cmap="gray")
        axes[1, 1].set_title("Ground Truth", fontsize=11)
        axes[1, 1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(output_dir, "scene3_results.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Scene 3] Résultats sauvegardés: {output_path}")
        plt.show()

    def run(self, output_dir="resultats_scene3"):
        """Exécute le pipeline complet de la Scène 3."""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "="*60)
        print("SCÈNE 3 : Routes Aériennes (K-Means k=3)")
        print("="*60)

        self.load_images()
        self.preprocess()
        self.segment_kmeans(k=3)
        self.identify_road_cluster()
        self.postprocess_morphology()
        self.extract_gt_roads()
        self.compute_metrics()
        self.visualize(output_dir)


# ============================================================
# SCÈNE 4 : Extraction de Personne (K-Means Multi-canal)
# ============================================================

class Scene4:
    """Extraction de personne avec K-Means multi-canal [R,G,B,D]."""

    def __init__(self, color_path, depth_path, gt_path=None):
        self.color_path = color_path
        self.depth_path = depth_path
        self.gt_path = gt_path
        self.metrics_computer = MetricsComputer()

    def load_images(self):
        """Charge les trois images."""
        img_color = cv2.imread(self.color_path)
        img_depth = cv2.imread(self.depth_path)

        if img_color is None or img_depth is None:
            raise FileNotFoundError(f"Impossible de charger les images Scene 4")

        # Redimensionner si nécessaire
        if img_color.shape[:2] != img_depth.shape[:2]:
            img_depth = cv2.resize(
                img_depth,
                (img_color.shape[1], img_color.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        self.img_color = img_color
        self.img_depth = img_depth

        # Charger GT si disponible
        self.gt_mask = None
        if self.gt_path and os.path.exists(self.gt_path):
            gt = cv2.imread(self.gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt = cv2.resize(
                    gt,
                    (img_color.shape[1], img_color.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                _, self.gt_mask = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

        print(f"[Scene 4] Images chargées. Taille: {img_color.shape[:2]}")

    def preprocess(self):
        """Prétraitement: Filtre Gaussien + conversion grayscale."""
        self.img_color_blur = cv2.GaussianBlur(self.img_color, (5, 5), 0)

        if len(self.img_depth.shape) == 3:
            img_depth_gray = cv2.cvtColor(self.img_depth, cv2.COLOR_BGR2GRAY)
        else:
            img_depth_gray = self.img_depth.copy()

        self.img_depth_blur = cv2.GaussianBlur(img_depth_gray, (5, 5), 0)
        print("[Scene 4] Prétraitement terminé")

    def build_feature_matrix(self):
        """Construit la matrice de features [R,G,B,D]."""
        h, w = self.img_color_blur.shape[:2]

        img_rgb = cv2.cvtColor(self.img_color_blur, cv2.COLOR_BGR2RGB)

        rgb_norm = img_rgb.astype(np.float32) / 255.0
        depth_norm = self.img_depth_blur.astype(np.float32) / 255.0

        rgb_flat = rgb_norm.reshape(-1, 3)
        depth_flat = depth_norm.reshape(-1, 1)

        self.features = np.concatenate([rgb_flat, depth_flat], axis=1)
        self.h, self.w = h, w

        print(f"[Scene 4] Matrice de features: {self.features.shape}")

    def segment_kmeans(self, n_clusters=3):
        """Segmentation par K-Means."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            init='k-means++'
        )

        labels = kmeans.fit_predict(self.features)
        self.labels_2d = labels.reshape(self.h, self.w)
        self.cluster_centers = kmeans.cluster_centers_

        print(f"[Scene 4] K-Means ({n_clusters} clusters) terminé")

    def identify_person_cluster(self):
        """Identifie le cluster personne (stratégie: centre + profondeur)."""
        h, w = self.labels_2d.shape
        n_clusters = self.cluster_centers.shape[0]

        # Stratégie 1: Position centrale (70%)
        cx1, cx2 = w // 4, 3 * w // 4
        cy1, cy2 = h // 4, 3 * h // 4
        center_region = self.labels_2d[cy1:cy2, cx1:cx2]

        center_counts = np.bincount(center_region.flatten(), minlength=n_clusters)
        center_score = center_counts / center_counts.sum()

        # Stratégie 2: Profondeur (30%) - valeur basse = personne
        depth_values = self.cluster_centers[:, 3]
        depth_score = 1.0 - (depth_values / (depth_values.max() + 1e-8))

        # Score combiné
        combined = 0.7 * center_score + 0.3 * depth_score
        self.person_cluster = int(np.argmax(combined))

        print(f"[Scene 4] Cluster personne: {self.person_cluster}")

    def postprocess_morphology(self):
        """Post-traitement morphologique."""
        binary_mask = (self.labels_2d == self.person_cluster).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        mask_open = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

        # Composante connexe la plus grande
        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask_close)

        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            self.final_mask = (comp_labels == largest).astype(np.uint8)
        else:
            self.final_mask = mask_close

        print("[Scene 4] Post-traitement morphologique terminé")

    def compute_metrics(self):
        """Calcule les métriques (si GT disponible)."""
        if self.gt_mask is not None:
            metriques = self.metrics_computer.compute_metrics_extended(self.final_mask, self.gt_mask)
            self.metrics_computer.print_metrics(metriques, "Scène 4")
        else:
            print("[Scene 4] Pas de Ground Truth → métriques non calculées")

    def visualize(self, output_dir="."):
        """Affiche les résultats."""
        img_rgb = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)

        n_cols = 5 if self.gt_mask is not None else 4
        fig_width = 5.5 * n_cols
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 5.5))

        axes[0].imshow(img_rgb)
        axes[0].set_title("Image Couleur", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(self.img_depth_blur, cmap="gray")
        axes[1].set_title("Image Profondeur", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(self.labels_2d, cmap="tab10")
        axes[2].set_title("K-Means Clusters", fontsize=11)
        axes[2].axis("off")

        axes[3].imshow(self.final_mask, cmap="gray")
        axes[3].set_title("Masque Prédit", fontsize=11)
        axes[3].axis("off")

        if self.gt_mask is not None:
            axes[4].imshow(self.gt_mask, cmap="gray")
            axes[4].set_title("Ground Truth", fontsize=11)
            axes[4].axis("off")

        fig.suptitle("Scène 4 : Extraction de la Personne",
                     fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(output_dir, "scene4_results.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Scene 4] Résultats sauvegardés: {output_path}")
        plt.show()

    def run(self, output_dir="resultats_scene4"):
        """Exécute le pipeline complet de la Scène 4."""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "="*60)
        print("SCÈNE 4 : Extraction de Personne (K-Means Multi-canal)")
        print("="*60)

        self.load_images()
        self.preprocess()
        self.build_feature_matrix()
        self.segment_kmeans(n_clusters=3)
        self.identify_person_cluster()
        self.postprocess_morphology()
        self.compute_metrics()
        self.visualize(output_dir)


# ============================================================
# ORCHESTRATEUR PRINCIPAL
# ============================================================

class ImageSegmentationPipeline:
    """Gestionnaire principal pour exécuter les scènes."""

    def __init__(self):
        self.scenes_config = {
            1: {
                "name": "Chat/Ciel/Sol/Arbres",
                "image": "Scene_1.png",
                "gt": "GT1.png",
                "class": Scene1,
            },
            2: {
                "name": "Disque Rétinien",
                "image": "Scene_2.png",
                "gt": "GT2.png",
                "class": Scene2,
            },
            3: {
                "name": "Routes Aériennes",
                "image": "Scene_3.png",
                "gt": "GT3.png",
                "class": Scene3,
            },
            4: {
                "name": "Extraction de Personne",
                "image_color": "Scene_4_RGB_1.png",
                "image_depth": "Scene_4_D_2.png",
                "gt": "GT4.png",
                "class": Scene4,
            },
        }

    def run_scene(self, scene_num):
        """Exécute une scène spécifique."""
        if scene_num not in self.scenes_config:
            print(f"Scène {scene_num} non trouvée")
            return

        config = self.scenes_config[scene_num]

        try:
            if scene_num == 4:
                scene = config["class"](
                    config["image_color"],
                    config["image_depth"],
                    config["gt"]
                )
            else:
                scene = config["class"](config["image"], config["gt"])

            scene.run()
        except FileNotFoundError as e:
            print(f"❌ Erreur Scène {scene_num}: {e}")
        except Exception as e:
            print(f"❌ Erreur Scène {scene_num}: {e}")

    def run_all(self):
        """Exécute toutes les scènes."""
        print("\n" + "="*70)
        print("EXÉCUTION DE TOUTES LES SCÈNES")
        print("="*70)

        for scene_num in sorted(self.scenes_config.keys()):
            self.run_scene(scene_num)
            print("\n")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    pipeline = ImageSegmentationPipeline()

    if len(sys.argv) > 1:
        args = sys.argv[1:]

        if "all" in args:
            pipeline.run_all()
        else:
            for arg in args:
                try:
                    scene_num = int(arg)
                    pipeline.run_scene(scene_num)
                except ValueError:
                    print(f"Argument invalide: {arg}")
    else:
        # Par défaut, exécuter toutes les scènes
        print("Usage: python image_segmentation_unified.py [scene_number|all]")
        print("\nExemples:")
        print("  python image_segmentation_unified.py 1          # Scène 1 seulement")
        print("  python image_segmentation_unified.py 1 2 3 4    # Scènes 1,2,3,4")
        print("  python image_segmentation_unified.py all        # Toutes les scènes")
        print("\nExécution de toutes les scènes par défaut...")
        pipeline.run_all()
