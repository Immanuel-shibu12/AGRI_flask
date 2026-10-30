"""
Train ML Classifier for Crop Pest & Leaf Disease Diagnosis
Extracts rich color, texture, and necrosis features from leaf imagery
and trains a scikit-learn RandomForestClassifier model.
Saves model to pest_model.pkl.
"""

import os
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pest_model.pkl")

CLASSES = [
    "healthy",
    "brown_spots",
    "yellowing_leaves",
    "chewed_holes",
    "powdery_mildew",
]

CLASS_METADATA = {
    "healthy": {
        "label": "Healthy Leaf — Optimal Foliage",
        "description": "Leaf texture, chlorophyll density, and vein structure are within standard healthy ranges.",
        "treatment": "No chemical intervention needed. Maintain balanced NPK nutrition and continue weekly scouting.",
        "ipm": "Regular monitoring, balanced irrigation, and preventive bio-fertilizers.",
    },
    "brown_spots": {
        "label": "Fungal Leaf Spot / Blast Lesion",
        "description": "Necrotic brown lesions with distinct dark margins, typical of Magnaporthe oryzae (Blast) or Cercospora leaf spot.",
        "treatment": "Apply Tricyclazole 75 WP (0.6 g/L) or Propiconazole 25 EC (1 mL/L). Improve field drainage.",
        "ipm": "Avoid excess nitrogen fertilizer, destroy infected stubble, and promote air circulation between plant rows.",
    },
    "yellowing_leaves": {
        "label": "Chlorosis / Nitrogen Deficiency or Viral Tungro",
        "description": "Interveinal or uniform leaf yellowing indicating nutrient deficiency or leafhopper-transmitted viral infection.",
        "treatment": "Apply foliar urea (2% solution) or micronutrient zinc spray. Control green leafhopper vectors if viral.",
        "ipm": "Soil test verification, vector population control using yellow sticky traps, and removal of diseased hills.",
    },
    "chewed_holes": {
        "label": "Foliage Damage / Caterpillar Feeding",
        "description": "Irregular perforations and margin feeding caused by Spodoptera, stem borer larvae, or leaf folders.",
        "treatment": "Apply Chlorantraniliprole 18.5 SC (0.3 mL/L) or Bacillus thuringiensis (Bt) bio-formulation (2 g/L).",
        "ipm": "Install pheromone traps (5 traps/acre), encourage predatory birds, and hand-pick egg clusters at tillering.",
    },
    "powdery_mildew": {
        "label": "Powdery Mildew / Fungal Surface Mold",
        "description": "White talcum-like superficial fungal growth across leaf blades inhibiting photosynthesis.",
        "treatment": "Spray Wettable Sulfur 80 WP (2.5 g/L) or Azoxystrobin (1 mL/L) in cool morning hours.",
        "ipm": "Prune overcrowded lower foliage to lower humidity and avoid late-evening overhead sprinkling.",
    },
}


def extract_features_from_image(img: Image.Image) -> np.ndarray:
    """
    Extract a multi-dimensional feature vector from a PIL Image:
    - RGB channel statistics (means, standard deviations, skewness)
    - Color ratios (Greenness Index, Yellowness Index, Necrosis Index, Whiteness Index)
    - High-frequency edge gradient energy (lesion boundary texture)
    """
    img_rgb = img.convert("RGB").resize((128, 128))
    arr = np.array(img_rgb).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total_pixels = 128 * 128

    # Channel stats
    mean_r, std_r = np.mean(r), np.std(r)
    mean_g, std_g = np.mean(g), np.std(g)
    mean_b, std_b = np.mean(b), np.std(b)

    # Normalized colors
    sum_rgb = r + g + b + 1e-5
    norm_r = r / sum_rgb
    norm_g = g / sum_rgb
    norm_b = b / sum_rgb

    # Agronomic color indices
    # Greenness: excess green (2G - R - B)
    ex_green = (2.0 * g - r - b) / 255.0
    mean_ex_green = np.mean(ex_green)

    # Yellowness: high R and G, low B
    yellow_mask = (r > 130) & (g > 130) & (b < 110)
    yellow_ratio = np.sum(yellow_mask) / total_pixels

    # Brown / Necrosis: dark low-intensity brown
    brown_mask = (r < 115) & (g < 95) & (b < 85) & (r > b)
    brown_ratio = np.sum(brown_mask) / total_pixels

    # Pure green foliage
    green_mask = (g > r * 1.1) & (g > b * 1.1) & (g > 55)
    green_ratio = np.sum(green_mask) / total_pixels

    # White powdery coating
    white_mask = (r > 175) & (g > 175) & (b > 175)
    white_ratio = np.sum(white_mask) / total_pixels

    # Texture / Edge energy (Sobel approximation)
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_energy = (np.mean(dx) + np.mean(dy)) / 255.0

    # Hole / darkness ratio (chewed void patterns)
    dark_hole_mask = (r < 40) & (g < 40) & (b < 40)
    hole_ratio = np.sum(dark_hole_mask) / total_pixels

    return np.array([
        mean_r, std_r, mean_g, std_g, mean_b, std_b,
        np.mean(norm_r), np.mean(norm_g), np.mean(norm_b),
        mean_ex_green, yellow_ratio, brown_ratio, green_ratio,
        white_ratio, edge_energy, hole_ratio
    ], dtype=np.float32)


def generate_synthetic_leaf(category: str, seed: int) -> Image.Image:
    """Generate representative training leaf images with realistic variations."""
    np.random.seed(seed)
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Base leaf shape & color
    base_g = int(np.random.uniform(120, 180))
    base_r = int(np.random.uniform(30, 80))
    base_b = int(np.random.uniform(20, 60))
    draw.ellipse([10, 10, 118, 118], fill=(base_r, base_g, base_b))

    if category == "healthy":
        # Lush uniform green with minor veins
        for _ in range(8):
            x1, y1 = np.random.randint(20, 100), np.random.randint(20, 100)
            draw.line([x1, y1, x1 + np.random.randint(-15, 15), y1 + np.random.randint(5, 25)],
                      fill=(base_r + 10, min(255, base_g + 20), base_b + 5), width=1)

    elif category == "brown_spots":
        # Dark brown blast / spot lesions with yellow halo
        num_spots = np.random.randint(4, 12)
        for _ in range(num_spots):
            cx, cy = np.random.randint(25, 100), np.random.randint(25, 100)
            rad = np.random.randint(4, 14)
            # Halo
            draw.ellipse([cx - rad - 2, cy - rad - 2, cx + rad + 2, cy + rad + 2], fill=(160, 140, 40))
            # Necrotic center
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(65, 35, 20))

    elif category == "yellowing_leaves":
        # Diffuse chlorosis yellow wash
        draw.ellipse([15, 15, 113, 113], fill=(int(np.random.uniform(180, 220)), int(np.random.uniform(170, 210)), 45))

    elif category == "chewed_holes":
        # Irregular holes eaten out
        num_holes = np.random.randint(3, 7)
        for _ in range(num_holes):
            hx, hy = np.random.randint(20, 100), np.random.randint(20, 100)
            hrad = np.random.randint(6, 16)
            draw.ellipse([hx - hrad, hy - hrad, hx + hrad, hy + hrad], fill=(15, 15, 15))

    elif category == "powdery_mildew":
        # White fungal powder blotches
        for _ in range(12):
            px, py = np.random.randint(20, 105), np.random.randint(20, 105)
            prad = np.random.randint(5, 18)
            draw.ellipse([px - prad, py - prad, px + prad, py + prad], fill=(215, 220, 215))

    return img.filter(ImageFilter.GaussianBlur(radius=0.7))


def train_and_save_model():
    print("Generating leaf dataset for training...")
    X, y = [], []
    samples_per_class = 200

    for cat_idx, cat in enumerate(CLASSES):
        for i in range(samples_per_class):
            leaf_img = generate_synthetic_leaf(cat, seed=cat_idx * 1000 + i)
            features = extract_features_from_image(leaf_img)
            X.append(features)
            y.append(cat)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print(f"Training RandomForestClassifier on {len(X_train)} samples across {len(CLASSES)} disease categories...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train_and_save_model()
