"""
Configuration centralisée pour le projet de classification
"""
import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TFLITE_DIR = MODELS_DIR / "tflite"

# Kaggle
KAGGLE_DATASET = "asdasdasasdas/garbage-classification"
KAGGLE_TOKEN = "KGAT_b301a4f90e5a03263c70251ad924c06c"

# Paramètres images
IMG_SIZE = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# Paramètres training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'brightness_range': [0.8, 1.2]
}

# Modèles disponibles
AVAILABLE_MODELS = {
    'mobilenetv2': {
        'name': 'MobileNetV2',
        'params': 3.5e6,
        'input_size': 224
    },
    'efficientnetb0': {
        'name': 'EfficientNetB0',
        'params': 5.3e6,
        'input_size': 224
    }
}

# Modèle par défaut
DEFAULT_MODEL = 'mobilenetv2'

# TFLite
QUANTIZE = True
REPRESENTATIVE_DATASET_SIZE = 100

# Métriques
METRICS_TO_TRACK = ['accuracy', 'precision', 'recall', 'f1']
TARGET_RECALL = 0.95  # Objectif critique pour détection mines

# Classes (sera mis à jour après téléchargement)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASSES)

# Seed pour reproductibilité
RANDOM_SEED = 42