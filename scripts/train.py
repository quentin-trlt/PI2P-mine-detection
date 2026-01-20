"""
Script 03: Training du modèle de classification
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.augmentation import create_augmentation_layer, create_dataset
from utils.metrics import plot_training_history


def set_seeds():
    """Fixe les seeds pour reproductibilité"""
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)


def load_data_from_directory(split_name):
    """Charge les données depuis un split"""
    split_dir = config.SPLITS_DIR / split_name

    file_paths = []
    labels = []

    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(config.CLASSES))}

    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_idx = class_to_idx[class_name]

            for img_path in class_dir.glob('*.jpg'):
                file_paths.append(str(img_path))
                labels.append(class_idx)

            for img_path in class_dir.glob('*.png'):
                file_paths.append(str(img_path))
                labels.append(class_idx)

    return np.array(file_paths), np.array(labels)


def create_model(model_name='mobilenetv2', num_classes=None):
    """Crée le modèle de classification"""
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    print(f"\nCréation du modèle: {model_name}")

    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)

    # Preprocessing spécifique
    if model_name == 'mobilenetv2':
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=config.INPUT_SHAPE,
            include_top=False,
            weights='imagenet'
        )
    elif model_name == 'efficientnetb0':
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=config.INPUT_SHAPE,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Modèle non supporté: {model_name}")

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


def compile_model(model):
    """Compile le modèle avec les métriques appropriées"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


def create_callbacks(model_name):
    """Crée les callbacks pour le training"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = config.CHECKPOINTS_DIR / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / 'best_model.keras'),
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(model_dir / 'training_log.csv')
        )
    ]

    return callbacks, model_dir


def train_model(model_name='mobilenetv2', fine_tune=True):
    """Lance l'entraînement du modèle"""
    set_seeds()

    print("=" * 60)
    print("TRAINING DU MODÈLE")
    print("=" * 60)

    # Charger les données
    print("\nChargement des données...")
    train_paths, train_labels = load_data_from_directory('train')
    val_paths, val_labels = load_data_from_directory('val')

    print(f"Train: {len(train_paths)} images")
    print(f"Val: {len(val_paths)} images")

    # Créer datasets
    train_dataset = create_dataset(train_paths, train_labels, training=True)
    val_dataset = create_dataset(val_paths, val_labels, training=False)

    # Créer modèle
    model, base_model = create_model(model_name)
    model = compile_model(model)

    print(f"\nParamètres du modèle: {model.count_params():,}")

    # Callbacks
    callbacks, model_dir = create_callbacks(model_name)

    # Phase 1: Training avec base model gelé
    print("\n" + "=" * 60)
    print("PHASE 1: Training tête de classification")
    print("=" * 60)

    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=min(20, config.EPOCHS // 2),
        callbacks=callbacks,
        verbose=1
    )

    if fine_tune:
        # Phase 2: Fine-tuning
        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning")
        print("=" * 60)

        # Dégeler le base model
        base_model.trainable = True

        # Recompiler avec learning rate plus faible
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        print(f"Paramètres entraînables: {model.count_params():,}")

        history2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.EPOCHS,
            initial_epoch=history1.epoch[-1],
            callbacks=callbacks,
            verbose=1
        )

        # Fusionner historiques
        for key in history1.history:
            history1.history[key].extend(history2.history[key])

    # Sauvegarder historique
    plot_training_history(
        history1,
        save_path=model_dir / 'training_history.png'
    )

    print(f"\n✓ Modèle sauvegardé dans: {model_dir}")
    print(f"✓ Meilleur modèle: {model_dir / 'best_model.keras'}")

    return model, history1, model_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Training du modèle')
    parser.add_argument(
        '--model',
        type=str,
        default=config.DEFAULT_MODEL,
        choices=['mobilenetv2', 'efficientnetb0'],
        help='Modèle à utiliser'
    )
    parser.add_argument(
        '--no-finetune',
        action='store_true',
        help='Désactiver le fine-tuning'
    )

    args = parser.parse_args()

    # Vérifier que les splits existent
    if not config.SPLITS_DIR.exists():
        print("✗ Splits non trouvés. Exécutez d'abord 02_prepare_data.py")
        sys.exit(1)

    train_model(
        model_name=args.model,
        fine_tune=not args.no_finetune
    )

    print("\n=== Training terminé ===")