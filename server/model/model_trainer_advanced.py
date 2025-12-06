"""
Advanced CNN trainer using Transfer Learning for better accuracy.

This version uses MobileNetV2 pre-trained on ImageNet as a feature extractor,
which typically achieves much higher accuracy than training from scratch.

Usage:
  python server/model/model_trainer_advanced.py --task disease --epochs 30
  python server/model/model_trainer_advanced.py --task pesticide --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a CNN using transfer learning for improved accuracy.")
    p.add_argument('--task', type=str, required=True, help='Task name, e.g., "disease" or "pesticide".')
    p.add_argument('--data-dir', type=Path, help='Path to dataset directory. If not set, auto-detects.')
    p.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size')
    p.add_argument('--img-size', type=int, nargs=2, default=[224, 224], metavar=('H', 'W'), help='Input image size')
    p.add_argument('--seed', type=int, default=123, help='Random seed')
    p.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    return p


def build_advanced_model(input_shape: tuple[int, int, int], num_classes: int, learning_rate: float) -> tf.keras.Model:
    """Build a model using transfer learning with MobileNetV2"""
    
    # Data augmentation - reduced for stability on synthetic data
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ], name='augmentation')

    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Build the model
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)  # MobileNet expects [-1, 1]
    x = data_augmentation(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    model = models.Model(inputs=inputs, outputs=outputs, name='mango_mobilenet')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model, base_model


def unfreeze_and_finetune(model, base_model, train_ds, val_ds, epochs, output_path, num_classes):
    """Fine-tune the model by unfreezing the base model"""
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning - Unfreezing base model layers")
    print("="*60)
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Fine-tune from layer 50 onwards (keep early layers frozen)
    for layer in base_model.layers[:50]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss,
        metrics=metrics
    )
    
    print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")
    
    # Callbacks for fine-tuning
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output_path),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=7, 
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        verbose=1,
        min_lr=1e-7
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt_cb, early_cb, reduce_lr_cb],
    )
    
    return history


def plot_history(history: tf.keras.callbacks.History, out_path: Path) -> None:
    hist = history.history
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in hist:
        plt.plot(hist['accuracy'], label='train_acc', linewidth=2)
    if 'val_accuracy' in hist:
        plt.plot(hist['val_accuracy'], label='val_acc', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='train_loss', linewidth=2)
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='val_loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = build_argparser().parse_args()

    # Define task-specific paths
    task_name = args.task
    artifacts_dir = THIS_DIR / 'artifacts' / task_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_model_path = artifacts_dir / 'mango_model.h5'
    history_json_path = artifacts_dir / 'history.json'
    training_plot_path = artifacts_dir / 'training_curves.png'
    labels_json_path = artifacts_dir / 'class_indices.json'

    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir.resolve()
    else:
        if task_name == 'disease':
            base_data_dir = ROOT_DIR / 'datasets' / task_name / 'MangoFruitDDS' / 'SenMangoFruitDDS_original'
            data_dir = base_data_dir
        else:
            data_dir = ROOT_DIR / 'datasets' / task_name

    if not data_dir.exists():
        raise SystemExit(f"Dataset directory not found: {data_dir}")
    
    print("="*60)
    print(f"ADVANCED TRAINING WITH TRANSFER LEARNING - Task: {task_name.upper()}")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {artifacts_dir}")
    print("="*60)

    img_height, img_width = args.img_size
    batch_size = args.batch_size

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=args.seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='int',
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=args.seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='int',
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nClasses detected: {class_names}")
    print(f"Number of classes: {num_classes}")

    # Cache and prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    print("\nBuilding model with MobileNetV2 backbone...")
    model, base_model = build_advanced_model((img_height, img_width, 3), num_classes, args.learning_rate)
    model.summary()

    # PHASE 1: Train with frozen base
    print("\n" + "="*60)
    print("PHASE 1: Initial training with frozen base model")
    print("="*60)
    
    initial_epochs = min(10, args.epochs // 2)
    
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output_model_path),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=[ckpt_cb, early_cb, reduce_lr_cb],
    )

    # PHASE 2: Fine-tune
    fine_tune_epochs = args.epochs - initial_epochs
    if fine_tune_epochs > 0:
        history_fine = unfreeze_and_finetune(
            model, base_model, train_ds, val_ds, 
            fine_tune_epochs, output_model_path, num_classes
        )
        
        # Combine histories
        for key in history.history:
            if key in history_fine.history:
                history.history[key].extend(history_fine.history[key])

    # Save final model
    model.save(output_model_path)
    print(f"\n✓ Model saved to: {output_model_path}")

    # Save artifacts
    labels_json_path.write_text(json.dumps({i: name for i, name in enumerate(class_names)}, indent=2))
    
    try:
        hist_serializable = {k: [float(x) for x in v] for k, v in history.history.items()}
        history_json_path.write_text(json.dumps(hist_serializable, indent=2))
    except Exception as e:
        print(f"Warning: Could not save history JSON: {e}")
        history_json_path.write_text(json.dumps({"error": str(e)}, indent=2))
    
    plot_history(history, training_plot_path)

    # Print final results
    final_val_acc = history.history.get('val_accuracy', [0])[-1] * 100
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE!")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print("="*60)
    print(f"✓ Model: {output_model_path}")
    print(f"✓ Labels: {labels_json_path}")
    print(f"✓ History: {history_json_path}")
    print(f"✓ Plot: {training_plot_path}")
    print("="*60)


if __name__ == '__main__':
    main()
