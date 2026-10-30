"""
PROJECT: Development of a Convolutional Neural Network (CNN) for
Dog-Cat Image Classification Using Three Convolutional Layers

Run: python main.py
Outputs are saved in the outputs/ folder.
"""
import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# --------------------------- Configuration ---------------------------
SEED = 42
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = BASE_DIR / "cats_and_dogs_filtered.zip"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "dog_cat_cnn_model.keras"

DATA_URL = (
    "https://storage.googleapis.com/"
    "tensorflow-1-public/course2/cats_and_dogs_filtered.zip"
)
class_names=["cats", "dogs"]
DATASET_CLASS_NAMES = ["cats", "dogs"]

np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def print_header(title):
    print("\n" + "=" * 72)
    print(title.center(72))
    print("=" * 72)


def download_dataset():
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0:
        print(f"Dataset archive found: {ZIP_PATH}")
        return

    print_header("DOWNLOADING DATASET")
    print("Downloading Cats vs Dogs filtered dataset...")
    try:
        request = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            ZIP_PATH.write_bytes(response.read())
        print(f"Download complete: {ZIP_PATH}")
    except Exception as exc:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
        print("\nDataset download failed.")
        print(f"Reason: {exc}")
        print("\nManual option:")
        print("1. Open this URL in your browser:")
        print(DATA_URL)
        print(f"2. Save the ZIP file as: {ZIP_PATH}")
        print("3. Run python main.py again.")
        sys.exit(1)


def prepare_dataset():
    download_dataset()
    extracted_dir = DATA_DIR / "cats_and_dogs_filtered"
    train_dir = extracted_dir / "train"
    val_dir = extracted_dir / "validation"

    if not train_dir.exists() or not val_dir.exists():
        print_header("EXTRACTING DATASET")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as archive:
                archive.extractall(DATA_DIR)
        except zipfile.BadZipFile:
            print("The downloaded file is not a valid ZIP.")
            print(f"Delete {ZIP_PATH} and download it again.")
            sys.exit(1)

    if not train_dir.exists() or not val_dir.exists():
        print("Dataset folders were not found after extraction.")
        print(f"Expected folders: {train_dir} and {val_dir}")
        sys.exit(1)

    print_header("LOADING DATASET")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", class_names=CLASS_NAMES,
        shuffle=True, seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", class_names=DATASET_CLASS_NAMES, shuffle=False
    )

    # This starter dataset has train/validation folders only. Validation is
    # monitored during training and also reported below; use a separate
    # untouched test folder for a formal independent test benchmark.
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
    )


def build_model():
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
    ], name="data_augmentation")

    # Exactly three Conv2D layers: 32, 64 and 128 filters.
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.40),
        layers.Dense(1, activation="sigmoid"),
    ], name="Dog_Cat_Three_Layer_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_training_graphs(history):
    epochs_ran = range(1, len(history.history["accuracy"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs_ran, history.history["accuracy"], marker="o",
            markersize=3, label="Training Accuracy")
    ax.plot(epochs_ran, history.history["val_accuracy"], marker="o",
            markersize=3, label="Validation Accuracy")
    ax.set_title("CNN Training Performance — Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_graph.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs_ran, history.history["loss"], marker="o",
            markersize=3, label="Training Loss")
    ax.plot(epochs_ran, history.history["val_loss"], marker="o",
            markersize=3, label="Validation Loss")
    ax.set_title("CNN Training Performance — Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_graph.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def evaluate_model(model, test_ds):
    print_header("FINAL MODEL EVALUATION")
    test_loss, keras_accuracy = model.evaluate(test_ds, verbose=1)

    y_true_parts = [
        labels.numpy().reshape(-1).astype(int)
        for _, labels in test_ds
    ]
    y_true = np.concatenate(y_true_parts)
    y_prob = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = None

    metrics = {
        "test_loss": float(test_loss),
        "keras_test_accuracy": float(keras_accuracy),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "evaluated_images": int(len(y_true)),
    }

    print("\n+----------------------------------------------------------+")
    print("|                    EVALUATION SUMMARY                   |")
    print("+----------------------------------------------------------+")
    print(f"| Test images evaluated : {len(y_true):>6}                       |")
    print(f"| Test loss             : {test_loss:>8.4f}                       |")
    print(f"| Accuracy              : {accuracy * 100:>7.2f}%                      |")
    print(f"| Precision             : {precision * 100:>7.2f}%                      |")
    print(f"| Recall                : {recall * 100:>7.2f}%                      |")
    print(f"| F1-score              : {f1 * 100:>7.2f}%                      |")
    print(f"| ROC-AUC               : {roc_auc if roc_auc is not None else 'N/A':>8}                       |")
    print("+----------------------------------------------------------+")

    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=CLASS_NAMES, zero_division=0, digits=4
    )
    print("\nCLASSIFICATION REPORT\n")
    print(report)

    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        "DOG-CAT CNN — CLASSIFICATION REPORT\n"
        + "=" * 45 + "\n" + report + "\n\nMETRICS\n"
        + json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASS_NAMES
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Dog–Cat CNN — Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return y_true, y_pred, y_prob


def save_prediction_gallery(model, test_ds, y_pred, y_prob, max_images=12):
    images_list, labels_list = [], []
    for images, labels in test_ds:
        images_list.append(images.numpy())
        labels_list.append(labels.numpy().reshape(-1).astype(int))
    images_all = np.concatenate(images_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    count = min(max_images, len(images_all))
    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= count:
            continue
        actual = CLASS_NAMES[int(labels_all[idx])]
        predicted = CLASS_NAMES[int(y_pred[idx])]
        confidence = float(y_prob[idx]) if y_pred[idx] == 1 else 1 - float(y_prob[idx])
        ax.imshow(np.clip(images_all[idx] / 255.0, 0, 1))
        ax.set_title(
            f"Actual: {actual}\nPredicted: {predicted}\nConfidence: {confidence * 100:.1f}%",
            color="green" if actual == predicted else "red",
            fontsize=10, fontweight="bold"
        )

    fig.suptitle(
        "Sample Dog–Cat Predictions (Green = Correct, Red = Incorrect)",
        fontsize=15, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_gallery.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_header("DOG–CAT IMAGE CLASSIFICATION USING CNN")
    print("Architecture : Three convolutional layers (32, 64, 128 filters)")
    print("Framework    : TensorFlow / Keras")
    print(f"TensorFlow   : {tf.__version__}")
    print(f"Epoch limit  : {EPOCHS}")

    train_ds, val_ds, test_ds = prepare_dataset()
    model = build_model()

    print_header("CNN MODEL ARCHITECTURE")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-6, verbose=1
        ),
    ]

    print_header("MODEL TRAINING")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        callbacks=callbacks, verbose=1
    )

    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
    model.save(MODEL_PATH)
    print(f"\nTrained model saved to: {MODEL_PATH}")

    save_training_graphs(history)
    y_true, y_pred, y_prob = evaluate_model(model, test_ds)
    save_prediction_gallery(model, test_ds, y_pred, y_prob)

    print_header("PROJECT COMPLETED SUCCESSFULLY")
    print(f"All output files are saved in:\n{OUTPUT_DIR}")
    print("\nGenerated files:")
    for filename in [
        "dog_cat_cnn_model.keras", "accuracy_graph.png", "loss_graph.png",
        "confusion_matrix.png", "prediction_gallery.png",
        "evaluation_metrics.json", "classification_report.txt"
    ]:
        print(f"  ✓ {filename}")
    print("\nNote: this dataset provides train and validation folders only.")
    print("The validation folder is used for monitoring and the reported")
    print("evaluation, so use a separate untouched test set for a formal benchmark.")


if __name__ == "__main__":
    main()
