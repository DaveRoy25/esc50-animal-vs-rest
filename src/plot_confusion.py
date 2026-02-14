import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def _make_stratified_val_dataset(
    data_dir: str,
    img_size=(128, 128),
    batch_size=32,
    val_fraction=0.2,
    seed=42,
):
    """
    Build a STRATIFIED validation dataset from a directory with class subfolders.

    This avoids the issue where TF's validation_split may accidentally produce a
    validation set containing only one class (not stratified).
    """
    # Keep stable class order
    class_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    rng = np.random.default_rng(seed)

    val_paths = []
    val_labels = []

    for label_idx, class_name in enumerate(class_names):
        pattern = os.path.join(data_dir, class_name, "*.png")
        paths = sorted(glob.glob(pattern))

        if len(paths) == 0:
            continue

        rng.shuffle(paths)

        n_val = int(round(len(paths) * val_fraction))
        # Ensure at least 1 sample per class if possible
        n_val = max(1, n_val)

        val_subset = paths[:n_val]
        val_paths.extend(val_subset)
        val_labels.extend([label_idx] * len(val_subset))

    # Shuffle whole validation set (optional)
    val_paths = np.array(val_paths)
    val_labels = np.array(val_labels, dtype=np.int32)
    perm = rng.permutation(len(val_paths))
    val_paths = val_paths[perm]
    val_labels = val_labels[perm]

    def load_png(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)  # keep consistent with typical TF pipelines
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    ds = ds.map(load_png, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, class_names, val_paths


def _plot_category_prediction_heatmap(ct: pd.DataFrame, title: str, top_n: int = 25):
    """
    Plot a heatmap for a small table (Category x PredLabel) using matplotlib only.
    """
    # Make sure both columns exist (in case some pred label never appears)
    for col in ["AnimalHistogram", "NonAnimalHistogram"]:
        if col not in ct.columns:
            ct[col] = 0

    # Keep just those two columns in a stable order
    ct = ct[["AnimalHistogram", "NonAnimalHistogram"]]

    # Pick top categories by total
    ct2 = ct.copy()
    ct2["_total"] = ct2.sum(axis=1)
    ct2 = ct2.sort_values("_total", ascending=False).drop(columns=["_total"]).head(top_n)

    mat = ct2.to_numpy()
    row_labels = ct2.index.tolist()
    col_labels = ct2.columns.tolist()

    plt.figure(figsize=(8, max(6, int(0.35 * len(row_labels) + 2))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()

    plt.xticks(range(len(col_labels)), col_labels, rotation=0)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("ESC-50 Category")

    # Annotate counts inside cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, str(int(mat[i, j])), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_confusion_and_category_matrix(
    model_path="animal_classifier.h5",
    data_dir="output_histograms",
    img_size=(128, 128),
    batch_size=32,
    threshold=0.5,
    val_fraction=0.2,
    seed=42,
    esc50_csv_path="data/esc50.csv",
    top_n_categories_to_plot=30,
):
    """
    1) Builds a STRATIFIED validation set from data_dir.
    2) Plots the normal 2x2 confusion matrix (Animal vs NonAnimal).
    3) Plots a "Category vs Predicted Label" matrix (ESC-50 categories vs binary output).

    NOTE:
    - The category plot is NOT a classic confusion matrix; it is a useful analysis
      showing which ESC-50 categories tend to be predicted as Animal / NonAnimal.
    """

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.\n")

    print("Building STRATIFIED validation dataset...")
    val_ds, class_names, val_paths = _make_stratified_val_dataset(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
    )

    print("Class names:", class_names)

    # Collect y_true
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

    print("\nValidation label counts:", np.bincount(y_true))
    for i, name in enumerate(class_names):
        print(f"  {i} -> {name}: {int(np.sum(y_true == i))}")

    # Predict
    print("\nPredicting on validation dataset...")
    y_prob = model.predict(val_ds, verbose=1)

    # Convert to y_pred
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_pred = (y_prob[:, 0] > threshold).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    # ---- 2x2 confusion matrix (true confusion matrix) ----
    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification report (binary classes):\n")
    try:
        print(classification_report(y_true, y_pred, target_names=class_names))
    except Exception:
        print(classification_report(y_true, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format="d")
    plt.title(f"Binary Confusion Matrix (threshold={threshold})")
    plt.tight_layout()
    plt.show()

    # ---- ESC-50 Category vs Predicted Label matrix ----
    print("\nLoading ESC-50 metadata for category mapping...")
    meta = pd.read_csv(esc50_csv_path)
    filename_to_category = dict(zip(meta["filename"], meta["category"]))

    rows = []
    for p, t, pred in zip(val_paths, y_true, y_pred):
        png = os.path.basename(p)
        wav = png.replace(".png", ".wav")
        category = filename_to_category.get(wav, "Unknown")

        rows.append({
            "category": category,
            "true_label": class_names[int(t)],
            "pred_label": class_names[int(pred)]
        })

    df = pd.DataFrame(rows)

    # Category x PredLabel counts (THIS is what you wanted to see visually)
    ct = pd.crosstab(df["category"], df["pred_label"])
    print("\nCategory x PredictedLabel table (top rows shown):")
    tmp = ct.copy()
    tmp["_total"] = tmp.sum(axis=1)
    tmp = tmp.sort_values("_total", ascending=False).drop(columns=["_total"])
    print(tmp.head(30))

    _plot_category_prediction_heatmap(
        ct,
        title="ESC-50 Category vs Model Prediction (counts)",
        top_n=top_n_categories_to_plot
    )

    # Optional: show FP/FN category lists (useful for report)
    animal_name = "AnimalHistogram"
    non_animal_name = "NonAnimalHistogram"

    fp = df[(df["true_label"] == non_animal_name) & (df["pred_label"] == animal_name)]
    fn = df[(df["true_label"] == animal_name) & (df["pred_label"] == non_animal_name)]

    print("\nTop FALSE POSITIVE categories (non-animal -> animal):")
    print(fp["category"].value_counts().head(20))

    print("\nTop FALSE NEGATIVE categories (animal -> non-animal):")
    print(fn["category"].value_counts().head(20))


if __name__ == "__main__":
    plot_confusion_and_category_matrix()
