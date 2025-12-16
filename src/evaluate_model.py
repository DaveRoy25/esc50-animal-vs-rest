import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def evaluate_model(model_path="animal_classifier.h5", data_dir="output_histograms", img_size=(128,128)):
    """
    Loads a trained CNN model and evaluates it on new or validation data.

    Args:
        model_path (str): Path to the trained .h5 model.
        data_dir (str): Folder containing spectrogram subfolders.
        img_size (tuple): Target image size for preprocessing.
    """
    meta = pd.read_csv("data/esc50.csv")
    filename_to_category = dict(zip(meta["filename"], meta["category"]))
     
    # Load the trained model
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!\n")

    # Load validation data
    print("Loading validation data for testing...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=32,
        shuffle=False  # IMPORTANT: keep stable for evaluation
    )

    class_names = val_ds.class_names
    file_paths = val_ds.file_paths  # aligned with shuffle=False

    print("Class names:", class_names)

    # Evaluate model on validation data
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(val_ds, verbose=1)
    print(f"\nValidation Accuracy: {test_acc*100:.2f}%")
    print(f"Validation Loss: {test_loss:.4f}\n")

    # pick 9 random files from the validation file list
    n_show = 9
    idx = np.random.choice(len(file_paths), size=n_show, replace=False)
    chosen_paths = [file_paths[i] for i in idx]

    # Load those images from disk (so names and images match 100%
    imgs = []
    true_labels = []
    for p in chosen_paths:
        # true label from folder name
        folder = os.path.basename(os.path.dirname(p))
        true_labels.append(class_names.index(folder))

        img = tf.keras.utils.load_img(p, target_size=img_size)
        img = tf.keras.utils.img_to_array(img)
        imgs.append(img)

    images = np.stack(imgs, axis=0)

    # # If model is binary (sigmoid), convert probabilities to class 0/1
    preds = model.predict(images)
    if preds.shape[1] == 1:
        pred_labels = (preds > 0.5).astype(int).flatten()
        probs = preds.flatten()
    else:
        # If model outputs 2-class softmax
        pred_labels = np.argmax(preds, axis=1)
        probs = np.max(preds, axis=1)

    # Plot
    plt.figure(figsize=(12, 10))
    for i in range(n_show):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))

        filename_png = os.path.basename(chosen_paths[i])
        filename_wav = filename_png.replace(".png", ".wav")
        sound_name = filename_to_category.get(filename_wav, "Unknown")

        t = class_names[true_labels[i]]
        p = class_names[pred_labels[i]]
        color = "green" if t == p else "red"

        plt.title(f"{filename_wav}\n({sound_name})\nP: {p} ({probs[i]:.2f}) | T: {t}",
                  color=color, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
