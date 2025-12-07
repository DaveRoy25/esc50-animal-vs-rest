import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(model_path="animal_classifier.h5", data_dir="output_histograms", img_size=(128,128)):
    """
    Loads a trained CNN model and evaluates it on new or validation data.

    Args:
        model_path (str): Path to the trained .h5 model.
        data_dir (str): Folder containing spectrogram subfolders.
        img_size (tuple): Target image size for preprocessing.
    """

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
        batch_size=32
    )
    class_names = val_ds.class_names
    val_ds = val_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    print("Class names:", class_names)


    # Evaluate model on validation data
    print("\n Evaluating model...")
    test_loss, test_acc = model.evaluate(val_ds)
    print(f"\nValidation Accuracy: {test_acc*100:.2f}%")
    print(f"Validation Loss: {test_loss:.4f}\n")

    # Pick a few random examples from validation set to visualize predictions
    plt.figure(figsize=(10, 10))
    for images, labels in val_ds.take(1):
        predictions = model.predict(images)
  # If model is binary (sigmoid), convert probabilities to class 0/1
        if predictions.shape[1] == 1:
            predicted_labels = (predictions > 0.5).astype(int).flatten()
        else:
            # If model outputs 2-class softmax
            predicted_labels = np.argmax(predictions, axis=1)
        
        # Optional debug print
        print("\nSample predictions:", predictions[:10].flatten())
        print("Predicted labels:", predicted_labels[:10])
        print("True labels:", labels[:10].numpy())

        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            true_label = class_names[labels[i]]
            pred_label = class_names[predicted_labels[i]]

            color = "green" if true_label == pred_label else "red"
            plt.title(f"P: {pred_label}\nT: {true_label}", color=color)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
