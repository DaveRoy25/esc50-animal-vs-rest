import tensorflow as tf
from load_dataset import load_dataset

def train_cnn(train_ds=None, val_ds=None, epochs=10):


    # If not provided, load default dataset
    if train_ds is None or val_ds is None:
        train_ds, val_ds = load_dataset("output_histograms")
        
    #print("Class mapping:", train_ds.class_names)

     # Define a deeper CNN with regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.4),  # prevent memorization
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'] #Track percentage of correct predictions.
    )

    # Print model summary
    model.summary()

    # Train
    #class_weights = {0: 1.0, 1: 4.0}  # Give 4× more importance to Animal
    class_weights = {0: 4.0, 1: 1.0}  # Give 4× more importance to Animal

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight = class_weights
    )

    # Save trained model
    model.save("animal_classifier.h5")
    print("Model saved as animal_classifier.h5")

    return model, history


if __name__ == "__main__":
    
    model, history = train_cnn()        # Train and get history
    from plot_training import plot_training
    plot_training(history)              #  show plots of training/validation accuracy/loss

