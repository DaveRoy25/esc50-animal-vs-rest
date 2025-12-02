import tensorflow as tf

'''
load and prepare labeled spectrograms for the CNN.
function will load spectrograms into TensorFlow as labeled batches.
1.Split the dataset into training and validation sets.
2.Automatically assign labels (0 = Non-animal, 1 = Animal) based on folder names.
3.Resize and batch the spectrograms for efficient model input.'''

def load_dataset(base_dir="output_histograms", img_size=(128,128), batch_size=32):
    """
    Loads spectrogram images from folders into TensorFlow datasets,
    automatically labels them (animal / non-animal), resizes them,
    and splits them into training and validation sets.

    Args:
        base_dir (str): Path to the main dataset folder.
                        Inside, you should have subfolders like:
                        output_histograms/
                          ├── AnimalHistogram/
                          └── NonAnimalHistogram/
        img_size (tuple): Target image size (height, width) for resizing.
                          CNNs usually require all images to have the same dimensions.
        batch_size (int): How many images to load per batch during training.
                          Typical values are 16, 32, or 64.

    Returns:
        train_ds (tf.data.Dataset): TensorFlow dataset for model training.
        val_ds (tf.data.Dataset): TensorFlow dataset for validation.
    """

    # ----------------------------------------
    # Create the training dataset.
    # TensorFlow automatically:
    #   - Reads all images inside the given folder
    #   - Labels them based on subfolder names
    #   - Resizes them to `img_size`
    #   - Splits them into subsets (training/validation)
    # ----------------------------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,              # folder containing AnimalHistogram/ and NonAnimalHistogram/
        validation_split=0.2,  # use 20% of the data for validation
        subset="training",     # this call loads the training subset
        seed=42,               # random seed for consistent splitting
        image_size=img_size,   # resize all images to the same size
        batch_size=batch_size  # number of images per batch
    )

    # ----------------------------------------
    # Create the validation dataset.
    # Same as above, but with subset="validation".
    # TensorFlow automatically ensures it uses the remaining 20%.
    # ----------------------------------------
    val_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # ----------------------------------------
    # (Optional but recommended)
    # Improve performance by caching and prefetching.
    # This tells TensorFlow to prepare the next batch while
    # the current one is being processed, using parallel threads.
    # ----------------------------------------
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Print dataset info for confirmation
    print("✅ Dataset loaded successfully:")
    print(f"  Training batches:   {len(train_ds)}")
    print(f"  Validation batches: {len(val_ds)}")

    # Return both datasets to be used later for model training
    return train_ds, val_ds
