import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox




#Ensures histogram output folders exist and returns their paths
def ensure_output_dirs(base_out_dir: str) -> dict:

    animal_dir = os.path.join(base_out_dir, "AnimalHistogram")
    non_animal_dir = os.path.join(base_out_dir, "NonAnimalHistogram")
    os.makedirs(animal_dir, exist_ok=True)
    os.makedirs(non_animal_dir, exist_ok=True)
    return {"animal": animal_dir, "non_animal": non_animal_dir}

#creates 2 txt files with the names of the .wav files split into animal and non-animal categories 
def categorize_sounds(csv_path: str, out_dir: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"esc50.csv not found at: {csv_path}")

    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(csv_path)

    is_animal = meta["target"].astype(int).between(0, 9)  # Assuming animal classes are labeled 0-9

    animals = meta.loc[is_animal, "filename"].tolist()
    non_animals = meta.loc[~is_animal, "filename"].tolist()

    anim_list = os.path.join(out_dir, "AnimalList.txt")
    non_list  = os.path.join(out_dir, "NonAnimalList.txt")
    with open(anim_list, "w", encoding="utf-8") as f:
        f.write("\n".join(animals))
    with open(non_list, "w", encoding="utf-8") as f:
        f.write("\n".join(non_animals))

    
    print(f"animals: {len(animals)} files -> {anim_list}")
    print(f"non-animals: {len(non_animals)} files -> {non_list}")

     # Optional quick sanity check
    #print("\nExample rows (first 3 animals, first 3 non-animals):")
    #print(meta.loc[is_animal, ["filename", "target", "category"]].head(3))
    #print(meta.loc[~is_animal, ["filename", "target", "category"]].head(3))

    return animals, non_animals

#Finds the correct full path to a given .wav file inside the dataset folder
def resolve_wav_path(data_dir: str, filename: str) -> str:
    p1 = os.path.join(data_dir, filename)
    p2 = os.path.join(data_dir, "audio", filename)
    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2
    raise FileNotFoundError(f"Missing WAV: {filename}")

#Generates and saves a Mel-spectrogram image from a waveform.
def save_spectrogram(y: np.ndarray, sr: int, out_png: str, title: str = ""):
    """Saves a spectrogram (heatmap-style) from a waveform."""
    plt.figure(figsize=(6, 2))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    if title:
        plt.title(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()

# Builds spectrograms for all animal and non-animal sounds
def build_all(csv_path: str, data_dir: str, out_dir: str):
    out_dirs = ensure_output_dirs(out_dir)
    animal_files, non_animal_files = categorize_sounds(csv_path, out_dir)

    # animals
    print(f"\nGenerating spectrograms for {len(animal_files)} animal sounds...")
    for fn in tqdm(animal_files,desc="Animal files",unit = "file"):
        wav = resolve_wav_path(data_dir, fn)
        y, sr = librosa.load(wav, sr=None)
        out_png = os.path.join(out_dirs["animal"], fn.replace(".wav", ".png"))
        save_spectrogram(y, sr, out_png, title=f"animal • {fn}")

    # non-animals
    for fn in tqdm(non_animal_files,desc="Non-animal files",unit = "file"):
        wav = resolve_wav_path(data_dir, fn)
        y, sr = librosa.load(wav, sr=None)
        out_png = os.path.join(out_dirs["non_animal"], fn.replace(".wav", ".png"))
        save_spectrogram(y, sr, out_png, title=f"non-animal • {fn}")

# Alert popup to inform user about processing time
def alert_popup(title: str , message: str ):
    root = tk.Tk()
    root.withdraw()  # hide main window
    messagebox.showwarning(title, message)
    root.destroy()

#main function for histogram creating
def main():
    #make directoiries for output histograms
    paths = ensure_output_dirs("output_histograms")
    print("Created/verified:")
    for k, p in paths.items():
        print(f"  {k}: {p} (exists={os.path.isdir(p)})")
    
    #categorise sounds into animal and non-animal lists
    CSV_PATH  = "data/esc50.csv"
    DATA_DIR = "data/audio"
    OUT_DIR  = "output_histograms"
    #categorize_sounds(CSV_PATH , OUT_DIR)
# Check if the data/audio folder exists or contains WAVs
    if not os.path.exists(DATA_DIR) or not any(f.endswith(".wav") for f in os.listdir(DATA_DIR)):
        alert_popup(
            "Audio Files Missing",
            r"No audio files were found in the '\data\audio' folder.""\n\n"
            r"Please make sure the ESC-50 dataset is extracted and the .wav files are placed inside '\data\audio'."
        )
        return  # Stop execution so the script doesn’t fail later

    alert_popup("Processing Notice","Creating spectrograms will take several minutes (about 10 minutes).\nPlease be patient!")
    build_all(CSV_PATH, DATA_DIR, OUT_DIR)

if __name__ == "__main__":
    main()
    