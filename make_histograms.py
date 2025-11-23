import os
import pandas as pd


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
    print("\nExample rows (first 3 animals, first 3 non-animals):")
    print(meta.loc[is_animal, ["filename", "target", "category"]].head(3))
    print(meta.loc[~is_animal, ["filename", "target", "category"]].head(3))

#main function for histogram creating
def main():
    #make directoiries for output histograms
    paths = ensure_output_dirs("output_histograms")
    print("Created/verified:")
    for k, p in paths.items():
        print(f"  {k}: {p} (exists={os.path.isdir(p)})")
    
    #categorise sounds into animal and non-animal lists
    csv_local = "data/esc50.csv"
    categorize_sounds(csv_local, "output_histograms")

if __name__ == "__main__":
    main()
    