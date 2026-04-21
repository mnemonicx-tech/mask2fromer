import json
import random
import os

def create_mini_val(
    input_json="/ephemeral/training_data/annotations/instances_val.json",
    output_json="/ephemeral/training_data/annotations/instances_val_mini.json",
    sample_size=5000,
    seed=42
):
    print(f"Loading {input_json} (this might take a minute for 1.4GB)...")
    with open(input_json, "r") as f:
        data = json.load(f)

    # Set random seed for reproducibility
    random.seed(seed)

    print(f"Found {len(data['images'])} total images.")
    
    # 1. Sample 5000 random images
    sampled_images = random.sample(data["images"], min(sample_size, len(data["images"])))
    
    # 2. Extract their IDs for fast lookup
    valid_image_ids = {img["id"] for img in sampled_images}
    print(f"Sampled {len(valid_image_ids)} images. Filtering annotations...")

    # 3. Keep only the annotations that belong to the sampled images
    sampled_annotations = [
        ann for ann in data["annotations"] 
        if ann["image_id"] in valid_image_ids
    ]
    print(f"Retained {len(sampled_annotations)} annotations.")

    # 4. Construct the new mini JSON
    mini_data = {
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": data["categories"],
        # Retain info and licenses if they exist
        "info": data.get("info", {}),
        "licenses": data.get("licenses", [])
    }

    # 5. Save the file
    print(f"Writing to {output_json}...")
    with open(output_json, "w") as f:
        json.dump(mini_data, f)
        
    print(f"✅ Success! Created {output_json}")
    print(f"Mini dataset size: {os.path.getsize(output_json) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_mini_val()
