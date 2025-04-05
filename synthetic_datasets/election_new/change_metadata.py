"""
for each metadata.json file under folders of the current folder, 
rename the "gt_hypotheses" into "ground_truth_hypotheses"
"""
import os
import json

# current directory the file is in
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")
# list all folders in the current directory
folders = [f for f in os.listdir(current_dir) if os.path.isdir(f"{current_dir}/{f}")]
# for each folder
for folder in folders:
    # check if metadata.json exists
    metadata_path = os.path.join(current_dir, folder, "metadata.json")
    if os.path.exists(metadata_path):
        # load the metadata.json file
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # rename the "gt_hypotheses" into "ground_truth_hypotheses"
        if "gt_hypotheses" in metadata:
            metadata["ground_truth_hypotheses"] = metadata.pop("gt_hypotheses")
            # save the modified metadata.json file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    # print the folder name
    print(f"Processed folder: {folder}")
# Note: This script assumes that the metadata.json file is in the format of a dictionary.
# If the format is different, the script may need to be modified accordingly.