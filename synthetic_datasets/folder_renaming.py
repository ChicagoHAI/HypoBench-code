import os
outer_folder = "HypoBench-code/synthetic_datasets/election_new"
for folder in os.listdir(outer_folder):
    if os.path.isdir(f"{outer_folder}/{folder}"):
        print(folder)
        new_folder_name = str(int(folder[0]) * 5) + folder[1:]
        os.rename(f"{outer_folder}/{folder}", f"{outer_folder}/{new_folder_name}")
