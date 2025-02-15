from synthetic_dataset_generation import *

import os

"""
Given a tweet, predict the person's voting preferences for 2024 US election. 

Make the above task description become as specific as possible, which will be used for generating a synthetic dataset with classification labels. The new description should be a new paragraph containing nothing about the dataset (including the purpose of the paragraph is for constructing a dataset, and how the dataset should be constructed). 
"""


def terminal_dataset_gen(args):
    task_description = open(f"dataset/{args.task}/task_description.txt", "r").read()

    dataset_generator = SyntheticDatasetGenerator(task_description, f"dataset/{args.task}/test_dataset.json", 
                                                args.task,
                                                num_templates=4, num_phrases=5, num_gaps_per_template=(3, 4), num_labels=3, 
                                                feature_generation_model=GPT4(model_name="gpt-4o"), text_generation_model=GPT4(model_name="gpt-4o"),
    )
    # reasoning, labels = dataset_generator.generate_labels()
    # print("finished generating labels")
    # dataset_generator.generate_template_blank_types()
    # print("finished generating template blank types")
    # dataset_generator.generate_features()
    # print("finished generating features")
    # dataset_generator.generate_templates()
    # print("finished generating templates")
    metadata = json.load(open(f"dataset/{args.task}/metadata.json", "r"))
    task_description = metadata["task_description"]
    labels = metadata["labels"]
    num_templates = metadata["num_templates"]
    phrase_types = metadata["phrase_types"]
    features = metadata["features"]
    templates = metadata["templates"]
    # reasonings, templates = dataset_generator.phrase_grammatical_enhancement(
    #     task_description=task_description, labels=labels, phrase_types=phrase_types, phrases=features, templates=templates
    # )
    print("finished generating grammatically enhanced phrases")
    dataset_generator.initialize_from_metadata(f"dataset/{args.task}/metadata.json")
    generated_texts = dataset_generator.construct_text_from_templates(templates)  # if need to have a new clf_model, just run this line and subsequent ones. 
    print("finished generating texts")
    dataset_generator.convert_to_huggingface(task_name=args.task)  # [text, label]
    print("finished converting to huggingface")
    a = 3

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset for a given task")
    parser.add_argument("--task", type=str, help="The task for which the dataset is to be generated", required=False, default="synthetic_election")
    # args = parser.parse_args()
    args = parser.parse_args(["--task", "synthetic_preference"])
    terminal_dataset_gen(args)