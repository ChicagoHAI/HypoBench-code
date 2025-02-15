"""
This file is responsible for the pipeline of creating synthetic datasets. Realizing for this dataset, 
we might have some counterfactual statements to encourage LLM to abandon common sense and purely rely on our algorithm of exploration
to find the best approaches towards solving the given task. 

Realizing it's possible to customize the settings of the dataset to adjust difficulties, and when creating new dataset classes in the file dataset_handling.py, 
pass in "data_path" parameter to specify which json file data to load. 
"""
"""
When inferencing, we have a set of texts (e.g, tweets), and labels for the texts for a corresponding task (e.g, classification). 
The objective is to come up with a set of textual features that are useful to determine the labels. 
When generating the dataset, therefore, we should define the task and its corresponding labels, and come up with ground truth feature sets, 
then generate texts based on the feature presence setting. 
"""

TEST_TASK_DESCRIPTION_ELECTION = """Given a set of political tweets from a person, determine the vote this person would give in the upcoming US election. 
The advocations from Democrats: "minimum wage increase", "universal healthcare", "gun control", "climate change", "LGBTQ rights".
The advocations from Republicans: "tax cut", "border security", "pro-life", "2nd amendment", "free market"."""

TEST_TASK_DESCRIPTION_TWEET_VOTE_PREFERENCE = """"""

MAX_CONCURRENT = 20
MAX_RETRIES = 3
"""
The pipeline should include the following: 
1. Extending a user-provided task description to be more detailed and covering necessary components for constructing a dataset
2. A pipeline for generating labels and features given the task description
3. A pipeline to generate tweets based on the features and labels
4. A dataset constructor that integrates all generated contents into manageable dataset objects
5. A relationship modelling that determines which kind of feature combination should lead to which label 
    (e.g, suppose we have 5 features and 2 labels, then we would have 32 possible feature enumerations, and each one should be assigned to one label. )
5. A comprehensive pipeline that combines all above components
"""
from model import GPT4

from .synthetic_dataset_prompting import *
import os
import json
import random
from typing import List, Dict, Tuple
import weave
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from copy import copy

class SyntheticDatasetGenerator:
    def __init__(self, init_task_description: str, output_path: str, task_name: str, 
                 num_templates: int, num_phrases: int, num_gaps_per_template: Tuple[int, int],# lower and upper bound
                 num_labels: int, 
                 feature_generation_model: GPT4, text_generation_model: GPT4,
                 **kwargs):
        self.output_path = output_path
        os.mkdir(os.path.dirname(output_path)) if not os.path.exists(os.path.dirname(output_path)) else None
        self.clf_model = LogisticRegression(multi_class="multinomial")  # realizing what we need is a randomly defined model, providing data and training might lead to undesirable models due to noisy data. 
        self.num_templates = num_templates
        self.task_name = task_name
        self.num_gaps_per_template = num_gaps_per_template
        self.num_phrases = num_phrases
        self.feature_generation_model = feature_generation_model
        self.text_generation_model = text_generation_model
        self.num_labels = num_labels
        self.task_description = init_task_description
        self.labels = []
        
        """
        We will ask LLMs to generate a set of templates, with gaps within each of the template, and the set of phrases that could be filled in the gaps.
        These phrases will all have a set of common properties that could be grouped together (they all express the same thing, but adjusted grammatically to fill those gaps), 
        and we will evaluate if each of the grammatically enhanced phrases are consistent with their prototype phrases.

        Labels will be assigned in the same manner using the logistic regression model, applied on prototype phrases. 

        """

        self.features = {}  # dict[list]; keys are the prototype phrases, and values are the set of phrases for each template. 
        self.templates = {} # dict[dict[list]]; keys are the template strings, values are dict of the prototype phrases, and their corresponding set of phrases enhanced for grammatical variations.
        self.dataset = None
        

    @weave.op()
    def task_description_extension(self, task_description: str = "") -> str:
        """
        Extend the user-provided task description to include more detailed information about the task. 
        """
        if task_description == "":
            task_description = self.task_description
        prompt = [
            {"role": "system", "content": task_description_extension_system},
            {"role": "user", "content": task_description_extension_user.format(task_description=task_description)}
        ]
        response = self.feature_generation_model.batch_generate([prompt], response_format=task_description_extension_response_format)[0]
        response_json = json.loads(response)
        self.task_description = response_json["new_task_description"]
        return response_json["reasoning"], response_json["new_task_description"]
    
    @weave.op()
    def generate_labels(self, task_description: str = "", **kwargs):
        """
        Generate labels for the dataset. 
        """
        if task_description == "":
            task_description = self.task_description
        prompt = [
            {"role": "system", "content": label_generation_system},
            {"role": "user", "content": label_generation_user.format(task_description=task_description, num_labels=self.num_labels)}
        ]
        response = self.feature_generation_model.batch_generate([prompt], response_format=label_generation_response_format, temperature=1.0, top_p=0.2)[0]
        response_json = json.loads(response)
        self.labels.extend(response_json["generated_labels"])
        return response_json["reasoning"], response_json["generated_labels"]


    @weave.op()
    def generate_template_blank_types(self, task_description: str = "", labels: List[str] = [], 
                                    **kwargs):
        """
        Generate a set of phrase types for blanks of generated twitter templates. 
        These phrase types will help generating templates that intentionally blank out these phrases, and allowing 
        future replacements. 
        """
        if task_description == "":
            task_description = self.task_description
        if len(labels) == 0:
            labels = self.labels
        labels_str = ", ".join(labels)
        num_blanks = " to ".join([str(number) for number in self.num_gaps_per_template])
        prompt = [
            {"role": "system", "content": template_blank_type_generation_system},
            {"role": "user", "content": template_blank_type_generation_user.format(task_description=task_description, labels=labels_str, num_blanks=num_blanks)}
        ]
        response = self.feature_generation_model.batch_generate([prompt], response_format=template_blank_type_generation_response_format, 
                                                                temperature=1.0, top_p=0.2)[0]
        response_json = json.loads(response)
        for phrase_type in response_json["phrase_types"]:
            self.features[phrase_type] = []
        return response_json["reasoning"], response_json["phrase_types"]

    @weave.op()
    def generate_features(self, task_description: str = "", labels: List[str] = [], 
                          phrase_types: List[str] = [], **kwargs):
        if task_description == "":
            task_description = self.task_description
        if len(labels) == 0:
            labels = self.labels
        if len(phrase_types) == 0:
            phrase_types = list(self.features.keys())
        labels_str = ", ".join(labels)
        # use parallel processing across phrase types
        parallel_prompts = [
            [
                {"role": "system", "content": feature_generation_system},
                {"role": "user", "content": feature_generation_user.format(task_description=task_description, labels=labels_str, phrase_type=phrase_type, num_phrases=self.num_phrases)}
            ] for phrase_type in phrase_types
        ]
        parallel_inputs = [{"prompt_list": [prompt], "response_format": feature_generation_response_format, "temperature": 1.0, "top_p": 0.5} for prompt in parallel_prompts]
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            responses = list(executor.map(lambda inputs: self.feature_generation_model.batch_generate(**inputs), parallel_inputs))
        all_reasoning = []
        for i, response in enumerate(responses):
            response_json = json.loads(response[0])
            self.features[phrase_types[i]].extend(response_json["generated_phrases"])
            all_reasoning.append(response_json["reasoning"])
            
        return all_reasoning, self.features

    @weave.op()
    def generate_templates(self, task_description: str = "", labels: List[str] = [], 
                            phrase_types: List[str] = [], phrases: Dict[str, List[str]] = {},
                           **kwargs):
        """
        To generate templates, we need to at least know the task description, and how many templates to generate. 

        The templates will be generated in the following manner:
        
        """
        if task_description == "":
            task_description = self.task_description
        if len(labels) == 0:
            labels = self.labels
        if len(phrase_types) == 0:
            phrase_types = list(self.features.keys())
        if len(phrases) == 0:
            phrases = self.features
        labels_str = ", ".join(labels)
        phrase_types_str = process_feature_str(phrase_types)
        template_generation_prompt = [
            {"role": "system", "content": template_generation_system},
            {"role": "user", "content": template_generation_user.format(task_description=task_description, 
                                                                        labels=labels_str, 
                                                                        num_templates=self.num_templates, 
                                                                        phrase_types=phrase_types_str,)}
        ]
        # implement checking procedure
        phrase_types_str_list = [f"[{phrase_type}]" for phrase_type in phrase_types]
        success = False
        response = self.feature_generation_model.batch_generate([template_generation_prompt], response_format=template_generation_response_format, 
                                                                temperature=1.0, top_p=0.2)[0]
        response_json = json.loads(response)

        self.templates = {template: {prototype: [] for prototype in phrases} for template in response_json["generated_templates"]}
        # check if each template contains all the elements in the phrase_types_str_list
        for template in response_json["generated_templates"]:
            if not all(phrase_type in template for phrase_type in phrase_types_str_list):
                del self.templates[template]
        # could consider performing a saving operation here.
        with open(f"{os.path.dirname(self.output_path)}/metadata.json", "w") as f:
            json.dump({"task_name": self.task_name, "task_description": task_description, "labels": labels, 
                       "num_templates": self.num_templates, "num_gap_per_template": self.num_gaps_per_template, 
                       "num_labels": self.num_labels, "num_phrases": self.num_phrases,
                       "phrase_types": phrase_types, "features": phrases, "templates": self.templates
                       }, f)
        return response_json["reasoning"], self.templates

    @weave.op()
    def phrase_grammatical_enhancement(self, task_description: str = "", labels: List[str] = [], 
                                    phrase_types: List[str] = [], phrases: Dict[str, List[str]] = {}, 
                                    templates: Dict[str, Dict[str, List[str]]] = {}, **kwargs):
        """
        Enhance the phrases for each template to be more grammatically diverse. 
        
        The goal is: for each 
        """
        if task_description == "":
            task_description = self.task_description
        if len(labels) == 0:
            labels = self.labels
        if len(phrase_types) == 0:
            phrase_types = list(self.features.keys())
        if len(phrases) == 0:
            phrases = self.features
        if len(templates) == 0:
            templates = self.templates
        self.task_description = task_description
        self.labels = labels
        self.features = phrases
        self.templates = templates
        labels_str = ", ".join(labels)
        # will perform parallel processing across templates and phrase types. 
        parallel_prompts = [
            [
                {"role": "user", "content": phrase_grammar_enhance_user.format(task_description=task_description, 
                                                                               phrase_type=phrase_type,
                                                                               phrases=phrases[phrase_type],
                                                                               text_template=template,
                                                                               rewritten_texts="\n\n".join([template.replace(f"{phrase_type}", phrase) for phrase in phrases[phrase_type]]
                                                                                 )
                                                                               )}
            ] for template in templates for phrase_type in phrase_types
        ]
        parallel_inputs = [{"prompt_list": [prompt], "response_format": None} for prompt in parallel_prompts]
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            reasoning_responses = list(executor.map(lambda inputs: self.text_generation_model.batch_generate(**inputs), parallel_inputs))
        formatting_prompts = [
            [
                {"role": "user", "content": phrase_grammar_enhance_formatting_prompt.format(response=response[0])}
            ] for response in reasoning_responses
        ]
        formatting_inputs = [{"prompt_list": [prompt], "response_format": phrase_grammar_enhance_response_format} for prompt in formatting_prompts]
        formatting_llm = GPT4()
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            responses = list(executor.map(lambda inputs: formatting_llm.batch_generate(**inputs), formatting_inputs))

        all_reasoning = []
        # update the templates; realizing the parallel-input orders are: [template1, phrase_type1], [template1, phrase_type2], ..., [template2, phrase_type1], [template2, phrase_type2], ...
        # Should be very careful here.
        for i, response in enumerate(responses):
            response_json = json.loads(response[0])
            template = list(templates.keys())[i // len(phrase_types)]
            phrase_type = phrase_types[i % len(phrase_types)]
            # remove square brackets if they exist
            enhanced_phrases = [phrase.replace("[", "").replace("]", "") for phrase in response_json["enhanced_phrases"]]
            self.templates[template][phrase_type].extend(enhanced_phrases)
            all_reasoning.append(reasoning_responses[i][0])
        # save templates
        with open(f"{os.path.dirname(self.output_path)}/metadata.json", "w") as f:
            json.dump({"task_description": task_description, "labels": labels, 
                       "task_name": self.task_name,
                       "num_templates": self.num_templates, "num_gap_per_template": self.num_gaps_per_template, 
                       "phrase_types": phrase_types, "features": phrases, "templates": self.templates}, f)
            
        return all_reasoning, self.templates

    def initialize_model(self):
        """
        The regression model should be initialized so that: 
        for each template and each phrase, they all have a unique score. 
        Realizing for each text, only one template and one phrase from each phrase type would be selected. 

        TODO: be ready for providing more hyperparameters to initialize the model. 
        """
        regression_feature_num = len(self.templates) + sum(len(phrases) for phrases in self.features.values())
        """
        We will explicitly let the first few features to be the template features, 
        and the rest to be the phrase features, following the order of how they are arranged 
        in self.features dict. 
        """
        # initialize the model, with the set of classes mapping to self.labels
        self.clf_model = LogisticRegression(multi_class="multinomial", max_iter=1000)
        self.clf_model.coef_ = np.random.uniform(-5.0, 5.0, (len(self.labels), regression_feature_num))
        self.clf_model.intercept_ = np.random.uniform(-1.0, 1.0, len(self.labels))
        self.clf_model.classes_ = np.array(self.labels)
        # save model
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "clf_model.pkl"), "wb") as f:
            pickle.dump(self.clf_model, f)
        return self.clf_model

    def initialize_from_metadata(self, metadata_path: str):
        metadata = json.load(open(metadata_path, "r"))
        self.task_description = metadata["task_description"]
        self.labels = metadata["labels"]
        self.num_templates = metadata["num_templates"]
        self.num_gaps_per_template = metadata["num_gap_per_template"]
        self.features = metadata["features"]
        self.templates = metadata["templates"]
        self.task_name = metadata["task_name"]
        self.dataset = None

    def construct_text_from_templates(self, templates: Dict[str, Dict[str, List[str]]], clf_model = None):
        """
        Construct texts from the templates. 
        """
        if clf_model is None:
            clf_model = self.initialize_model()
        if len(templates) == 0:
            templates = self.templates
        templates = self.templates
        """
        In principle, we should try to enumerate all the feature combinations. E.g,
        suppose we have phrase types [A, B], and each phrase type has 3 phrases, then we would have 9, or 3^2, possible feature combinations.
        One approach is to design a recursive function that generates all the possible feature combinations, and then assign labels to each of them.
        In principle, the recursion should have a for-loop inside, and within the for-loop one of the phrase types is selected, and then the recursion is called again on 
        the remaining set of selectable phrase types.

        This method should initilize the recursive data structure, and then call the recursive function to generate all the possible feature combinations.
        It's possible to have a phrase_type list, and have an index to indicate which phrase type is currently being selected. The recursion should focus on 
        [curr_idx:] phrase types to avoid redundancy.
        the data structure will be a list with shape 1 + len(phrase_types), and each element is a list of phrases (1 is the template), 
        and the range of each element would be from zero to len(phrases) - 1.

        The recursive operation should take in this list and generate a label according to the initialized clf_model. 
        """
        # construct the recursive data structure
        feature_array = [0] + [0 for _ in range(len(self.features))]
        generated_texts = [] # a list of dict, containing vital information to be converted into a .csv file.
        # after a careful consideration, will have a for-loop here to control the template str
        for i in range(len(templates)):
            feature_array[0] = i
            self.construct_template_recursive(templates, list(templates.keys())[i], feature_array, 1, generated_texts)
        # save the generated texts
        with open(self.output_path, "w") as f:
            json.dump(generated_texts, f)
        return generated_texts
    
    
    def construct_template_recursive(self, templates, template_str, feature_array: List[int], curr_idx, generated_texts):
        """
        The recursive function to construct the template. 
        
        recursion return condition: curr_idx == len(self.features); 
            generated_texts should have a dictionary added, and with a label from the clf_model assigned to it. 
        If not in return condition: 
            a for-loop that constructs a new feature_array based on the passed in one, and increment the curr_idx by 1. 
            then call the recursion on the new feature_array.
        """        
        if curr_idx > len(self.features):
            # construct the text
            text = copy(template_str)
            phrase_types_arr = feature_array[1:]  # a list of numbers
            # print(feature_array) # for debugging only
            # perform replacement. 
            template_specific_features = templates[template_str] # Dict[str, List[str]]
            """
            conditions to check: 
            len(phrase_types) == len(template_specific_features) == len(feature_array) - 1
            phrase_types[j] < len(template_specific_features[jth_key]); if not, need to abandon this modification, e.g, do nothing. 
            """
            phrase_type_keys = [key for key in template_specific_features.keys()] # list of phrase type keys
            if not all(phrase_type < len(template_specific_features[phrase_type_keys[i]]) for i, phrase_type in enumerate(phrase_types_arr)):
                return # do nothing
            # perform text modification; 
            for index in range(len(phrase_types_arr)):
                curr_type = phrase_type_keys[index]
                curr_phrase_idx = phrase_types_arr[index]
                curr_phrase = template_specific_features[curr_type][curr_phrase_idx]
                text = text.replace(f"[{curr_type}]", curr_phrase)
            # assign label
            """
            recall the clf_model we've constructed has size (len(self.labels), len(self.templates) + sum(len(phrases) for phrases in self.features.values())). 
            We should construct an input array with size (1, len(self.templates) + sum(len(phrases) for phrases in self.features.values())). 
            Handle templates and phrases one by one. 
            """
            input_array = np.zeros((1, len(self.templates) + sum(len(phrases) for phrases in self.features.values())))
            # handle templates
            input_array[0, feature_array[0]] = 1
            blank_features = [self.features[key][feature_array[1:][idx]] for idx, key in enumerate(phrase_type_keys)] # this outputs the set of selected features in the form of a text. 
            # handle phrases respectively; need to know the starting point of each phrase type in the input_array. 
            curr_idx = len(self.templates)
            
            for k, phrase_idx in enumerate(phrase_types_arr):
                input_array[0, curr_idx + phrase_idx] = 1
                curr_idx += len(self.features[phrase_type_keys[k]])
            # assign label
            label = self.clf_model.predict(input_array)
            generated_texts.append({
                "text": text,
                "template": template_str,
                "blank_features": blank_features,
                "feature_idx_array": feature_array,
                "np_input_array": input_array.tolist(), 
                "label": str(label[0]).lower(),
            })
            return
        else:
            curr_phrase_type = list(self.features.keys())[curr_idx - 1]
            for i in range(len(self.features[curr_phrase_type])):
                new_feature_array = copy(feature_array)
                new_feature_array[curr_idx] = i
                self.construct_template_recursive(templates, template_str, new_feature_array, curr_idx + 1, generated_texts)
            return

    def convert_to_huggingface(self, data_path: str = "", task_name: str = "", train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        if data_path == "":
            data_path = self.output_path
        if task_name == "":
            task_name = self.task_name
        data = json.load(open(data_path, "r"))
        # shuffle the data
        random.shuffle(data)
        # split the data
        train_data = data[:int(len(data) * train_val_test_ratio[0])]
        val_data = data[int(len(data) * train_val_test_ratio[0]): int(len(data) * (train_val_test_ratio[0] + train_val_test_ratio[1]))]
        test_data = data[int(len(data) * (train_val_test_ratio[0] + train_val_test_ratio[1])):]
        """
        to convert into hugging face format, it should be a dict with keys being variable names, and values being lists of data.
        key_names = data[0].keys()
        hf_data = {key: [] for key in key_names}
        for entry in data:
            for key in key_names:
                hf_data[key].append(entry[key])
        
        with open(f"{os.path.dirname(data_path)}/{os.path.basename(data_path)}", "w") as f:
            json.dump(hf_data, f)
        return hf_data
        
        """
        tmp_dict = {"train": train_data, "val": val_data, "test": test_data}
        for set_name, set_data in tmp_dict.items():
            hf_data = {key: [] for key in set_data[0].keys()}
            for entry in set_data:
                for key in hf_data.keys():
                    hf_data[key].append(entry[key])
            with open(f"{os.path.dirname(data_path)}/{task_name}_{set_name}.json", "w") as f:
                json.dump(hf_data, f)
    