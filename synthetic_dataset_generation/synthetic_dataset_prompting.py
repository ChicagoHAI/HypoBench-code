from typing import List, Tuple, Dict
import json
import os
from model import GPT4
from concurrent.futures import ThreadPoolExecutor
import weave
SELF_CONSISTENCY = 5

task_description_extension_system = """"""

task_description_extension_user = """

{task_description}


"""

task_description_extension_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_description_extension_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Additional supportive reasoning for aiding the new task description process. At most two paragraphs. "
                },
                "new_task_description": {
                    "type": "string",
                    "description": "The refined task description that includes the necessary components for synthetic dataset generation."
                }
            },
            "required": ["reasoning", "new_task_description"],
            "additionalProperties": False
        },
        "strict": True
    }
}



label_generation_system = """You are an expert in classification tasks and synthetic dataset creation for social science research. Your task is to generate a diverse and meaningful set of classification labels for a given task. The labels must be:
1. Relevant to the task and grounded in the provided context.
2. Diverse enough to cover typical, nuanced, and counterintuitive classifications.
3. Actionable for further data generation steps, ensuring they capture the complexity of social science phenomena."""

label_generation_user = """Here is the description of the classification task:

{task_description}

Your task is to generate {num_labels} classification labels relevant to this task. Ensure the labels:
1. Are specific and meaningful for the task.
2. Capture diverse aspects, including typical and counterintuitive classifications, where applicable.
3. Are clearly described to ensure they can guide the generation of features and synthetic text.

Please list the classification labels and provide a brief explanation for each.
"""

label_generation_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "label_generation_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Additional supportive reasoning for aiding the label generation process. At most two paragraphs. "
                },
                "generated_labels": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A classification label relevant to the task."
                    }
                }
            },
            "required": ["reasoning", "generated_labels"],
            "additionalProperties": False
        },
        "strict": True
    }
}

template_blank_type_generation_system = """You are a social scientist tasked with creating a synthetic dataset for the task that will be provided by the user. 
To do this, you need to define a set of phrase types that can be used as placeholders in generated textual templates. These phrase types should be relevant to the task's expression format and the provided prediction labels.

Example: As a person with [gender], I advocate [advocation] and support [opinion], where "gender", "advocation", and "opinion" are phrase types, acting as placeholders in the generated textual templates.
"""

template_blank_type_generation_user = """Here is the description of the classification task and the specified labels:
{task_description}

Labels: 
{labels}

Your task is to define {num_blanks} phrase types that can be used as placeholders in the generated textual templates following the instructions provided in the system prompt.
Please list the phrase types that are relevant to the provided task and labels.
A response format will be provided. """

template_blank_type_generation_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "template_blank_type_generation_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Additional supportive reasoning for aiding the phrase type generation process. At most two paragraphs. "
                },
                "phrase_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A phrase type relevant to the task, consisting of one or several words relevant and important for the classification task."
                    }
                }
            },
            "required": ["reasoning", "phrase_types"],
            "additionalProperties": False
        },
        "strict": True
    }
}


feature_generation_system = """You are an expert in feature engineering for social science research and synthetic dataset creation. Your task is to generate a diverse set of descriptive phrases for a given classification task with labels and a phrase type as a placeholder given from a text template. 
The generated phrases must:
1. Be relevant to the task and grounded in social science principles.
2. Be diverse and representative of the classification labels and the phrase type provided.
3. Be described in a way that facilitates the generation of synthetic text and classification.

Example: if the phrase type is "opinion" and one of the labels is "Democrat", one possible generated phrase could be "supports progressive policies".
"""

# 
feature_generation_user = """Here is the description of the classification task and the specified labels:
{task_description}

Labels: 
{labels}

Phrase Type:
{phrase_type}

Generate {num_phrases} descriptive phrases relevant to this task following the instructions provided in the system prompt.
A response format will be provided. """

feature_generation_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "feature_generation_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Additional supportive reasoning for aiding the feature generation process. At most two paragraphs. "
                },
                "generated_phrases": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A descriptive phrase relevant to the task, consisting of one or several words."
                    }
                }
            },
            "required": ["reasoning", "generated_phrases"],
            "additionalProperties": False
        },
        "strict": True
    }
}

template_generation_system = """You are a social scientist tasked with creating a synthetic dataset for the task that will be provided by the user.
To do this, you need to generate a set of textual templates that can be used to create synthetic text data. These templates should be relevant to the task's expression format and the provided prediction labels.
Each template should leave blankspaces for filling the set of phrase types provided by the user. 
Example: if the user provided phrase type "gender", the prompt template generated must have a blank space available for filling this information. 
E.g, the generated template could include components like: "As a person with [gender], ...", where "[gender]" is a placeholder for the phrase type.
You must mark all the blankspaces in the template with the phrase type provided by the user, as well as square brackets to indicate the placeholder, like the example shown above. 
"""

template_generation_user = """Here is the description of the classification task and the specified labels:
{task_description}

Labels:
{labels}

The data template must leave blankspaces for filling the set of phrase types shown below: 
{phrase_types}

Generate {num_templates} textual templates that can be used to create synthetic text data following the instructions provided in the system prompt. All templates must include placeholders for ALL the provided phrase types.
A response format will be provided. Remember to mark all the blankspaces in the template with the phrase type provided by the user and square brackets to indicate the placeholder. """

template_generation_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "template_generation_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Additional supportive reasoning for aiding the template generation process. At most two paragraphs. "
                },
                "generated_templates": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A textual template that can be used to create synthetic text data, with placeholders for the provided phrase types."
                    }
                }
            },
            "required": ["reasoning", "generated_templates"],
            "additionalProperties": False
        },
        "strict": True
    }
}

phrase_grammar_enhance_system = """You are an expert in social science research and synthetic dataset creation. Your task is to enhance the grammar of a given phrase for a specific phrase type in a generated text prompt template. 
Realizing each text template has a set of blankspaces that are marked with the type of phrase they should contain in a square bracket, you must ensure that the grammar of the phrase within the square bracket is natural with the surrounding text.
The user will provide the specific phrase type you need to enhance. You should first identify where the phrase type is located in the template, and then for the provided set of phrases with the user-specified phrase type, modify the grammar to ensure it is natural with the surrounding text.

Example: 
If a text template has this format: "I am a [gender]. I enjoy [activities]. "
And the user specified the phrase type "activities", with one of the phrases being "play the guitar", you should modify the phrase to be: "the activity of playing the guitar" to ensure the grammar is natural with the surrounding text.
E.g, when plugging in the modified phrase to the location, the sentence should read: "I enjoy the activity of playing the guitar."
You must replace the phrase into a clause to ensure the grammar is coherent with the surrounding text."""

phrase_grammar_enhance_system = """You are an expert in social science research and synthetic dataset creation. Your task is to enhance the grammar of a given phrase for a specific phrase type in a generated text prompt template. 

You will perform the operations following a series of steps. 
Step 1: remind yourself the original text template by reiterating it here, along with the explicit instruction "not to modify any single letter of the template".
Step 2: rewrite those phrases in the sentences with enhanced grammar for the specified phrase type, without changing the template.
Step 3: list those modified phrases from the sentences.
"""

phrase_grammar_enhance_user = """You are an expert in social science research and synthetic dataset creation. Your task is to enhance the grammar of a given phrase for a specific phrase type in a generated text prompt template. 

Here is the description of the classification task, the text template, and user specified phrase-type and corresponding phrases:
{task_description}

Text Template:
{text_template}
Phrase Type: {phrase_type}
Phrases: 
{phrases}

Direct plug in sentences:
{rewritten_texts}

You should output your responses step by step following the instructions below:
First, remind yourself with the text template and the instruction to never replace any components of it, including all the square brackets for parsing by explicitly stating it in your response. 
Then rewrite the above direct plug in sentences with only modifications for the phrases within the blankspace "[{phrase_type}]" to ensure grammatical coherence with the surrounding text. The modified phrases should take the form "[modified phrase]", where the square brackets are explicitly kept. You may consider using clauses. 
Your rewritten sentences must keep the square brackets of "{phrase_type}", instead of removing it and erroneously changing the surrounding text of the template for this phrase_type."""

phrase_grammar_enhance_formatting_prompt = """Below is a response generated by another LLM for the phrase grammar enhancement task.
You will be provided with a response format, and your objective is to parse the response into the given format to return critical information to the user.
You should first identify the set of phrases and list them in the reasoning section of the provided response format, then list them in the enhanced_phrases section of the response format, without additional marks such as square brackets or quotation marks.

{response}
"""

phrase_grammar_enhance_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "phrase_grammar_enhance_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Identify the phrases and its corresponding location, and provide a brief instruction for outputting the modified phrases in the correct format."
                },
                "enhanced_phrases": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A phrase with enhanced grammar for the specified phrase type. The phrase should not be a sentence, and should NOT include square brackets or quotation marks."
                    }
                }
            },
            "required": ["reasoning", "enhanced_phrases"],
            "additionalProperties": False
        },
        "strict": True
    }
}

def process_feature_str(features: List[str]) -> str:
    feature_str = ""
    for idx, feature in enumerate(features):
        feature_str += f"{idx + 1}. {feature}\n"
    return feature_str


feature_relevance_assess_system = """You are an expert in social science research and synthetic dataset creation. Your task is to determine if each trait within a set of personality traits is relevant to the provided task.
The set of traits will be provided by the user, and you must assess them based on each trait's relevance and importance for the classification task.
A response format will be provided. 
"""

feature_relevance_assess_user = """Here is the description of the classification task and the specified characteristics:

{task_description}

Below are the set of personality traits that are relevant to the task:
{features}

Your task is to provide a check on whether each of the provided features is relevant and important for the classification task, thus should be kept in the dataset."""
# features are fed in the format of the output of process_feature_input

feature_relevance_assess_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "feature_relevance_assessment_response",
        "schema": {
            "type": "object",
            "properties": {
                "relevance_array": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {
                                "type": "number",
                                "description": "Index of the feature, as indicated in the provided list of features. Realizing all indexes begin with 1, not 0!"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Supportive reasoning for the relevance assessment of the feature in one sentence."
                            },
                            "relevance": {
                                "type": "boolean",
                                "description": "Whether the feature is relevant and important to the task."
                            }
                        },
                        "required": ["feature", "reasoning", "relevance"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["relevance_array"],
            "additionalProperties": False
        },
        "strict": True
    }
}

@weave.op()
def traits_relevance_assess_self_consistency(task_description: str, features: List[str], model: GPT4, 
                                    num_samples=SELF_CONSISTENCY, temperature=1.0, top_p=0.5, **kwargs) -> Dict[str, float]:
    """
    """
    @weave.op()
    def traits_relevance_assess_single():
        assessing_prompt = [
            {
                "role": "system",
                "content": feature_relevance_assess_system
            },
            {
                "role": "user",
                "content": feature_relevance_assess_user.format(task_description=task_description, features=process_feature_str(features))
            }
        ]
        response = model.batch_generate([assessing_prompt], response_format=feature_relevance_assess_response_format, temperature=temperature, top_p=top_p)[0]
        response = json.loads(response)
        return response["relevance_array"]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(executor.map(lambda _: traits_relevance_assess_single(), range(num_samples)))
    feature_occurrences = {}
    for response in responses:
        for feature in response:
            feature_idx = feature["feature"] - 1
            if feature_idx >= len(features) or feature_idx <= 0:
                    continue
            if feature_idx not in feature_occurrences:
                feature_occurrences[feature_idx] = 0
            if feature["relevance"]:
                feature_occurrences[feature_idx] += 1
    # for each feature, if the relevance is greater than half of the samples, then it is considered relevant
    feature_relevance = {features[idx]: (occurrences / num_samples) > 0.5 for idx, occurrences in feature_occurrences.items()}
    feature_relevance_percentage = {features[idx]: (occurrences / num_samples) for idx, occurrences in feature_occurrences.items()}
    return feature_relevance, feature_relevance_percentage


