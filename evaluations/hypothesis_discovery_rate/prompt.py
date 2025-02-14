import json

def build_variable_discovery_prompt(gold_variables, generated_hypothesis):
    prompt = f"""\
You are an assistant that evaluates discovered features in a generated hypothesis.
Gold (true) variables: {gold_variables}
Generated hypothesis: {generated_hypothesis}
Please list out the variables from the generated hypothesis in a JSON list, e.g.:
["A", "B", "C"]
No extra text.
"""
    return prompt

def build_relationship_correctness_prompt(gold_relationships, gen_relationships):
    prompt = f"""\
You are an assistant that rates how closely a generated relationship matches the gold relationship.
Gold relationships: {gold_relationships}
Generated relationships: {gen_relationships}
Return a numeric score between 0.0 and 1.0 for each discovered feature, summarizing correctness.
Return data in JSON format, e.g.:
{{
  "scores": [0.75, 0.5, 1.0]
}}
"""
    return prompt

def build_variable_discovery_with_model_prompt(gold_hypothesis, generated_hypothesis):
    """
    Create a prompt that asks a model to output how many features overlap
    in a subtle or non-obvious way between the gold hypothesis and the generated hypothesis.
    """
    prompt = f"""\
You are an AI assistant that determines subtle feature matches between a gold hypothesis and a generated hypothesis.
Gold hypothesis: {gold_hypothesis}
Generated hypothesis: {generated_hypothesis}
Please return a JSON object with:
{{
  "gold_feature_count": number of unique features in the gold hypothesis,
  "matched_feature_count": number of discovered gold features in the generated hypothesis
}}
No extra text.
"""
    return prompt
