from typing import Dict, List
from hypogenic.algorithm.summary_information import SummaryInformation

def create_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def format_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)

PROMPTS = {
    "variable_matching": {
        "system": (
            "You are an expert evaluator analyzing hypotheses about relationships "
            "between input variables (features) and predicted outcomes (labels/classes).\n\n"
            "Your task is to identify when two hypotheses discuss the SAME INPUT VARIABLE(S) or FEATURE(S).\n\n"
            "Important instructions:\n"
            "- Match ONLY based on input variables/features discussed.\n"
            "- DO NOT match based on predicted outcomes, labels, or classes.\n"
            "- For hypotheses mentioning multiple variables/features, respond 'yes' if ANY input variable matches.\n"
            "- Ignore the direction, thresholds, or specific values of the relationships.\n"
            "- Predicted outcomes or labels must be completely ignored when determining matches.\n"
            "- Empty or invalid hypotheses should always return 'no'.\n\n"
            "Examples:\n\n"
            "Hypothesis A: 'Students with high math scores and 2+ publications are admitted.'\n"
            "Hypothesis B: 'Students with high math scores are rejected.'\n"
            "Return: 'yes' (matching input variable: math scores)\n\n"
            "Hypothesis A: 'Users who frequently watch science documentaries tend to be classified as science enthusiasts.'\n"
            "Hypothesis B: 'Users mentioning climate change tend to be classified as science enthusiasts.'\n"
            "Return: 'no' (the first discusses 'watching documentaries', the second discusses 'mentioning climate change'; the matching label 'science enthusiasts' is irrelevant)\n\n"
            "Hypothesis A: 'If entertainment preference is watching health-related TV shows, users are classified as Health-Conscious Eater.'\n"
            "Hypothesis B: 'Expressing enthusiasm for outdoor activities indicates health-conscious eating.'\n"
            "Return: 'no' (entertainment preference vs. outdoor activities; shared labels like Health-Conscious Eater should NOT count as matching)\n\n"
            "Responses must be exactly 'yes' or 'no'."
        ),
        "user": (
            "Determine if these two hypotheses discuss any of the same INPUT VARIABLES or FEATURES.\n\n"
            "True Hypothesis: {hyp_true}\n"
            "Generated Hypothesis: {hyp_gen}\n\n"
            "Remember:\n"
            "- DO NOT consider predicted outcomes, labels, or classes.\n"
            "- Focus ONLY on the input variables/features being discussed.\n"
            "- Ignore relationship directions, thresholds, or specific values.\n"
            "- Respond 'yes' if ANY input variable is shared; otherwise, respond 'no'.\n"
            "- Return 'no' for empty or invalid hypotheses.\n\n"
            "Response should be exactly 'yes' or 'no'."
        )
    },
    "relationship_correctness": {
        "system": (
            "You are an expert evaluator analyzing hypotheses about relationships between input variables (features) and predicted outcomes (labels/classes).\n\n"
            "Your task is to evaluate how correctly a generated hypothesis captures the relationships described by the true hypothesis.\n\n"
            "Important guidelines:\n"
            "- Evaluate BOTH the variables/features AND the direction or nature of their relationships to predicted outcomes.\n"
            "- Clearly contradictory relationships should always receive a score of 0.0.\n"
            "- For composite hypotheses (multiple conditions), assign partial scores proportionally.\n"
            "- Ignore irrelevant additional information if the main relationships and conditions are accurately captured.\n"
            "- Empty or invalid hypotheses always score 0.0.\n\n"
            "Scoring Examples:\n\n"
            "True: 'Students with A in math AND 2+ publications are admitted.'\n"
            "Generated: 'Students with A in math are admitted.'\n"
            "Score: 0.5 (captures one of two conditions)\n\n"
            "True: 'Students with A in math will be admitted.'\n"
            "Generated: 'Students with F in math will be admitted.'\n"
            "Score: 0.0 (clearly contradictory relationship)\n\n"
            "True: 'Users watching health-related shows prefer health-conscious eating.'\n"
            "Generated: 'Users who enjoy hiking prefer health-conscious eating.'\n"
            "Score: 0.5 (captures correct predicted outcome but uses incorrect variable)\n\n"
            "True: 'Users mentioning outdoor activities prefer healthy food.'\n"
            "Generated: 'Users mentioning outdoor activities prefer healthy food.'\n"
            "Score: 1.0 (perfect match)\n\n"
            "Scoring scale:\n"
            "- 1.0: Perfectly matches all variables and relationships\n"
            "- 0.75: Captures primary relationship correctly but misses minor details\n"
            "- 0.5: Partially correct (correct relationship or correct outcome, but missing important variables or conditions)\n"
            "- 0.25: Minimal correct alignment (barely relevant but somewhat aligned in intent)\n"
            "- 0.0: Incorrect, contradictory, or invalid/empty hypothesis\n"
        ),
        "user": (
            "Evaluate how correctly the generated hypothesis captures the relationships described in the true hypothesis.\n\n"
            "True Hypothesis: {hyp_true}\n"
            "Generated Hypothesis: {hyp_gen}\n\n"
            "Provide only the numerical score (0, 0.25, 0.5, 0.75, or 1.0)."
        )
    }
}

def get_variable_matching_prompt(hyp_true: str, 
                                 hyp_gen: str) -> List[Dict[str, str]]:
    """Generate prompt for variable matching evaluation."""
    # Handle empty hypothesis case
    if not hyp_gen or not hyp_true:
        return create_messages("Empty hypotheses should return no.", 'Return "no".')
        
    prompts = PROMPTS["variable_matching"]
    user_content = format_prompt(
        prompts["user"],
        hyp_true=hyp_true,
        hyp_gen=hyp_gen
    )
    return create_messages(prompts["system"], user_content)

def get_relationship_correctness_prompt(hyp_true: str, 
                                        hyp_gen: str) -> List[Dict[str, str]]:
    """Generate prompt for relationship correctness evaluation."""
    prompts = PROMPTS["relationship_correctness"]
    user_content = format_prompt(
        prompts["user"],
        hyp_true=hyp_true,
        hyp_gen=hyp_gen
    )
    return create_messages(prompts["system"], user_content)

