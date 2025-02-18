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
            "between variables and target labels.\n\n"
            "Your task is to identify when hypotheses discuss the same variables "
            "or features, regardless of the relationship they describe.\n"
            "For composite hypotheses with multiple variables, respond 'yes' if "
            "ANY of the variables match.\n\n"
            "Examples:\n"
            '"Students with high math scores and 2+ publications are admitted" and '
            '"Students with high math scores are rejected"\n'
            'should return "yes" because they both discuss math scores.\n\n'
            '"Students with high math scores and 2+ publications are admitted" and '
            '"Students with 3+ publications are admitted"\n'
            'should return "yes" because they both discuss publications.\n\n'
            'Empty or invalid hypotheses should always return "no".'
        ),
        "user": (
            "Determine if these two hypotheses discuss any of the same variables/features, "
            "regardless of what they say about them.\n\n"
            "True Hypothesis: {hyp_true}\n"
            "Generated Hypothesis: {hyp_gen}\n\n"
            "Focus ONLY on what variables/features are being discussed:\n"
            "- Ignore the direction of relationships (positive/negative)\n"
            "- Ignore thresholds or specific values\n"
            "- Ignore whether one is more specific than the other\n"
            "- For composite hypotheses (with multiple variables), match if ANY variable is shared\n"
            "- Return 'no' for empty or invalid hypotheses\n\n"
            'Response should be exactly "yes" or "no".'
        )
    },
    "relationship_correctness": {
        "system": (
            "You are an expert evaluator analyzing hypotheses about relationships "
            "between variables and target labels.\n"
            "Your task is to evaluate how correctly a generated hypothesis captures "
            "the relationships described in the true hypothesis.\n"
            "For composite hypotheses, partial matches should receive partial scores.\n\n"
            "Examples:\n"
            "True: 'Students with A in math AND 2+ publications are admitted'\n"
            "Generated: 'Students with A in math are admitted'\n"
            "Score: 0.5 (captures one of two conditions)\n\n"
            "True: 'Students with A in math will be admitted'\n"
            "Generated: 'Students with F in math will be admitted'\n"
            "Score: 0.0 (contradictory relationship)\n\n"
            "Empty or invalid hypotheses should always receive a score of 0.0"
        ),
        "user": (
            "Rate how correctly the Generated Hypothesis captures the relationship "
            "compared to the True Hypothesis.\n\n"
            "True Hypothesis: {hyp_true}\n"
            "Generated Hypothesis: {hyp_gen}\n\n"
            "Rate from 0 to 1:\n"
            "- 1.0: Perfectly captures all relationships and conditions\n"
            "- 0.75: Captures main relationship but missing minor conditions\n"
            "- 0.5: Captures some conditions correctly (e.g., one out of two conditions)\n"
            "- 0.25: Slightly related but mostly incomplete or imprecise\n"
            "- 0.0: Wrong relationship, contradictory, or empty/invalid hypothesis\n\n"
            "Provide only the numerical score (0-1)."
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

