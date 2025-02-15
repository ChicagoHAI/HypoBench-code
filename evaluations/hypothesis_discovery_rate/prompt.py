from typing import Dict, List
from hypogenic.algorithm.summary_information import SummaryInformation

# System prompt for all evaluations
SYSTEM_PROMPT = """You are an expert evaluator of hypotheses about relationships between variables and target labels. 
Your task is to carefully analyze hypotheses and provide precise evaluations."""

# Templates for different evaluation tasks
VARIABLE_MATCHING_TEMPLATE = """Compare these two hypotheses and determine if they discuss the same variable or concept:

True Hypothesis: {hyp_true}
Generated Hypothesis: {hyp_gen}

Response should be exactly 'yes' or 'no'."""

RELATIONSHIP_CORRECTNESS_TEMPLATE = """Rate how correctly the Generated Hypothesis captures the relationship compared to the True Hypothesis.

True Hypothesis: {hyp_true}
Generated Hypothesis: {hyp_gen}

Rate from 0 to 1:
- 1.0: Perfectly captures the relationship
- 0.75: Mostly correct but missing some nuance
- 0.5: Partially correct
- 0.25: Slightly correct but mostly wrong
- 0.0: Completely wrong or opposite relationship

Provide only the numerical score (0-1)."""

def create_messages(content: str) -> List[Dict[str, str]]:
    """Create a standard message format with system and user messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]

def format_prompt(template: str, hyp_true: str, hyp_gen: str) -> str:
    """Format a template with true and generated hypotheses."""
    return template.format(hyp_true=hyp_true, hyp_gen=hyp_gen)

def get_variable_matching_prompt(hyp_true: SummaryInformation, hyp_gen: SummaryInformation) -> List[Dict[str, str]]:
    """Generate prompt for variable matching evaluation."""
    content = format_prompt(
        VARIABLE_MATCHING_TEMPLATE,
        hyp_true.hypothesis,
        hyp_gen.hypothesis
    )
    return create_messages(content)

def get_relationship_correctness_prompt(hyp_true: SummaryInformation, hyp_gen: SummaryInformation) -> List[Dict[str, str]]:
    """Generate prompt for relationship correctness evaluation."""
    content = format_prompt(
        RELATIONSHIP_CORRECTNESS_TEMPLATE,
        hyp_true.hypothesis,
        hyp_gen.hypothesis
    )
    return create_messages(content)

