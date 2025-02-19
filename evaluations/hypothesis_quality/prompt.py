from typing import Dict, List

PROMPTS = {
    "clarity": {
        "system": (
            "You are an expert evaluator analyzing the clarity of scientific hypotheses.\n"
            "Your task is to evaluate how clearly and unambiguously the hypothesis is stated.\n\n"
            "Clarity Scale (1-5):\n"
            "1: Highly ambiguous - The hypothesis is presented in a highly ambiguous manner, "
            "lacking clear definition and leaving significant room for interpretation or confusion.\n\n"
            "2: Somewhat clear but vague - The hypothesis is somewhat defined but suffers from "
            "vague terms and insufficient detail, making it challenging to grasp its meaning "
            "or how it could be tested.\n\n"
            "3: Moderately clear - The hypothesis is stated in a straightforward manner, but "
            "lacks the depth or specificity needed to fully convey its nuances, assumptions, "
            "or boundaries.\n\n"
            "4: Clear and precise - The hypothesis is clearly articulated with precise terminology "
            "and sufficient detail, providing a solid understanding of its assumptions and "
            "boundaries with minimal ambiguity.\n\n"
            "5: Exceptionally clear - The hypothesis is exceptionally clear, concise, and specific, "
            "with every term and aspect well-defined, leaving no room for misinterpretation and "
            "fully encapsulating its assumptions, scope, and testability."
        ),
        "user": (
            "Evaluate the clarity of this hypothesis:\n\n"
            "Hypothesis: {hypothesis}\n\n"
            "Consider:\n"
            "- Are terms well-defined?\n"
            "- Is the relationship clearly stated?\n"
            "- Could different readers interpret it differently?\n\n"
            "Format your response as:\n"
            "Score: [1-5]\n"
            "Reasoning: [explanation]"
        )
    },
    "novelty": {
        "system": (
            "You are an expert evaluator analyzing the novelty of scientific hypotheses.\n"
            "Your task is to evaluate how novel the hypothesis is compared to existing knowledge.\n\n"
            "Novelty Scale (1-5):\n"
            "1: Not novel - The hypothesis has already been shown, proven, or is widely known, "
            "closely mirroring existing ideas without introducing any new perspectives.\n\n"
            "2: Minimally novel - The hypothesis shows slight novelty, introducing minor variations "
            "or nuances that build upon known ideas but do not offer significant new insights.\n\n"
            "3: Moderately novel - The hypothesis demonstrates moderate novelty, presenting some "
            "new perspectives or angles that provide meaningful, but not groundbreaking, avenues "
            "for exploration.\n\n"
            "4: Notably novel - The hypothesis is notably novel, offering unique nuances or "
            "perspectives that are well-differentiated from existing ideas, representing valuable "
            "and fresh contributions to the field.\n\n"
            "5: Highly novel - The hypothesis is highly novel, introducing a pioneering perspective "
            "or idea that has not been previously explored, opening entirely new directions for "
            "future research."
        ),
        "user": (
            "Evaluate the novelty of this hypothesis compared to existing knowledge:\n\n"
            "Existing Knowledge:\n{known_hypotheses}\n\n"
            "Hypothesis: {hypothesis}\n\n"
            "Format your response as:\n"
            "Score: [1-5]\n"
            "Reasoning: [explanation]"
        )
    },
    "plausibility": {
        "system": (
            "You are an expert evaluator analyzing the plausibility of scientific hypotheses.\n"
            "Your task is to evaluate if the hypothesis makes logical sense and aligns with scientific reasoning.\n\n"
            "Plausibility Scale (1-5):\n"
            "1: Not plausible - The hypothesis does not make sense at all, lacking logical or "
            "empirical grounding and failing to align with established knowledge or principles.\n\n"
            "2: Minimally plausible - The hypothesis has significant plausibility challenges, "
            "making sense in limited contexts but contradicting existing evidence or lacking "
            "coherence with established theories.\n\n"
            "3: Moderately plausible - The hypothesis makes sense overall and aligns with "
            "general principles or existing knowledge but has notable gaps or uncertainties "
            "that raise questions about its validity.\n\n"
            "4: Mostly plausible - The hypothesis is mostly plausible, grounded in logical "
            "reasoning and existing evidence, with only minor uncertainties or assumptions "
            "that could reasonably be addressed.\n\n"
            "5: Highly plausible - The hypothesis is highly plausible, fully aligning with "
            "established knowledge and logical reasoning, will likely be supported in experiments "
            "or theoretical consistency, and highly likely to be true."
        ),
        "user": (
            "Evaluate the plausibility of this hypothesis:\n\n"
            "Hypothesis: {hypothesis}\n\n"
            "Consider:\n"
            "- Does it make logical sense?\n"
            "- Are the relationships reasonable?\n"
            "- Could this be tested?\n\n"
            "Format your response as:\n"
            "Score: [1-5]\n"
            "Reasoning: [explanation]"
        )
    }
}

def format_hypotheses_list(hypotheses: List[str]) -> str:
    return "\n".join(f"- {h}" for h in hypotheses)

def get_prompt(prompt_type: str, **kwargs) -> List[Dict[str, str]]:
    """Generate prompt for specified evaluation type."""
    if prompt_type not in PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    prompts = PROMPTS[prompt_type]
    user_content = prompts["user"].format(**kwargs)
    return [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": user_content}
    ]
