from typing import Dict, List, Tuple
from statistics import mean
import logging
import json
from hypogenic.utils import load_hypotheses
from hypogenic.logger_config import LoggerConfig
from hypogenic.LLM_wrapper import LLMWrapper
from .prompt import get_prompt, format_hypotheses_list  # Added missing import

logger_name = "HypoBench - Hypothesis Quality"

def evaluate_aspect(hypothesis: str,
                   known_hypotheses: List[str],
                   aspect: str,
                   api: LLMWrapper,
                   logger: logging.Logger) -> Tuple[float, str]:
    """Evaluate a single aspect (clarity, novelty, or plausibility) of a hypothesis."""
    logger.debug(f"\nEvaluating {aspect} for hypothesis: {hypothesis}")
    
    kwargs = {"hypothesis": hypothesis}
    if aspect == "novelty":
        kwargs["known_hypotheses"] = format_hypotheses_list(known_hypotheses)
        
    messages = get_prompt(aspect, **kwargs)
    response = api.generate(messages)
    logger.debug(f"Response: {response}")
    
    try:
        # More robust parsing
        score = None
        reasoning = ""
        
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith('score:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except ValueError:
                    continue
            elif line.lower().startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if score is None:
            raise ValueError("Could not find valid score in response")
            
        if not (1 <= score <= 5):
            raise ValueError(f"Score {score} not in range 1-5")
            
        logger.info(f"{aspect.capitalize()} score: {score}")
        logger.debug(f"Reasoning: {reasoning}")
        return score, reasoning
        
    except Exception as e:
        logger.error(f"Failed to parse {aspect} response: {response}")
        logger.error(f"Error: {str(e)}")
        return 1.0, f"Failed to parse {aspect} response"

def evaluate_single_hypothesis(hypothesis: str,
                             known_hypotheses: List[str],
                             api: LLMWrapper,
                             logger: logging.Logger) -> Dict:
    """Evaluate a single hypothesis on all quality metrics."""
    aspects = ["clarity", "novelty", "plausibility"]
    scores = {}
    reasoning = {}
    
    for aspect in aspects:
        score, reason = evaluate_aspect(hypothesis, known_hypotheses, aspect, api, logger)
        scores[aspect] = score
        reasoning[aspect] = reason
    
    return {
        "hypothesis": hypothesis,
        "scores": scores,
        "reasoning": reasoning,
        "overall_score": mean(scores.values())
    }

def evaluate_quality(metadata_file: str,
                    hypotheses_file: str,
                    api: LLMWrapper,
                    logger: logging.Logger = None) -> Dict:
    """Evaluate quality of all generated hypotheses."""
    if logger is None:
        logger = LoggerConfig.get_logger(logger_name)
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    known_hypotheses = metadata["known_hypotheses"]
    generated_hypotheses = load_hypotheses(hypotheses_file)
    
    logger.info(f"\nEvaluating quality of hypotheses from {hypotheses_file}")
    
    results = {
        "hypotheses_details": [],
        "num_known": len(known_hypotheses),
        "num_generated": len(generated_hypotheses)
    }
    
    # Evaluate each hypothesis
    for gen_hyp in generated_hypotheses.keys():
        eval_result = evaluate_single_hypothesis(gen_hyp, known_hypotheses, api, logger)
        results["hypotheses_details"].append(eval_result)
    
    # Compute aggregate metrics
    for metric in ["clarity", "novelty", "plausibility"]:
        avg = mean(h["scores"][metric] for h in results["hypotheses_details"])
        results[f"avg_{metric}"] = avg
        logger.info(f"Average {metric}: {avg:.2f}")
    
    avg_overall = mean(h["overall_score"] for h in results["hypotheses_details"])
    results["avg_overall"] = avg_overall
    logger.info(f"Average overall score: {avg_overall:.2f}")
    
    return results
