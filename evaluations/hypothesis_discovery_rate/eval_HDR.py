from hypogenic.algorithm.summary_information import SummaryInformation
from hypogenic.LLM_wrapper import (
    GPTWrapper,
    LLMWrapper,
    LocalVllmWrapper,
    llm_wrapper_register,
)
from hypogenic.utils import load_hypotheses
from hypogenic.logger_config import LoggerConfig
import logging
import json

from typing import Dict, List, Set, Tuple
from statistics import mean
from .prompt import get_variable_matching_prompt, get_relationship_correctness_prompt

logger_name = "HypoBench - Hypothesis Discovery Rate Evaluation"

def evaluate_pairwise_discovery(true_hyp: str, 
                                gen_hyp: str, 
                                api: LLMWrapper) -> Tuple[bool, float]:
    """Evaluate if hypotheses match and their relationship correctness."""
    logger = LoggerConfig.get_logger(logger_name)
    
    # Check if hypotheses discuss same variable
    match_messages = get_variable_matching_prompt(true_hyp, gen_hyp)
    logger.debug(f"\nComparing hypotheses:")
    logger.debug(f"True hypothesis: {true_hyp}")
    logger.debug(f"Generated hypothesis: {gen_hyp}")
    
    is_match_response = api.generate(match_messages)
    is_match = is_match_response.lower() == 'yes'
    logger.debug(f"Variable match response: {is_match_response}")
    
    if is_match:
        # Get relationship correctness score
        rel_messages = get_relationship_correctness_prompt(true_hyp, gen_hyp)
        score_response = api.generate(rel_messages)
        score = float(score_response)
        logger.debug(f"Relationship score: {score}")
        return True, score
    
    return False, 0.0

def compute_hdr(true_hypotheses: Dict[str, SummaryInformation], 
                generated_hypotheses: Dict[str, SummaryInformation],
                api: LLMWrapper) -> Dict:
    """Compute HDR metrics using hypothesis strings from the dictionaries."""
    total_true_hyps = len(true_hypotheses)
    discovered_count = 0
    all_rel_scores = []
    
    # Loop over hypothesis strings (keys) instead of SummaryInformation instances
    for true_hyp_str in true_hypotheses.keys():
        hyp_scores = []
        discovered = False
        
        # Compare with each generated hypothesis string
        for gen_hyp_str in generated_hypotheses.keys():
            is_match, score = evaluate_pairwise_discovery(true_hyp_str, gen_hyp_str, api)
            if is_match:
                discovered = True
                hyp_scores.append(score)
        
        if discovered:
            discovered_count += 1
            if hyp_scores:
                all_rel_scores.append(mean(hyp_scores))
    
    # Calculate metrics
    fdr = discovered_count / total_true_hyps if total_true_hyps else 0
    rc = mean(all_rel_scores) if all_rel_scores else 0
    hdr = fdr * rc
    
    return {
        "feature_discovery_rate": fdr,
        "relationship_correctness": rc,
        "hypothesis_discovery_rate": hdr,
        "discovered_count": discovered_count,
        "total_hypotheses": total_true_hyps
    }

def evaluate_hdr(metadata_file: str, 
                hypotheses_file: str,
                api: LLMWrapper,
                logger: logging.Logger = None) -> Dict:
    """
    Evaluate Hypothesis Discovery Rate (HDR) metrics.
    
    Args:
        metadata_file: Path to metadata JSON containing ground truth hypotheses
        hypotheses_file: Path to generated hypotheses JSON
        api: Initialized LLM wrapper
        logger: Optional logger instance
    
    Returns:
        Dictionary containing HDR metrics
    """
    if logger is None:
        logger = LoggerConfig.get_logger(logger_name)
        
    # Load metadata and create true hypotheses dict
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    true_hypotheses = {
        hyp: SummaryInformation(hyp) 
        for hyp in metadata["ground_truth_hypotheses"]
    }
    
    # Load and evaluate generated hypotheses
    generated_hypotheses = load_hypotheses(hypotheses_file)
    logger.info(f"\nEvaluating hypotheses from {hypotheses_file}")
    
    results = compute_hdr(true_hypotheses, generated_hypotheses, api)
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")
        
    return results
