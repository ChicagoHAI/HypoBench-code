from hypogenic.algorithm.summary_information import SummaryInformation
from typing import Dict, List, Set, Tuple
from statistics import mean
from .prompt import get_variable_matching_prompt, get_relationship_correctness_prompt

def evaluate_pairwise_discovery(true_hyp: SummaryInformation, 
                                gen_hyp: SummaryInformation, 
                                api) -> Tuple[bool, float]:
    """Evaluate if hypotheses match and their relationship correctness."""
    # Check if hypotheses discuss same variable
    match_messages = get_variable_matching_prompt(true_hyp, gen_hyp)
    is_match = api.generate(match_messages).lower() == 'yes'
    
    if is_match:
        # Get relationship correctness score
        rel_messages = get_relationship_correctness_prompt(true_hyp, gen_hyp)
        score = float(api.generate(rel_messages))
        return True, score
    
    return False, 0.0

def compute_hdr(true_hypotheses: Dict[str, SummaryInformation], 
                generated_hypotheses: Dict[str, SummaryInformation],
                api) -> Dict:
    """Compute HDR metrics using SummaryInformation instances."""
    total_true_hyps = len(true_hypotheses)
    discovered_count = 0
    all_rel_scores = []
    
    for true_id, true_hyp in true_hypotheses.items():
        hyp_scores = []
        discovered = False
        
        # Compare with each generated hypothesis
        for gen_id, gen_hyp in generated_hypotheses.items():
            is_match, score = evaluate_pairwise_discovery(true_hyp, gen_hyp, api)
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
