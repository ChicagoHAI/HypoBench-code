import json

def compute_feature_discovery_rate(gold_variables, discovered_variables):
    # Intersection
    intersection = set(gold_variables).intersection(set(discovered_variables))
    fdr = len(intersection) / len(gold_variables) if gold_variables else 0.0
    return fdr, intersection

def compute_relationship_correctness(scores):
    # Average correctness if we have any valid scores
    if len(scores) == 0:
        return 0.0
    return sum(scores) / len(scores)

def evaluate_HDR(gold_variables, gold_relationships, generated_hypothesis, discovered_variables, gen_relationships, relationship_scores):
    # Compute FDR
    fdr, intersection = compute_feature_discovery_rate(gold_variables, discovered_variables)
    # Compute average relationship correctness
    avg_rc = compute_relationship_correctness(relationship_scores)
    # Compute HDR
    hdr = fdr * avg_rc
    return {
        "feature_discovery_rate": fdr,
        "relationship_correctness": avg_rc,
        "HDR": hdr
    }

def evaluate_HDR_multiple(gold_hypotheses, generated_hypotheses, gold_vars_map, gold_rels_map, gen_vars_map, gen_rels_map, rel_scores_map):
    """
    gold_hypotheses: list of gold hypothesis IDs
    generated_hypotheses: list of generated hypothesis IDs
    gold_vars_map: dict { gold_id: [vars...] }
    gold_rels_map: dict { gold_id: {var: relation, ...} }
    gen_vars_map: dict { gen_id: [vars...] }
    gen_rels_map: dict { gen_id: {var: relation, ...} }
    rel_scores_map: dict { (gold_id, gen_id): [scores...] }  # multiple matched features
    """
    total_gold_features = 0
    total_discovered_features = 0
    accumulated_correctness = 0.0

    for g_id in gold_hypotheses:
        g_vars = gold_vars_map[g_id]
        total_gold_features += len(g_vars)

        # Collect matched generated hypotheses
        matched_feature_scores = []
        discovered_count = 0

        for gen_id in generated_hypotheses:
            # Check intersection
            intersection = set(g_vars).intersection(set(gen_vars_map[gen_id]))
            if len(intersection) > 0:
                # This means we've discovered some features
                discovered_count += len(intersection)
                # Relationship correctness for these features
                scores = rel_scores_map.get((g_id, gen_id), [])
                if scores:
                    matched_feature_scores.append(sum(scores) / len(scores))

        if discovered_count > 0:
            total_discovered_features += discovered_count
            if matched_feature_scores:
                # Average correctness for matched generated hypotheses
                accumulated_correctness += sum(matched_feature_scores) / len(matched_feature_scores)

    fdr = float(total_discovered_features) / total_gold_features if total_gold_features else 0.0
    avg_correctness = accumulated_correctness / len(gold_hypotheses) if gold_hypotheses else 0.0
    hdr = fdr * avg_correctness

    return {
        "total_gold_features": total_gold_features,
        "total_discovered_features": total_discovered_features,
        "feature_discovery_rate": fdr,
        "average_relationship_correctness": avg_correctness,
        "HDR": hdr
    }

def evaluate_HDR_multiple_with_model(
    gold_hypotheses,
    generated_hypotheses,
    model_call_func
):
    """
    Uses a model_call_func(gold_hypo, gen_hypo) -> {"gold_feature_count": int, "matched_feature_count": int}
    to determine subtle or non-obvious feature discovery.
    Returns an HDR dict similar to evaluate_HDR_multiple but relies on model-based feature matching.
    """
    total_gold_features = 0
    total_discovered_features = 0

    for g_hypo in gold_hypotheses:
        # For each gold, find the matched features across all generated
        matched_features = 0
        gold_feature_count = 0
        for gen_hypo in generated_hypotheses:
            result = model_call_func(g_hypo, gen_hypo)
            if not gold_feature_count:
                gold_feature_count = result.get("gold_feature_count", 0)
            matched_features += result.get("matched_feature_count", 0)

        total_gold_features += gold_feature_count
        total_discovered_features += matched_features

    fdr = float(total_discovered_features) / total_gold_features if total_gold_features else 0.0
    # Placeholder relationship correctness; you can integrate an LLM-based approach as needed
    avg_correctness = 1.0
    hdr = fdr * avg_correctness

    return {
        "model_based_feature_discovery_rate": fdr,
        "average_relationship_correctness": avg_correctness,
        "HDR": hdr
    }

# For example usage:
def run_eval_example():
    gold_vars = ["A", "B", "C"]
    gold_rels = {"A": "positive", "B": "mixed", "C": "negative"}
    generated_hypo = "larger A => True, larger C => True"
    discovered_vars = ["A", "C"]
    gen_rels = {"A": "positive", "C": "positive"}  # example
    rel_scores = [1.0, 0.0]  # example placeholders

    results = evaluate_HDR(
        gold_vars, gold_rels,
        generated_hypo, discovered_vars,
        gen_rels, rel_scores
    )
    print(json.dumps(results, indent=2))
