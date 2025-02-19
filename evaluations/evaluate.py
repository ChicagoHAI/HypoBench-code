from hypogenic.LLM_wrapper import llm_wrapper_register
from hypogenic.logger_config import LoggerConfig
import logging
import argparse
from pathlib import Path
from typing import Dict, List

#different evaluation methods
from hypothesis_discovery_rate.eval_HDR import evaluate_hdr
from hypothesis_quality.eval_quality import evaluate_quality
# TODO: Import other evaluation methods as they're implemented

def setup_evaluator(args):
    """Initialize logger and LLM wrapper."""
    logger = LoggerConfig.get_logger("HypoBench - Evaluation")
    LoggerConfig.setup_logger(logging.DEBUG)
    
    api = llm_wrapper_register.build(args.model_type)(
        model=args.model_name, 
        path_name=args.model_path
    )
    return logger, api

def run_evaluations(args, logger, api):
    """Run all specified evaluations."""
    results = {}
    
    if args.hdr or args.all:
        logger.info("\nRunning Hypothesis Discovery Rate evaluation:")
        hdr_results = evaluate_hdr(
            args.metadata,
            args.hypotheses,
            api,
            logger
        )
        results["hdr"] = hdr_results
    
    if args.quality or args.all:
        logger.info("\nRunning Hypothesis Quality evaluation:")
        quality_results = evaluate_quality(
            args.metadata,
            args.hypotheses,
            api,
            logger
        )
        results["quality"] = quality_results
    
    # TODO: Add other evaluation types here
    
    return results

def main():
    parser = argparse.ArgumentParser(description='HypoBench Evaluation Suite')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, required=True, help='Type of model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of model')
    parser.add_argument('--model_path', type=str, help='Path to model')
    
    # Data files
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata JSON')
    parser.add_argument('--hypotheses', type=str, required=True, help='Path to hypotheses JSON')
    
    # Evaluation types
    parser.add_argument('--all', action='store_true', help='Run all evaluations')
    parser.add_argument('--hdr', action='store_true', help='Run HDR evaluation')
    parser.add_argument('--novelty', action='store_true', help='Run novelty evaluation')
    parser.add_argument('--quality', action='store_true', help='Run quality evaluation')
    # TODO: Add other evaluation type flags
    
    args = parser.parse_args()
    
    logger, api = setup_evaluator(args)
    results = run_evaluations(args, logger, api)
    
    # Print summary of all results
    logger.info("\nEvaluation Summary:")
    for eval_type, metrics in results.items():
        logger.info(f"\n{eval_type.upper()} Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    main()
