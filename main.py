"""
main.py

Command-line entry point for the NIDS pipeline.

Usage
-----
    python main.py train --models random_forest svm knn --binary
    python main.py train --models random_forest --no-binary --sample-data
    python main.py predict --input data/sample_traffic.csv --model random_forest
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_nsl_kdd, load_sample_data
from src.evaluate import (
    evaluate_model,
    full_report,
    plot_confusion_matrix,
    plot_metric_comparison,
    save_metrics_json,
)
from src.predict import predict_csv
from src.preprocessing import prepare_datasets
from src.train import save_model, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def cmd_train(args):
    if args.sample_data:
        logger.warning(
            "Using SYNTHETIC sample data (--sample-data). This is for pipeline "
            "smoke-testing only and does NOT reflect real model performance."
        )
        train_df, test_df = load_sample_data()
    else:
        train_df, test_df = load_nsl_kdd(auto_download=not args.no_download)

    artifacts = prepare_datasets(train_df, test_df, binary=args.binary)
    X_train, X_test = artifacts["X_train"], artifacts["X_test"]
    y_train, y_test = artifacts["y_train"], artifacts["y_test"]

    target_names = None
    if not args.binary and artifacts["target_encoder"] is not None:
        target_names = list(artifacts["target_encoder"].classes_)

    results = []
    for name in args.models:
        model = train_model(name, X_train, y_train)
        save_model(model, name)

        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

        print(f"\n{'=' * 60}\nClassification report: {name}\n{'=' * 60}")
        print(full_report(model, X_test, y_test, name, target_names=target_names))

        plot_confusion_matrix(model, X_test, y_test, name, target_names=target_names)

    if len(results) > 1:
        plot_metric_comparison(results)
    save_metrics_json(results)

    print("\nSummary:")
    for r in results:
        line = f"  {r['model']:<15} accuracy={r['accuracy']:.4f}  f1={r['f1_score']:.4f}"
        if "false_positive_rate" in r:
            line += f"  FPR={r['false_positive_rate']:.4f}"
        print(line)


def cmd_predict(args):
    predict_csv(Path(args.input), Path(args.output), model_name=args.model)
    print(f"Predictions written to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Network Intrusion Detection System (NIDS)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train one or more models and evaluate them")
    p_train.add_argument(
        "--models", nargs="+", default=["random_forest"],
        choices=["random_forest", "svm", "knn"],
        help="Which model(s) to train (default: random_forest)",
    )
    p_train.add_argument(
        "--binary", dest="binary", action="store_true", default=True,
        help="Binary classification: normal vs attack (default)",
    )
    p_train.add_argument(
        "--no-binary", dest="binary", action="store_false",
        help="Multiclass classification: normal/dos/probe/r2l/u2r",
    )
    p_train.add_argument(
        "--sample-data", action="store_true",
        help="Use small synthetic data instead of downloading NSL-KDD (for quick smoke tests)",
    )
    p_train.add_argument(
        "--no-download", action="store_true",
        help="Do not attempt to download NSL-KDD; expect files already in data/raw/",
    )
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Classify traffic records from a CSV file")
    p_predict.add_argument("--input", required=True, help="Path to input CSV (raw NSL-KDD-style columns)")
    p_predict.add_argument("--output", default="predictions.csv", help="Path to write predictions CSV")
    p_predict.add_argument(
        "--model", default="random_forest", choices=["random_forest", "svm", "knn"],
        help="Which trained model to use",
    )
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
