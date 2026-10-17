import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="GraphShield AI phase-by-phase research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Validate PaySim, export Parquet, audit training graph")
    ingest.add_argument("--input", type=Path, required=True, help="CSV path inside the project")
    ingest.add_argument("--config", type=Path, default=Path("config/phase1.json"))
    ingest.add_argument("--run-name", required=True, help="New unique artifact directory name")
    feature_cmd = commands.add_parser("features", help="Build strictly historical train/validation features")
    feature_cmd.add_argument("--source-run", required=True)
    feature_cmd.add_argument("--run-name", required=True)
    feature_cmd.add_argument("--config", type=Path, default=Path("config/phase2.json"))
    train_cmd = commands.add_parser("train", help="Fit and compare validation-only transaction baselines")
    train_cmd.add_argument("--feature-run", required=True)
    train_cmd.add_argument("--run-name", required=True)
    train_cmd.add_argument("--config", type=Path, default=Path("config/phase2.json"))
    synthetic_cmd = commands.add_parser("generate-synthetic", help="Build the separate coordinated-activity scenario benchmark")
    synthetic_cmd.add_argument("--run-name", required=True)
    synthetic_cmd.add_argument("--config", type=Path, default=Path("config/phase3.json"))
    rings_cmd = commands.add_parser("detect-rings", help="Replay historical graph rules on development worlds")
    rings_cmd.add_argument("--source-run", required=True)
    rings_cmd.add_argument("--run-name", required=True)
    rings_cmd.add_argument("--config", type=Path, default=Path("config/phase4.json"))
    learned_cmd = commands.add_parser("train-rings", help="Train and compare development-only graph ranking models")
    learned_cmd.add_argument("--rule-run", required=True)
    learned_cmd.add_argument("--run-name", required=True)
    learned_cmd.add_argument("--config", type=Path, default=Path("config/phase5.json"))
    bundle_cmd = commands.add_parser("prepare-serving", help="Export a verified local serving bundle")
    bundle_cmd.add_argument("--learned-run", default="learned-v1")
    bundle_cmd.add_argument("--bundle-name", default="serving-v1")
    replay_cmd = commands.add_parser("replay", help="Ingest and score one synthetic development network")
    replay_cmd.add_argument("--bundle-name", default="serving-v1")
    replay_cmd.add_argument("--database", type=Path, default=Path("artifacts/phase6/demo.sqlite"))
    replay_cmd.add_argument("--world-id")
    replay_cmd.add_argument("--model")
    replay_cmd.add_argument("--policy", choices=["tuned_threshold","budget_only"], default="tuned_threshold")
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            run_pipeline(args.input, args.config, args.run_name)
        elif args.command == "features":
            from .baselines import generate_features
            generate_features(args.source_run, args.run_name, args.config)
        elif args.command == "train":
            from .baselines import train_baselines
            train_baselines(args.feature_run, args.run_name, args.config)
        elif args.command == "detect-rings":
            from .rings import run_rules
            run_rules(args.source_run, args.run_name, args.config)
        elif args.command == "train-rings":
            from .learned_rings import train_rings
            train_rings(args.rule_run, args.run_name, args.config)
        elif args.command == "prepare-serving":
            from .serving import prepare_bundle
            prepare_bundle(args.learned_run, args.bundle_name)
            print("Serving bundle prepared.")
        elif args.command == "replay":
            import json
            from .serving import Bundle, InvestigationService
            from .service_store import Store
            bundle = Bundle(args.bundle_name)
            service = InvestigationService(bundle, Store(args.database))
            job = service.request_replay(args.world_id or bundle.meta['default_world'], args.model, args.policy)
            print(json.dumps(service.run_replay(job['replay_id']), indent=2))
        else:
            from .synthetic import generate_benchmark
            generate_benchmark(args.run_name, args.config)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"GraphShield: {exc}\n")


if __name__ == "__main__":
    main()
