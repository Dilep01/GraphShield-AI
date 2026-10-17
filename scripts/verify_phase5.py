"""Check historical graph inputs, saved model predictions and development evaluation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from graphshield.learned_rings import verify_learned  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_learned(args.run_name), indent=2))
