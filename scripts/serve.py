"""Launch GraphShield from its checkout; installed wheels provide graphshield-serve."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from graphshield.launch import main

if __name__=='__main__':
    main()
