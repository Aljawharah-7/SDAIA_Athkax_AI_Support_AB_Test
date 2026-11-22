"""تشغيل المشروع كاملاً بالترتيب الصحيح: python main.py"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["day1_experiment_design.py", "day2_data_and_quality_checks.py",
         "day3_ab_analysis.py", "day4_causal_robustness_decision.py"]

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    for step in STEPS:
        print(f"\n{'=' * 70}\n>>> تشغيل {step}\n{'=' * 70}")
        runpy.run_path(str(ROOT / step), run_name="__main__")
    print("\nاكتمل المشروع. راجع data و outputs/charts و outputs/results")
