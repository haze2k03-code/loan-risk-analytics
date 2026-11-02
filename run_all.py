"""Runs the whole pipeline in order. Usage:  python run_all.py"""
import subprocess
import sys

STEPS = ["src/prepare.py", "src/run_sql.py", "src/model.py", "src/charts.py", "src/summary.py"]

for step in STEPS:
    print(f"\n{'=' * 70}\nRunning {step}\n{'=' * 70}")
    if subprocess.run([sys.executable, step]).returncode != 0:
        sys.exit(f"\nStopped: {step} failed. Fix the error above and run again.")

print("\nAll done. Start the dashboard with:  streamlit run app.py")
