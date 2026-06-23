
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for script in ["summarize_tables.py","generate_figures.py","run_demo_simulation.py"]:
    subprocess.check_call([sys.executable, str(ROOT/"scripts"/script)])
