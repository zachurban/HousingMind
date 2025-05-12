"""
preprocess_all.py

CLI script to run all core preprocessing steps in sequence:
1. Chunk PDFs from /raw_documents
2. Generate instruction pairs from structured lookup tables
3. Generate example QA pairs from HUD Notices

Usage:
    python preprocess_all.py
"""

import os
import subprocess

scripts = [
    "pdf_chunker.py",
    "generate_instruction_pairs.py",
    "generate_notice_qa.py"
]

print("🚀 Running HousingMind preprocessing pipeline...")

for script in scripts:
    print(f"▶️ Running {script}...")
    result = subprocess.run(["python", script], cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"❌ Error running {script}")
        break
    else:
        print(f"✅ Finished {script}")

print("🎉 All preprocessing steps completed.")
