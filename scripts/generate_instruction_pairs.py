"""
generate_instruction_pairs.py

This script reads structured housing data from `/lookup_tables` and produces
instruction-style QA pairs for use in instruction-tuning or RAG pipelines.
"""

import pandas as pd
import json
from pathlib import Path

# Load data
fmr_df = pd.read_csv("../lookup_tables/fmr_by_year.csv")
income_df = pd.read_csv("../lookup_tables/income_limits.csv")

# Build prompt-response pairs
instruction_pairs = []

# Example 1: FMR questions
for _, row in fmr_df.iterrows():
    prompt = (
        f"What is the 2024 Fair Market Rent for a 2-bedroom unit in ZIP code {row['zip']} "
        f"({row['county']}, {row['state']})?"
    )
    response = f"The 2024 FMR for a 2-bedroom unit in {row['county']} ({row['zip']}), {row['state']} is ${row['br_2']}."
    instruction_pairs.append({"prompt": prompt, "response": response})

# Example 2: Income limit questions
for _, row in income_df.iterrows():
    prompt = (
        f"What is the income limit for a household of {row['household_size']} in {row['county']}, {row['state']} "
        f"for the year {row['year']}?"
    )
    response = (
        f"For {row['year']}, the income limits in {row['county']}, {row['state']} for a household of {row['household_size']} are:
"
        f"- Extremely Low Income (30% AMI): ${row['extremely_low']}
"
        f"- Very Low Income (50% AMI): ${row['very_low']}
"
        f"- Low Income (80% AMI): ${row['low_income']}"
    )
    instruction_pairs.append({"prompt": prompt, "response": response})

# Save as JSONL
output_path = Path("../instruction_data/generated_from_lookups.jsonl")
with open(output_path, "w") as f:
    for item in instruction_pairs:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(instruction_pairs)} instruction pairs.")
