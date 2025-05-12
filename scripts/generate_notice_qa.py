"""
generate_notice_qa.py

Manually assistive script to turn HUD Notice excerpts into instruction-tuning Q&A entries.
"""

import json
from pathlib import Path

# Example entries (in practice, you'd parse or annotate these manually)
examples = [
    {
        "section": "PIH Notice 2012-32 Rev 4",
        "excerpt": "Under RAD, tenants retain rights such as the right to return post-rehab and cannot be rescreened.",
        "question": "What rights do tenants retain under RAD according to PIH Notice 2012-32 Rev 4?",
        "answer": "Under RAD, tenants have the right to return after rehabilitation and cannot be rescreened for eligibility."
    },
    {
        "section": "PIH Notice 2023-10",
        "excerpt": "PHAs must notify participants 60 days before any termination action.",
        "question": "How much notice must a PHA provide before a termination under PIH Notice 2023-10?",
        "answer": "PHAs must notify participants at least 60 days in advance before any termination action."
    }
]

output_path = Path("../instruction_data/hud_notice_qa.jsonl")
with open(output_path, "w") as f:
    for entry in examples:
        qa = {"prompt": entry["question"], "response": entry["answer"]}
        f.write(json.dumps(qa) + "\n")

print(f"Wrote {len(examples)} notice-based Q&A pairs to {output_path}")
