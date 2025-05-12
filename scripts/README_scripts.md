# ⚙️ HousingMind Scripts

This directory contains utility scripts used to process, clean, and generate training data from housing-related documents and reference tables.

---

## 📜 Script Index

### ✅ `generate_instruction_pairs.py`
- Converts structured data (FMR, income limits) into JSONL prompt-response pairs.
- Output is used for instruction-tuning language models.

### 📄 `pdf_chunker.py`
- Reads HUD-related PDF documents and splits them into text chunks (~1000 characters).
- Useful for preprocessing documents for annotation, summarization, or RAG ingestion.

### 🏛️ `generate_notice_qa.py`
- Demonstrates how to manually convert HUD Notice excerpts into QA instruction data.
- Generates JSONL-formatted prompt-response entries from policy summaries.

---

## ⛓️ Dependencies

To run these scripts, install the following:

```bash
pip install pandas PyMuPDF
```

---

## 💡 Notes

- Ensure paths to `/lookup_tables`, `/raw_documents`, and `/instruction_data` are correct before execution.
- Outputs are typically saved as `.jsonl` for easy integration into training pipelines.

---

## 🤝 Contributions

If you create a new script (e.g., tokenizer, tokenizer-aware chunker, metadata tagger), please add a section here explaining its purpose and usage.
