# 📓 HousingMind Notebooks

This folder contains sample notebooks demonstrating how to explore, preprocess, and use the HousingMind dataset for AI applications in affordable housing.

---

## Included Notebooks

### ✅ `housingmind_instruction_demo.ipynb`
- Loads and previews instruction-tuning data (prompt-response pairs).
- Converts to Alpaca-style format for model fine-tuning.
- Explores prompts/responses and checks dataset integrity.

### 📄 `housingmind_pdf_ingestion_demo.ipynb`
- Demonstrates how to extract text from HUD-related PDFs using PyMuPDF.
- Useful for building new instruction data or RAG-ready corpora.
- Provides example code for reading and previewing raw documents.

### 🔍 `housingmind_semantic_search_demo.ipynb`
- Shows how to use SentenceTransformers to enable semantic search over prompts.
- Helps identify related instructions, summarize guidance, or build a retrieval pipeline.
- Great for AI assistants and contextual lookup tools.

---

## Requirements

Before running these notebooks, install:

```bash
pip install pandas sentence-transformers PyMuPDF
```

---

## 📬 Questions or Contributions?

Open an issue in the main repo or email us with ideas for more notebooks or enhancements.

