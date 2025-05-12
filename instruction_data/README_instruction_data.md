# 🧠 instruction_data

This directory contains training-ready instruction-tuning data for fine-tuning or aligning language models to the domain of affordable housing policy, HUD regulations, and public housing operations.

---

## 📂 File Types

- `.jsonl` — Primary format for prompt-response pairs, ideal for LLM instruction tuning.
- `.csv` — Tabular prompt-response formats, optionally used for simpler ingestion or legacy tooling.
- `.txt` — Chunked, unstructured data (e.g., PDF excerpts) for manual review or preprocessing pipelines.

---

## 🧾 File Descriptions

### ✅ `housingmind_instructions.jsonl`
- Manually curated prompt-response pairs.
- Source: HUD notices, Admin Plans, RAD guides, internal policy documents.
- Format:
  ```json
  {
    "prompt": "What rights do tenants retain under RAD?",
    "response": "Tenants have the right to return post-rehab and cannot be rescreened under RAD."
  }
  ```

### 🔁 `generated_from_lookups.jsonl`
- Auto-generated Q&A pairs based on data from `/lookup_tables`.
- Useful for grounding models in factual program data (FMRs, income limits).

### 🏛️ `hud_notice_qa.jsonl`
- Example instruction pairs based on manually annotated excerpts from HUD notices.
- Good for modeling regulatory citation, summarization, or multi-step policy explanations.

### 📄 `pdf_chunks.txt`
- Raw text extracted from PDF notices using `pdf_chunker.py`.
- Unstructured but segmentable into future prompts or for use in retrieval pipelines.

---

## 📌 Tips for Contributors

- Always verify responses are **factually accurate** and **compliant** with HUD policy.
- Use proper citations when referencing specific HUD notices (e.g., "PIH Notice 2024-16").
- Avoid duplicating prompts unless testing variations of style or format.
- JSONL entries must be newline-delimited and UTF-8 encoded.

---

## 🧬 Future Files

- `chat_style_dialogues.jsonl` — Multi-turn conversations between tenant/landlord/AI
- `state_specific_qa.jsonl` — Region-specific prompts using HFA and PHA policies
- `translated_prompts_es.jsonl` — Spanish-language variants of high-frequency prompts

