# HousingMind
A curated training dataset for fine-tuning large language models on U.S. affordable housing policy, public housing operations, HUD regulations, and voucher program administration. Designed for compliance automation, technical assistance, and intelligent document generation. This project is not endorsed or approved by any U.S. Government entity at this time. 

**Training Data for Affordable Housing Language Models**


## 📦 Dataset Structure

/housingmind
│

├── /raw_documents # Source PDFs, handbooks, notices, etc.

├── /instruction_data # Prompt-response pairs in JSONL or CSV

├── /lookup_tables # Structured reference data (e.g., FMRs, subsidy limits)

├── /scripts # Tools to preprocess or augment data

└── README.md


---

## 🔍 Use Cases

- 🧾 Generate compliant letters and notices (e.g., rent increases, terminations)
- 🧠 Summarize PIH Notices, RAD Guides, or Admin Plans
- 🏢 Simulate PHA operations (portability, inspections, SEMAP audits)
- 🤝 Assist tenants, landlords, and staff through AI-powered chatbots
- 📊 Create financial forecasts, cost reasonableness tests, and subsidy models

---

## 🧠 Example Instruction Entries


{
  "prompt": "Explain how RAD conversion affects tenant rights under PIH Notice 2012-32 Rev 4.",
  "response": "Under RAD, tenants retain key rights, including no rescreening, the right to return post-rehab, and grievance procedures. They must be notified and offered meetings, and receive a new lease after conversion to PBV or PBRA."
}


## 🛠️ Tools & Format
JSONL for instruction-tuning

CSV for structured lookups and tabular QA

PDF & DOCX ingestion supported for upstream pretraining or RAG

## 📚 Sources
Data has been collected from publicly available government and nonprofit resources, including:

HUD Handbooks, Notices, and Regulations

PHA Administrative Plans

LIHTC, RAD, MTW, and PBRA guidance

Urban Institute, NLIHC, and Eviction Lab reports

State Housing Finance Agency documentation

## 🔐 License
This dataset is provided for research, technical assistance, and educational purposes only. No proprietary, sensitive, or personally identifiable information is included.

License: MIT License

## 🤝 Contributing
We welcome pull requests with:

New instruction pairs

Annotated regulatory examples

Local/state-specific policies

Tooling to preprocess or visualize data

Please fork the repo and submit a PR with a clear description of changes.

## 🧭 Future Directions
Add multilingual support for Spanish-language housing assistance

Incorporate chat-style dialogue from real housing counseling scenarios

Expand with synthetic but realistic financial modeling data

## 🧱 Built By
HousingMind is maintained by housing professionals, policy researchers, and technologists committed to equitable, efficient delivery of affordable housing through next-generation tools.

Got suggestions or want to collaborate? Open an issue or email us.
