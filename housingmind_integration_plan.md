# HousingMind RAG Integration Plan

## Current Structure
```
/housingmind
├── /raw_documents        # Source PDFs, handbooks, notices, etc.
├── /instruction_data     # Prompt-response pairs in JSONL or CSV
├── /lookup_tables        # Structured reference data (e.g., FMRs, subsidy limits)
├── /scripts              # Tools to preprocess or augment data
└── README.md
```

## Proposed New Structure
```
/housingmind
├── /raw_documents        # [UNCHANGED] Source PDFs, handbooks, notices, etc.
├── /instruction_data     # [UNCHANGED] Prompt-response pairs in JSONL or CSV
├── /lookup_tables        # [UNCHANGED] Structured reference data
├── /scripts              # [UNCHANGED] Original preprocessing scripts
├── /rag                  # [NEW] RAG Implementation
│   ├── /vector_db        # [AUTO-CREATED] ChromaDB storage (add to .gitignore)
│   ├── /scripts          # RAG-specific scripts
│   │   ├── process_documents.py
│   │   ├── setup_vector_db.py
│   │   └── rag_engine.py
│   ├── app.py            # Streamlit web interface
│   ├── setup.py          # Automated setup script
│   ├── requirements.txt  # RAG dependencies
│   ├── .env.example      # Configuration template
│   ├── README.md         # RAG-specific documentation
│   └── QUICK_START.md    # Quick reference
├── README.md             # [UPDATE] Main repo README with both capabilities
├── .gitignore            # [UPDATE] Add vector_db, .env
└── LICENSE               # [UNCHANGED] MIT License
```

## Integration Steps

### Step 1: Prepare Repository
```bash
cd /path/to/HousingMind

# Create RAG directory
mkdir -p rag/scripts

# Create branch for this feature
git checkout -b feature/add-rag-implementation
```

### Step 2: Copy RAG Files
```bash
# Copy all RAG implementation files to rag/ directory
cp /path/to/housingmind_implementation/app.py rag/
cp /path/to/housingmind_implementation/setup.py rag/
cp /path/to/housingmind_implementation/requirements.txt rag/
cp /path/to/housingmind_implementation/.env.example rag/
cp /path/to/housingmind_implementation/README.md rag/
cp /path/to/housingmind_implementation/QUICK_START.md rag/

# Copy scripts
cp /path/to/housingmind_implementation/scripts/*.py rag/scripts/
```

### Step 3: Update Configuration Files

**Update rag/scripts/setup_vector_db.py** - Change default paths:
```python
# OLD
documents_dir="../raw_documents"
persist_dir="../vector_db"

# NEW  
documents_dir="../../raw_documents"
persist_dir="../vector_db"
```

**Update rag/scripts/rag_engine.py** - Change default paths:
```python
# OLD
vector_db = HousingMindVectorDB(
    persist_directory=os.getenv("VECTOR_DB_PATH", "../vector_db"),
    collection_name=os.getenv("COLLECTION_NAME", "housing_docs")
)

# NEW
vector_db = HousingMindVectorDB(
    persist_directory=os.getenv("VECTOR_DB_PATH", "./vector_db"),
    collection_name=os.getenv("COLLECTION_NAME", "housing_docs")
)
```

**Update rag/.env.example** - Add comment about paths:
```bash
# Vector DB Configuration
# Paths relative to /rag directory
VECTOR_DB_PATH=./vector_db
COLLECTION_NAME=housing_docs
```

### Step 4: Update .gitignore
```bash
# Add to root .gitignore (create if doesn't exist)
cat >> .gitignore << 'EOF'

# RAG Vector Database (large binary files)
rag/vector_db/
rag/.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
```

### Step 5: Create Updated Main README
```bash
# Use the new comprehensive README provided below
# Copy to root directory as README.md
```

### Step 6: Commit and Push
```bash
# Add all new files
git add .

# Commit
git commit -m "Add RAG implementation with vector database and Streamlit UI

- Added RAG query engine with ChromaDB vector database
- Implemented document processing pipeline for housing policy docs
- Created Streamlit web interface for easy querying
- Added CLI for technical users
- Configured to use existing /raw_documents as data source
- Includes comprehensive documentation and quick start guide"

# Push to GitHub
git push origin feature/add-rag-implementation

# Create pull request on GitHub
# (Go to GitHub web interface to create PR)
```

## Post-Integration Setup for Users

After merging, users will:

```bash
# 1. Clone repository
git clone https://github.com/zachurban/HousingMind.git
cd HousingMind

# 2. Set up RAG
cd rag
cp .env.example .env
# Edit .env to add OPENAI_API_KEY

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated setup
python setup.py

# 5. Start using
python scripts/rag_engine.py
# OR
streamlit run app.py
```

## Benefits of This Structure

1. **Non-invasive**: Doesn't change existing structure
2. **Clear separation**: RAG lives in `/rag`, training data in existing locations
3. **Shared resources**: RAG uses existing `/raw_documents`
4. **Independent operation**: Can work on fine-tuning or RAG separately
5. **Professional presentation**: Shows dual capability (data + application)

## File Size Considerations

The vector database will grow with your documents:
- ~1000 documents = ~500MB vector_db
- ~5000 documents = ~2-3GB vector_db

This is why we add `rag/vector_db/` to `.gitignore` - users build it locally.

## Documentation Strategy

- **Root README.md**: Overview of entire HousingMind platform
- **rag/README.md**: Detailed RAG implementation docs
- **rag/QUICK_START.md**: Quick reference for RAG users

## Next Steps

1. Copy files as shown in Step 2
2. Make the path adjustments in Step 3
3. Update .gitignore (Step 4)
4. Review the new main README (provided below)
5. Commit and push (Step 6)
6. Create pull request on GitHub
7. Merge when ready

After integration, HousingMind will be both:
- **Training dataset** for fine-tuning LLMs
- **Production RAG application** for immediate use
