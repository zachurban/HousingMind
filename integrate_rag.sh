#!/bin/bash

# HousingMind RAG Integration Script
# This script copies all RAG files into your existing HousingMind repository

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HousingMind RAG Integration${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "housingmind_implementation" ]; then
    echo -e "${YELLOW}Error: housingmind_implementation directory not found${NC}"
    echo "Please run this script from the directory containing housingmind_implementation/"
    exit 1
fi

# Get target directory
echo -e "${BLUE}Enter path to your HousingMind repository:${NC}"
read -p "(e.g., /home/user/HousingMind): " HOUSINGMIND_DIR

# Expand ~ if present
HOUSINGMIND_DIR="${HOUSINGMIND_DIR/#\~/$HOME}"

# Verify directory exists
if [ ! -d "$HOUSINGMIND_DIR" ]; then
    echo -e "${YELLOW}Error: Directory $HOUSINGMIND_DIR not found${NC}"
    exit 1
fi

# Verify it's a HousingMind repo
if [ ! -d "$HOUSINGMIND_DIR/raw_documents" ]; then
    echo -e "${YELLOW}Warning: raw_documents directory not found${NC}"
    read -p "Continue anyway? (yes/no): " CONTINUE
    if [ "$CONTINUE" != "yes" ]; then
        echo "Aborting."
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✓ Target directory verified: $HOUSINGMIND_DIR${NC}"
echo ""

# Create RAG directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p "$HOUSINGMIND_DIR/rag/scripts"
echo -e "${GREEN}✓ Created rag/scripts directory${NC}"

# Copy main RAG files
echo ""
echo -e "${BLUE}Copying RAG files...${NC}"

cp housingmind_implementation/app.py "$HOUSINGMIND_DIR/rag/"
echo -e "${GREEN}✓ Copied app.py${NC}"

cp housingmind_implementation/setup.py "$HOUSINGMIND_DIR/rag/"
echo -e "${GREEN}✓ Copied setup.py${NC}"

cp housingmind_implementation/requirements.txt "$HOUSINGMIND_DIR/rag/"
echo -e "${GREEN}✓ Copied requirements.txt${NC}"

cp housingmind_implementation/.env.example "$HOUSINGMIND_DIR/rag/"
echo -e "${GREEN}✓ Copied .env.example${NC}"

cp housingmind_implementation/README.md "$HOUSINGMIND_DIR/rag/"
echo -e "${GREEN}✓ Copied RAG README.md${NC}"

cp housingmind_implementation/QUICK_START.md "$HOUSINGMIND_DIR/rag/"
echo -e "${GREEN}✓ Copied QUICK_START.md${NC}"

# Copy scripts
echo ""
echo -e "${BLUE}Copying scripts...${NC}"

cp housingmind_implementation/scripts/process_documents.py "$HOUSINGMIND_DIR/rag/scripts/"
echo -e "${GREEN}✓ Copied process_documents.py${NC}"

cp housingmind_implementation/scripts/setup_vector_db.py "$HOUSINGMIND_DIR/rag/scripts/"
echo -e "${GREEN}✓ Copied setup_vector_db.py${NC}"

cp housingmind_implementation/scripts/rag_engine.py "$HOUSINGMIND_DIR/rag/scripts/"
echo -e "${GREEN}✓ Copied rag_engine.py${NC}"

# Copy documentation files
echo ""
echo -e "${BLUE}Copying documentation...${NC}"

cp updated_main_README.md "$HOUSINGMIND_DIR/README_NEW.md"
echo -e "${GREEN}✓ Copied new main README as README_NEW.md${NC}"
echo -e "${YELLOW}  Review README_NEW.md and replace README.md when ready${NC}"

cp gitignore_file.txt "$HOUSINGMIND_DIR/.gitignore_additions"
echo -e "${GREEN}✓ Copied .gitignore additions${NC}"
echo -e "${YELLOW}  Append .gitignore_additions to your .gitignore file${NC}"

cp housingmind_integration_plan.md "$HOUSINGMIND_DIR/INTEGRATION_PLAN.md"
echo -e "${GREEN}✓ Copied integration plan${NC}"

# Make scripts executable
chmod +x "$HOUSINGMIND_DIR/rag/setup.py"
chmod +x "$HOUSINGMIND_DIR/rag/scripts/"*.py

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Integration Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Review the integration plan:"
echo -e "   ${YELLOW}cat $HOUSINGMIND_DIR/INTEGRATION_PLAN.md${NC}"
echo ""
echo "2. Update your .gitignore:"
echo -e "   ${YELLOW}cat $HOUSINGMIND_DIR/.gitignore_additions >> $HOUSINGMIND_DIR/.gitignore${NC}"
echo ""
echo "3. Review and replace main README:"
echo -e "   ${YELLOW}mv $HOUSINGMIND_DIR/README_NEW.md $HOUSINGMIND_DIR/README.md${NC}"
echo ""
echo "4. Setup RAG:"
echo -e "   ${YELLOW}cd $HOUSINGMIND_DIR/rag${NC}"
echo -e "   ${YELLOW}cp .env.example .env${NC}"
echo -e "   ${YELLOW}# Add your OPENAI_API_KEY to .env${NC}"
echo -e "   ${YELLOW}pip install -r requirements.txt${NC}"
echo -e "   ${YELLOW}python setup.py${NC}"
echo ""
echo "5. Commit to Git:"
echo -e "   ${YELLOW}cd $HOUSINGMIND_DIR${NC}"
echo -e "   ${YELLOW}git checkout -b feature/add-rag-implementation${NC}"
echo -e "   ${YELLOW}git add .${NC}"
echo -e "   ${YELLOW}git commit -m 'Add RAG implementation'${NC}"
echo -e "   ${YELLOW}git push origin feature/add-rag-implementation${NC}"
echo ""
echo -e "${GREEN}All files copied successfully!${NC}"
echo ""
