#!/bin/bash
# Build script for DeepWiki Plugin UI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building DeepWiki Plugin UI...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMPLATE_DIR="$SCRIPT_DIR/template"
DIST_DIR="$SCRIPT_DIR/dist"

# Check if template directory exists
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo -e "${RED}Error: Template directory not found at $TEMPLATE_DIR${NC}"
    exit 1
fi

cd "$TEMPLATE_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Build
echo -e "${YELLOW}Building UI...${NC}"
npm run build

# Verify build output
if [ ! -d "$DIST_DIR" ]; then
    echo -e "${RED}Error: Build output directory not found at $DIST_DIR${NC}"
    exit 1
fi

if [ ! -f "$DIST_DIR/index.html" ]; then
    echo -e "${RED}Error: index.html not found in build output${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful!${NC}"
echo -e "${GREEN}Output directory: $DIST_DIR${NC}"
echo ""
echo "Files:"
ls -lh "$DIST_DIR"

echo ""
echo -e "${GREEN}Ready for deployment!${NC}"
echo ""
echo "The UI will be served at:"
echo "  https://your-domain.com/app/ui_host/deepwiki/ui/{project_id}/"
