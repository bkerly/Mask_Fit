#!/bin/bash

# NIOSH Mask Fitting System - Startup Script

echo "=========================================="
echo "NIOSH Respirator Mask Fitting System"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher from python.org"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip is not installed."
    echo "Please install pip first."
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
else
    echo "Dependencies already installed."
fi

echo ""
echo "=========================================="
echo "Starting application..."
echo "=========================================="
echo ""
echo "The application will open in your default web browser."
echo "If it doesn't open automatically, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Run the Streamlit app
streamlit run mask_fitting_app.py
