#!/usr/bin/env bash

# Upgrade pip (important)
python -m pip install --upgrade pip

# Install spacy model after requirements.txt
python -m spacy download en_core_web_sm
