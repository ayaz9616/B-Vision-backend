#!/usr/bin/env bash

# Upgrade pip (important)
python3 -m pip install --upgrade pip

# Install spacy model after requirements.txt
python3 -m spacy download en_core_web_sm
