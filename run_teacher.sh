#!/bin/bash
# Launcher for the pokemon-teacher systemd user service: the SLM plays its own
# environment and records demonstrations to demos/ for the trainer.
set -u
cd "/home/abbas/Documents/Projects/Pokemon Agent" || exit 1
export PYTHONDONTWRITEBYTECODE=1
rm -rf __pycache__
exec .venv/bin/python teacher.py
