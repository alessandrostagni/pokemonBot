"""
Small script for saving space by deleting old models up to a certain episode
Usage: python clean_models.py path episode

e.g: python clean_models.py models_battle 5
"""

import sys
import os

path = sys.argv[1]
target_episode = int(sys.argv[2])

for model_folder in os.listdir(path):
    episode = int(model_folder.split('_')[1])
    if episode < target_episode:
        os.system(os.path.join(path, model_folder))
