#!/bin/bash

# Define the folder names to delete
folders=("data" "feature_selection" "figures" "model_evaluation" "model_training")

# Iterate through each folder and delete it if it exists
for folder in "${folders[@]}"; do
  if [ -d "$folder" ]; then
    rm -r "$folder"
    echo "Deleted folder: $folder"
  else
    echo "Folder does not exist: $folder"
  fi
done