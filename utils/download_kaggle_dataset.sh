#!/bin/bash

TARGET_DIR="$(dirname "$0")/../dataset"
mkdir -p "$TARGET_DIR"

curl -L -o "$TARGET_DIR/adni-dataset.zip" \
  https://www.kaggle.com/api/v1/datasets/download/ashimariam/adni-dataset

unzip -o "$TARGET_DIR/adni-dataset.zip" -d "$TARGET_DIR"

rm "$TARGET_DIR/adni-dataset.zip"

echo "Down to the Target Directory $TARGET_DIR"
