#!/bin/bash

# Check if the weights directory does not exist, then create it
if [ ! -d "weights" ]; then
  mkdir weights
fi

# Clean the weights directory
rm -rf weights/*.onnx

# Download the files and save them to the weights directory
wget -O weights/w600k_r50.onnx https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx
wget -O weights/det_2.5g.onnx https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_2.5g.onnx
wget -O weights/det_500m.onnx https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_500m.onnx
wget -O weights/det_10g.onnx https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_10g.onnx
wget -O weights/genderage.onnx https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/genderage.onnx
