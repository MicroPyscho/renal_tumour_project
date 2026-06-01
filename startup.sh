#!/bin/bash
export PYTHONPATH="/app/src:$PYTHONPATH"
echo "Pre-loading DenseNet121..."
python /app/preload_and_serve.py