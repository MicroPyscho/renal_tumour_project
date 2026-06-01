#!/bin/bash
echo "Pre-loading DenseNet121..."
python -c "
import sys
sys.path.insert(0, '/app/src')
from cnnClassifier.pipeline.prediction import get_model
get_model('DenseNet121')
print('Model ready.')
"
exec gunicorn --bind 0.0.0.0:7860 --timeout 120 --workers 1 app:app