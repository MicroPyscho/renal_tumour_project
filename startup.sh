#!/bin/bash
echo "Pre-loading DenseNet121 into memory..."
python -c "
from cnnClassifier.pipeline.prediction import get_model
get_model('DenseNet121')
print('Model ready.')
"
echo "Starting gunicorn..."
exec gunicorn --bind 0.0.0.0:7860 --timeout 120 --workers 1 app:app
