import sys
sys.path.insert(0, '/app/src')

from cnnClassifier.pipeline.prediction import get_model
get_model('DenseNet121')
print('Model ready.')

import app as flask_app
flask_app.app.run(host='0.0.0.0', port=7860, threaded=True)