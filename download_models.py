from huggingface_hub import hf_hub_download
import os

os.makedirs('/home/user/checkpoints', exist_ok=True)
for f in ['best_DenseNet121.keras','best_ResNet50V2.keras','best_EfficientNetV2B3.keras']:
    print(f'Downloading {f}...')
    hf_hub_download(repo_id='MicroPyscho/AidRenal-models', filename=f, repo_type='model', local_dir='/home/user/checkpoints')
    print(f'Done: {f}')