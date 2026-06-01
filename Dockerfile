FROM python:3.11-slim-bookworm

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"

RUN python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/tmp/checkpoints', exist_ok=True)
for f in ['best_DenseNet121.keras','best_ResNet50V2.keras','best_EfficientNetV2B3.keras']:
    hf_hub_download(repo_id='MicroPyscho/AidRenal-models', filename=f, repo_type='model', local_dir='/tmp/checkpoints')
    print(f'Done: {f}')
"

USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

COPY --chown=user . /app

CMD ["/bin/bash", "/app/startup.sh"]