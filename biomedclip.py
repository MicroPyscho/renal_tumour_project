"""
Run this ONCE on your machine (requires internet, ~900 MB download).
After it completes the model is cached locally and prediction.py
will automatically use it without further downloads.

    python download_biomedclip.py
"""
import open_clip
import torch

print("Downloading BiomedCLIP (~900 MB)...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
tokenizer = open_clip.get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

# Smoke test
from PIL import Image
import numpy as np
img = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
tokens = tokenizer(["a CT scan", "a photograph"])
with torch.no_grad():
    img_f = model.encode_image(preprocess(img).unsqueeze(0))
    txt_f = model.encode_text(tokens)
    score = (img_f / img_f.norm() @ (txt_f / txt_f.norm()).T).softmax(dim=-1)

print(f"✓ BiomedCLIP downloaded and verified (CT score on blank image: {score[0,0].item():.2f})")
print("prediction.py will now use semantic CT gating automatically.")