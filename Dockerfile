FROM python:3.11-slim-bookworm

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BiomedCLIP during build so it's cached in the image
RUN python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"

USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

COPY --chown=user . /app

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "app:app"]