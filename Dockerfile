FROM python:3.11-slim-bookworm

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH="/app/src"

COPY --chown=user . /app

RUN python download_models.py

CMD ["python", "app.py"]