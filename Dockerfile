FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install application dependencies.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application.
COPY api/ ./api/

# Copy the trained housing pipeline.
COPY models/selected_house_price_model.joblib \
     ./models/selected_house_price_model.joblib

# Copy model metadata.
COPY artifacts/model_selection.json \
     ./artifacts/model_selection.json

COPY artifacts/model_lineage.json \
     ./artifacts/model_lineage.json

# Run the API as a non-root user.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]