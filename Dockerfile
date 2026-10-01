FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
# Train the model at build time with: docker build --build-arg ... or mount models/model.joblib
ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "hotel_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
