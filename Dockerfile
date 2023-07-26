# First stage: load the model
FROM python:3.9 AS model
WORKDIR /app
RUN pip install torch==1.9.0 transformers==4.21.2
RUN mkdir /app/model
RUN python -c "from transformers import DistilBertTokenizer, DistilBertModel; \
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased'); \
    model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased'); \
    tokenizer.save_pretrained('/app/model'); \
    model.save_pretrained('/app/model')"

# Second stage: build the final image
FROM python:3.9-slim
WORKDIR /app
# Copy the cached model from the first stage
COPY --from=model /app/model /app/model
# Copy the Flask application
COPY . .
RUN pip install flask==2.0.2 flask_cors==3.0.10
CMD ["python", "app.py"]
