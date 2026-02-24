
FROM linearrag-base:v1.0

COPY . /app

CMD ["sw-python", "run", "python", "main.py"]