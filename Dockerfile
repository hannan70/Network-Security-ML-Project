FROM python:3.12-alpine

WORKDIR /app

COPY . /app

RUN apk update && \
    apk add --no-cache gcc g++ musl-dev linux-headers

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]