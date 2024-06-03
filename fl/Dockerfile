FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x commands/CNN/fedavg-non-iid.sh

CMD ["./commands/CNN/fedavg-non-iid.sh"]
