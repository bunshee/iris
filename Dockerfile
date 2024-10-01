FROM python:3.12

WORKDIR .

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload"]
