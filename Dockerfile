FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt /code

RUN pip install -r /code/requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . /code

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
