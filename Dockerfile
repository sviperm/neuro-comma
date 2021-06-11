FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ADD requirements.txt /requirements.txt

RUN pip install -U pip wheel cython && \
    pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r /requirements.txt

WORKDIR /app

ADD src /app

RUN python /app/download_pretrained_models.py

