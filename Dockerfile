# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
WORKDIR /app

COPY ./src ./src
COPY ./server ./app

RUN pip install -U pip wheel setuptools cython && \
    pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r app/requirements.txt


ENV PYTHONPATH="src"
ENV MODULE_NAME="app.main"

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
