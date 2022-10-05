FROM nvidia/cuda:latest AS cuda

FROM tiagopeixoto/graph-tool:latest
RUN pacman -S python-pip --noconfirm

WORKDIR /app

COPY requirements.txt .
RUN pip install --user -r requirements.txt

COPY ./src/ src/
COPY ./conf/ conf/
COPY main.py main.py

ENTRYPOINT ["python", "main.py"]
