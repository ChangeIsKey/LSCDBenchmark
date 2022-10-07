FROM nvidia/cuda:10.2-base

FROM tiagopeixoto/graph-tool:latest
RUN pacman -S python-pip which --noconfirm

WORKDIR /app

COPY requirements.txt .
RUN pip install --user -r requirements.txt

ENTRYPOINT ["python"]
