FROM nvidia/cuda:10.2-base

FROM tiagopeixoto/graph-tool:latest
RUN pacman -S python-pip git --noconfirm

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/root/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc"

WORKDIR /app
ENV PATH=/root/.local/bin/:${PATH}

COPY requirements.txt .
RUN pip install -r requirements.txt