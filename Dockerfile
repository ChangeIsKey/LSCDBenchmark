FROM nvidia/cuda:10.2-base

FROM tiagopeixoto/graph-tool:latest
RUN pacman -S python-pip git sudo --noconfirm

ARG USERNAME=vscode
ARG USER_UID=1200
ARG USER_GID=${USER_UID}
ARG HOME=/home/${USERNAME}

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

RUN mkdir -p ${HOME}/.vscode-server/extensions \
    ${HOME}/.vscode-server-insiders/extensions \
    && chown -R ${USERNAME} \
    ${HOME}/.vscode-server \
    ${HOME}/.vscode-server-insiders

WORKDIR /app
ENV PATH=/root/.local/bin/:${PATH}

COPY requirements.txt .
RUN pip install -r requirements.txt