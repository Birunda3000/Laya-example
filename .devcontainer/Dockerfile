FROM python:3.13-slim-bookworm

# Instala dependências do SO e Node.js v22 (para o Pi)
RUN apt-get update && \
    apt-get install -y curl vim netcat-openbsd git && \
    curl -fsSL https://deb.nodesource.com/setup_22.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && rm nodesource_setup.sh

# Instala o gerenciador de pacotes uv
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"

# EXTREMAMENTE IMPORTANTE: Evita falha do compilador JIT do PyTorch na imagem slim ao rodar o Laya
ENV TORCH_DISABLE_NATIVE_JIT=1

# 1. Instala o Pi coding agent
RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent

# 2. Instala o RTK e todas as extensoes do Pi
RUN pi install npm:@dietrichgebert/ponytail && \
    pi install npm:pi-subagents && \
    pi install npm:pi-memory && \
    pi install npm:pi-rtk-optimizer && \
    pi install npm:pi-deepseek-search && \
    pi install npm:@alexanderfortin/pi-deepseek-usage

# 3. Instala as skills do workflow (Matt Pocock / AI Hero)
RUN npx skills add mattpocock/skills --skill=grill-me -y -g && \
    npx skills add mattpocock/skills --skill=domain-model -y -g && \
    npx skills add mattpocock/skills --skill=to-prd -y -g && \
    npx skills add mattpocock/skills --skill=to-issues -y -g && \
    npx skills add mattpocock/skills --skill=tdd -y -g

WORKDIR /app