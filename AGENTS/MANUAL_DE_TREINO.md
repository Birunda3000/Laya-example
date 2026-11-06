# Manual de Retreinamento, Testes e Exportação do Modelo para Novos Ambientes

Este manual fornece o passo a passo operacional para:
1. **Retreinar o modelo** quando novos dados, tabelas ou colunas forem adicionados;
2. **Executar testes e auditoria de qualidade** para validar se o novo modelo está pronto para produção;
3. **Extrair apenas o modelo treinado** (um artefato ultraleve de apenas **~3.1 MB**);
4. **Implantar e executar o modelo em outro ambiente completamente diferente** (outro servidor, nuvem, contêiner Docker isolado ou máquina local).

---

## 1. Visão Geral da Arquitetura do Modelo

A nossa solução adota uma arquitetura desacoplada e modular:

```text
┌────────────────────────────────────────────────────────┐
│  Leitor Universal Base: nandakishorm/vega-U (0.8B)     │  <- Baixado automaticamente pelo
│  (Causal linear state-space, congelado, multilíngue)   │     Hugging Face / VegaML (Cache)
└──────────────────────────┬─────────────────────────────┘
                           │ Extrai representações textuais
┌──────────────────────────▼─────────────────────────────┐
│  Cabeça Logística Especializada + Calibração Conformal │  <- NOSSO MODELO TREINADO
│  - heads.npz          (3.1 MB)                         │     (Apenas estes 2 arquivos
│  - calibration.json   (2.7 KB)                         │      precisam ser transferidos!)
└────────────────────────────────────────────────────────┘
```

> **Vantagem Estratégica:** Você **não precisa** transferir gigabytes de pesos de rede neural para levar seu modelo para outro ambiente. Apenas os **3.1 MB** da pasta `models/vega_anonymizer_calibration` contêm todo o conhecimento aprendido e calibrado.

---

## 2. Como Treinar Novamente com Dados Novos

### Cenário A: Você recebeu novos arquivos de benchmark (formato JSON)

Se você recebeu novos datasets anotados no mesmo formato JSON dos benchmarks (`datasets -> columns`):

1. **Coloque os novos arquivos na pasta `data/`** (exemplo: `data/anonymization_benchmark_novos_dados.json`).
2. **Re-execute a preparação dos dados:**
   O script [`src/prepare_data.py`](file:///app/src/prepare_data.py) achata as colunas, elimina qualquer vazamento de dados (*data leakage*) e faz a divisão agrupada (*Grouped Split*) por tabela para que colunas do mesmo dataset não se misturem entre treino e teste:
   ```bash
   /app/.venv/bin/python -m src.prepare_data \
     --data-dir data \
     --output-dir data/processed \
     --lang both \
     --test-size 0.2 \
     --seed 42
   ```
   *Saída esperada:* Novos arquivos `data/processed/train.jsonl` e `data/processed/test.jsonl`.

---

### Cenário B: Você deseja adicionar novas linhas diretamente ao treino

Se você já possui exemplos pontuais em formato JSONL (cada linha contendo `dataset_id`, `state` e `expected`):
1. Faça o *append* dos novos registros em `data/processed/train.jsonl`:
   ```bash
   cat novos_exemplos.jsonl >> data/processed/train.jsonl
   ```
2. Certifique-se de que nenhum `dataset_id` novo esteja presente simultaneamente em `test.jsonl`.

---

### Executando o Retreinamento e Re-Calibração

Para retreinar as cabeças logísticas e recalcular as temperaturas estatísticas e o limiar conformal ($q$-hat):

```bash
/app/.venv/bin/python -m src.train_fit \
  --train-file data/processed/train.jsonl \
  --test-file data/processed/test.jsonl \
  --calibration-dir models/vega_anonymizer_calibration \
  --target-precision 0.95 \
  --alpha 0.10 \
  --folds 5 \
  --batch-size 16
```

#### Parâmetros Ajustáveis no Treinamento:
* `--target-precision`: Alvo regulatório de precisão mínima (padrão `0.95`). Se a sua organização exige rigor extremo (ex: conformidade bancária/médica), use `--target-precision 0.99`. O modelo recalibrará o limiar conformal para garantir que as decisões autônomas tenham $\ge 99\%$ de acerto.
* `--folds`: Número de partições da validação cruzada para fitting da cabeça (padrão `5`). Para bases com mais de 2.000 colunas, você pode usar `10`.
* `--skip-zero-shot`: Adicione esta flag se quiser pular a avaliação inicial do modelo base e ir direto ao treino, economizando tempo de CPU.

---

## 3. Como Fazer os Testes e Validação do Novo Modelo

Após o retreinamento, siga este roteiro de 3 etapas para certificar a qualidade do modelo antes do deploy:

### Etapa 3.1: Auditoria do Relatório Automático (`evaluation_report.json`)
O script de treino gera automaticamente o arquivo `models/vega_anonymizer_calibration/evaluation_report.json`.
Inspecione as métricas pós-fit:

```bash
/app/.venv/bin/python -c '
import json
with open("models/vega_anonymizer_calibration/evaluation_report.json") as f:
    rep = json.load(f)
post = rep["after_fit"]
print(f"=== RESULTADOS DO NOVO MODELO ===")
print(f"Acurácia Global:      {post[\"accuracy\"]*100:.2f}%")
print(f"Precisão Global:      {post[\"precision\"]*100:.2f}%")
print(f"Revocação Global:     {post[\"recall\"]*100:.2f}%")
print(f"ROC-AUC:              {post[\"roc_auc\"]:.4f}")
print(f"Brier Score:          {post[\"brier_score\"]:.4f}")
print(f"Precisão Rota Auto:   {post[\"routing\"][\"auto_precision\"]*100:.2f}%")
print(f"Cobertura Rota Auto:  {post[\"routing\"][\"auto_rate\"]*100:.2f}%")
'
```

#### Critérios de Aprovação para Deploy:
- [x] **Precisão na Rota Automática (`auto_precision`):** Deve ser $\ge 95\%$ (ou $\ge 99\%$ se configurado).
- [x] **ROC-AUC:** Deve ser $\ge 0.90$.
- [x] **Brier Score:** Deve ser $\le 0.15$ (indica alta calibração de probabilidade).

---

### Etapa 3.2: Execução da Suíte de Testes Automatizados
Rode a suíte de testes de regressão do repositório para verificar conformidade com schema, isolamento contra vazamento de dados e contratos da API:

```bash
/app/.venv/bin/python -m unittest discover -s tests -v
```
Todos os testes devem retornar `OK`.

---

### Etapa 3.3: Teste Manual Rápido de Inferência
Valide uma predição em tempo real com dados fictícios:

```bash
/app/.venv/bin/python -c '
from src.predictor import AnonymizationDecider

decider = AnonymizationDecider()
res = decider.predict_column(
    dataset_meta={"dataset_name": "teste", "domain": "rh", "purpose": "folha"},
    column_data={
        "name": "cpf_colaborador",
        "data_type": "string",
        "null_percentage": 0.0,
        "cardinality": 500,
        "detectors": {"cpf": True}
    }
)
print("Decisão:", res)
assert res["should_anonymize"] is True, "Falha: CPF deveria ser anonimizado!"
assert res["route"] == "auto", "Falha: deveria ter entrado na rota auto!"
print("Teste manual passou com sucesso!")
'
```

---

## 4. Como Extrair APENAS o Modelo

Para levar o modelo treinado para qualquer outro local, você só precisa de **dois arquivos**:
1. `models/vega_anonymizer_calibration/heads.npz` (Pesos da cabeça treinada - ~3.1 MB)
2. `models/vega_anonymizer_calibration/calibration.json` (Parâmetros estatísticos e limiares de roteamento - ~2.7 KB)

### Comando para Empacotar o Modelo:
Execute no terminal:

```bash
tar -czvf modelo_anonimizador_laya.tar.gz -C models/vega_anonymizer_calibration heads.npz calibration.json
```

Isso gerará o arquivo compacto:
📦 **`modelo_anonimizador_laya.tar.gz`** (apenas **~2.8 MB comprimido**).

Você pode enviar esse arquivo via `scp`, S3, Google Cloud Storage, Git LFS ou anexar em seu pipeline de CI/CD.

---

## 5. Como Usar o Modelo em Outro Ambiente Completamente Diferente

Nesta seção, mostramos como utilizar o modelo extraído em uma máquina virgem (servidor Linux, macOS, AWS EC2, GCP Compute Engine ou contêiner Docker).

### 5.1. Pré-Requisitos no Ambiente de Destino
* Python 3.10, 3.11, 3.12 ou 3.13.
* Apenas duas bibliotecas principais instaladas no ambiente Python:
  ```bash
  pip install laya==0.4.24 vegaml==0.15.0 torch
  ```

---

### 5.2. Descompactando o Modelo no Ambiente de Destino
No servidor novo, crie uma pasta para o modelo e descompacte o arquivo `.tar.gz`:

```bash
mkdir -p /opt/meu_servico/modelo_calibrado
tar -xzvf modelo_anonimizador_laya.tar.gz -C /opt/meu_servico/modelo_calibrado
```

Agora você terá:
* `/opt/meu_servico/modelo_calibrado/heads.npz`
* `/opt/meu_servico/modelo_calibrado/calibration.json`

---

### 5.3. Script Python Mínimo e Autocontido (Zero Dependências do Repositório Original)

Você **não precisa clonar este repositório** no ambiente de destino. Basta criar um script simples em Python (ex: `decisor.py`):

```python
"""
decisor.py - Script autocontido para rodar o modelo em qualquer ambiente.
"""

import laya

# 1. Definição da pergunta oficial (contrato de governança)
QUESTIONS = {
    "should_anonymize": {
        "type": "noul",
        "instructions": (
            "Given the dataset context (domain, purpose, row and column counts) and the column's "
            "metadata, statistical quantiles, and PII detector flags, should this column "
            "be anonymized to protect personal, sensitive, or re-identifiable information?"
        ),
        "criteria": {
            "true": (
                "The column contains direct PII (e.g., CPF, CNPJ, SSN, email, phone, person name) "
                "or quasi-identifiers (dates, locations, ZIP/CEP codes, demographic or high-cardinality "
                "attributes) that pose a privacy or re-identification risk in this dataset's domain and purpose."
            ),
            "false": (
                "The column contains non-sensitive data, generic system metrics, aggregated values, "
                "random/technical identifiers, or public attributes that do not identify individuals in this context."
            ),
        },
        "labels": {"true": "A", "false": "B"},
    }
}

# 2. Inicializar o agente Laya com motor VegaML
agent = laya.Agent(vegaml=True)

# 3. Carregar os pesos e calibração extraídos (os 3.1 MB)
CAMINHO_CALIBRACAO = "/opt/meu_servico/modelo_calibrado"
agent.load_calibration(CAMINHO_CALIBRACAO)
print(f"Modelo carregado com sucesso a partir de {CAMINHO_CALIBRACAO}!")

# 4. Função utilitária para montar o estado da coluna
def montar_estado(dataset_nome, dominio, proposito, nome_coluna, tipo_dado, cardinalidade=100, detectores=None):
    return {
        "dataset": {
            "dataset_name": dataset_nome,
            "domain": dominio,
            "purpose": proposito
        },
        "column": {
            "name": nome_coluna,
            "data_type": tipo_dado,
            "cardinality": cardinalidade,
            "detectors": detectores or {}
        }
    }

# 5. Fazer uma predição real:
estado = montar_estado(
    dataset_nome="cadastro_clientes",
    dominio="financeiro",
    proposito="faturamento",
    nome_coluna="email_pessoal",
    tipo_dado="string",
    cardinalidade=8500,
    detectores={"email": True}
)

resultado = agent.predict(estado, QUESTIONS)
resposta = resultado["answers"]["should_anonymize"]

score_float = float(resposta["noul"])
deve_anonimizar = bool(score_float >= 0.5)
rota = resposta.get("route", "auto")

print("\n--- Resultado da Inferência ---")
print(f"Score Contínuo:  {score_float:.4f}")
print(f"Deve Anonimizar: {deve_anonimizar}")
print(f"Rota Decisória:  {rota} (auto = automatizado | review = auditoria humana)")
```

---

### 5.4. Exemplo de Dockerfile Mínimo para Produção no Novo Ambiente

Se você quiser subir uma API em contêiner Docker isolado em qualquer cluster Kubernetes ou nuvem (AWS ECS, Google Cloud Run, Azure Container Apps):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências mínimas de inferência
RUN pip install --no-cache-dir \
    uvicorn \
    fastapi \
    pydantic \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    laya==0.4.24 \
    vegaml==0.15.0

# Copiar os pesos do modelo (os 3.1 MB)
COPY modelo_calibrado/ /app/modelo_calibrado/

# Copiar os scripts do servidor
COPY server.py /app/server.py
COPY src/ /app/src/

# Configurar variáveis de ambiente
ENV CALIBRATION_DIR=/app/modelo_calibrado
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Para construir e rodar a imagem Docker em qualquer máquina:
```bash
docker build -t laya-anonymizer:v1 .
docker run -d -p 8000:8000 --name anonymizer laya-anonymizer:v1
```

E testar com `curl`:
```bash
curl http://localhost:8000/health
```
Retorno:
```json
{
  "status": "healthy",
  "service": "laya-anonymization-decider",
  "calibration_loaded": true
}
```

---

## 6. Resumo dos Comandos em 1 Minuto (Cheat Sheet)

| Objetivo | Comando |
| :--- | :--- |
| **Preparar Dados Novos** | `python -m src.prepare_data --lang both` |
| **Retreinar e Calibrar** | `python -m src.train_fit --target-precision 0.95` |
| **Rodar Testes de Qualidade**| `python -m unittest discover -s tests -v` |
| **Exportar Modelo (3.1 MB)** | `tar -czvf modelo.tar.gz -C models/vega_anonymizer_calibration heads.npz calibration.json` |
| **Carregar no Novo Ambiente**| `agent = laya.Agent(vegaml=True); agent.load_calibration("caminho_dos_arquivos")` |
