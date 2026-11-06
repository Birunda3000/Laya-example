# Manual de Instruções: Esteira de Decisão de Anonimização (Laya & VegaML)

Este manual documenta a arquitetura, o funcionamento individual e a utilização isolada de todos os módulos desenvolvidos neste repositório.

A arquitetura foi projetada de forma **estritamente modular e desacoplada**. Isso significa que você pode utilizar a esteira completa ou **apenas partes isoladas** (por exemplo: usar a engenharia de dados sem usar o modelo Laya, rodar a API sem os pesos treinados em modo Zero-Shot, ou plugar outros modelos como XGBoost, OpenAI, Claude ou modelos locais).

---

## 1. Visão Geral da Arquitetura e Estrutura de Arquivos

```text
/app
├── AGENTS/
│   ├── GUIDE.md                 # Guia técnico e regras do benchmark
│   ├── TIPS.md                  # Dicas e documentação oficial da engine Laya
│   ├── RELATORIO_FINAL.md       # Relatório executivo com resultados empíricos
│   └── MANUAL_DE_INSTRUCOES.md  # Este manual prático de instruções
├── data/
│   ├── anonymization_benchmark_pt.json   # Benchmark bruto (Português)
│   ├── anonymization_benchmark_en.json   # Benchmark bruto (Inglês)
│   └── processed/                        # Dados gerados sem leakage
│       ├── train.jsonl                   # 628 colunas de 16 datasets
│       ├── test.jsonl                    # 118 colunas de 4 datasets
│       └── split_summary.json            # Metadados e distribuição de classes
├── models/
│   └── vega_anonymizer_calibration/      # Artefatos treinados e calibrados
│       ├── heads.npz                     # Pesos da cabeça logística TTTHead
│       ├── calibration.json              # Temperatura e limiar conformal q-hat
│       └── evaluation_report.json        # Relatório de métricas before/after
├── src/
│   ├── __init__.py                       # Inicializador do pacote
│   ├── schema.py                         # Definição noul e construtor anti-leakage
│   ├── prepare_data.py                   # Flattening e Grouped Split bilíngue
│   ├── train_fit.py                      # Treino de cabeças VegaAgent.fit + calibração
│   ├── train_rlcd.py                     # Fine-tuning de pesos RLCD / LoRA
│   └── predictor.py                      # Wrapper de inferência em Python
├── server.py                             # API REST em FastAPI
└── README_PIPELINE.md                    # Guia rápido de comandos
```

---

## 2. Como Usar Cada Módulo Isoladamente

---

### Módulo A: Engenharia de Dados e Blindagem Anti-Leakage (`src/schema.py` e `src/prepare_data.py`)

#### Para que serve?
Lê benchmarks tabulares complexos, achata a hierarquia (1 coluna = 1 registro de inferência), remove estritamente qualquer vazamento de dados (*data leakage*) e realiza partição agrupada (*Grouped Split*) por dataset para garantir que colunas de uma mesma tabela nunca estejam simultaneamente no treino e no teste.

#### 1. Executando via CLI:
```bash
# Execução padrão: bilíngue (PT + EN), 20% para teste (4 datasets), com seed fixa
/app/.venv/bin/python -m src.prepare_data --lang both --output-dir data/processed

# Gerar dados APENAS em Português:
/app/.venv/bin/python -m src.prepare_data --lang pt --output-dir data/processed_pt

# Gerar dados APENAS em Inglês:
/app/.venv/bin/python -m src.prepare_data --lang en --output-dir data/processed_en

# Remover o campo 'semantic_type' (para testar inferência cega sem dica semântica):
/app/.venv/bin/python -m src.prepare_data --drop-semantic-type --output-dir data/processed_no_semantic
```

#### 2. Usando o Construtor de Estado isoladamente em seu código Python:
A função `build_column_state` pode ser importada e utilizada em qualquer projeto para limpar metadados de tabelas:

```python
from src.schema import build_column_state

dataset_meta = {
    "dataset_name": "tb_clientes",
    "domain": "financeiro",
    "purpose": "análise de risco de crédito",
    "num_rows": 50000,
    "num_cols": 12,
}

column_data = {
    "name": "num_documento",
    "description": "Número de identificação do titular",
    "data_type": "string",
    "semantic_type": "cpf",
    "null_percentage": 0.01,
    "cardinality": 49500,
    "quantiles": None,
    "detectors": {"cpf": True, "cnpj": False, "email": False},
    # Campos que causariam vazamento:
    "should_anonymize": True,
    "reason": "Documento individual identificável",
}

# O construtor remove automaticamente 'should_anonymize' e 'reason':
clean_state = build_column_state(dataset_meta, column_data, drop_semantic_type=False)
print(clean_state)
```

---

### Módulo B: Se Você NÃO Quiser Usar o Modelo Treinado Laya / VegaML

Se você preferir treinar **outros modelos** (como XGBoost, Random Forest, Scikit-Learn ou LLMs externos via OpenAI/Claude/Gemini/Ollama), você pode usar os dados gerados pela nossa esteira:

#### Opção B1: Treinar um Classificador Tabular Tradicional (Scikit-Learn / XGBoost)
Os arquivos `data/processed/train.jsonl` e `test.jsonl` contêm atributos estruturados prontos para extração de features:

```python
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def jsonl_to_features(jsonl_path):
    X, y = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            col = row["state"]["column"]
            # Extrair features tabulares:
            feats = {
                "null_percentage": col.get("null_percentage") or 0.0,
                "cardinality": col.get("cardinality") or 0,
                "is_string": 1 if col.get("data_type") == "string" else 0,
                "is_numeric": 1 if col.get("data_type") in ["integer", "float"] else 0,
            }
            # Adicionar flags dos detectores de PII:
            detectors = col.get("detectors") or {}
            for det_name, det_val in detectors.items():
                feats[f"det_{det_name}"] = 1 if det_val else 0

            X.append(feats)
            y.append(row["expected"]["should_anonymize"])
    return pd.DataFrame(X), y


# Carregar dados preparados pela esteira:
X_train, y_train = jsonl_to_features("data/processed/train.jsonl")
X_test, y_test = jsonl_to_features("data/processed/test.jsonl")

# Treinar modelo alternativo:
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

#### Opção B2: Chamar um LLM Externo (OpenAI, Claude, Gemini ou Ollama Local)
Você pode utilizar o `state` sanitizado gerado por `build_column_state` e aplicar as diretrizes de `src/schema.py`:

```python
import json
from src.schema import QUESTIONS, build_column_state


def format_prompt_for_llm(dataset_meta, column_data):
    state = build_column_state(dataset_meta, column_data)
    instructions = QUESTIONS["should_anonymize"]["instructions"]
    criteria = QUESTIONS["should_anonymize"]["criteria"]

    prompt = f"""
{instructions}

Critérios para Decisão:
- ANONIMIZAR (TRUE): {criteria["true"]}
- NÃO ANONIMIZAR (FALSE): {criteria["false"]}

Dados da Coluna a Avaliar:
{json.dumps(state, indent=2, ensure_ascii=False)}

Responda em formato JSON:
{{"should_anonymize": true/false, "confidence": 0.0 a 1.0, "reason": "justificativa"}}
"""
    return prompt


# Envie o prompt acima para gpt-4o, claude-3-5-sonnet, gemini-1.5-pro ou ollama run llama3
```

---

### Módulo C: O Decisor em Python (`src/predictor.py`)

A classe `AnonymizationDecider` encapsula toda a lógica de inferência sem necessitar de API HTTP ou servidor web.

#### 1. Uso Padrão (Carregando a Calibração Treinada):
```python
from src.predictor import AnonymizationDecider

# Instancia o motor e carrega automaticamente os pesos de 'models/vega_anonymizer_calibration'
decider = AnonymizationDecider(
    calibration_dir="models/vega_anonymizer_calibration",
    use_vega=True,
    default_threshold=0.5,
)

# Avaliar uma única coluna:
res = decider.predict_column(
    dataset_meta={"dataset_name": "usuarios", "domain": "rh", "purpose": "folha"},
    column_data={
        "name": "salario_bruto",
        "data_type": "float",
        "null_percentage": 0.0,
        "cardinality": 420,
        "detectors": {},
    },
)

print(f"Deve Anonimizar? {res['should_anonymize']}")
print(f"Probabilidade (score_float): {res['score_float']}")
print(f"Rota de Decisão: {res['route']}")  # 'auto' ou 'review'
```

#### 2. Como usar o Decisor SEM os Pesos Calibrados (Modo Zero-Shot Puro):
Se você quiser avaliar como o modelo original de fábrica se comporta sem carregar a calibração que nós treinamos, basta passar um diretório inexistente ou vazio:

```python
from src.predictor import AnonymizationDecider

# Sem calibração (Zero-Shot puro do leitor Vega):
decider_zeroshot = AnonymizationDecider(
    calibration_dir="diretorio_inexistente", use_vega=True
)

res = decider_zeroshot.predict_column(dataset_meta, column_data)
print("Decisão Zero-Shot:", res)
```

#### 3. Como usar com outro Backbone de Linguagem (ex: Checkpoint Transformer Multilingual):
Se você quiser utilizar a arquitetura Transformer padrão do Laya (`convaiinnovations/laya-multilingual` ou checkpoint local) em vez do VegaML:

```python
from src.predictor import AnonymizationDecider

# Desativa VegaML e usa modelo Transformer base:
decider_hf = AnonymizationDecider(calibration_dir="", use_vega=False)
```

#### 4. Avaliando um Dataset Completo em Lote (`predict_dataset`):
Processa dezenas ou centenas de colunas com paralelismo vetorizado em lote (`predict_batch`):

```python
dataset_payload = {
    "dataset_name": "cadastro_geral",
    "domain": "comercial",
    "purpose": "vendas",
    "columns": [
        {"name": "id", "data_type": "int", "cardinality": 1000, "detectors": {}},
        {
            "name": "cpf",
            "data_type": "str",
            "cardinality": 990,
            "detectors": {"cpf": True},
        },
        {
            "name": "total_compra",
            "data_type": "float",
            "cardinality": 500,
            "detectors": {},
        },
    ],
}

report = decider.predict_dataset(dataset_payload, threshold=0.5, batch_size=16)

print(f"Total de Colunas: {report['total_columns']}")
print(f"Marcadas para Anonimizar: {report['anonymize_columns_count']}")
print(f"Aprovadas Automaticamente: {report['auto_route_count']}")
print(f"Encaminhadas para Revisão Humana: {report['review_route_count']}")
```

---

### Módulo D: Treinamento e Re-Calibração (`src/train_fit.py` e `src/train_rlcd.py`)

#### 1. Re-calibrando o modelo com Novas Metas de Precisão:
Se você quiser ser ainda mais conservador e exigir **99% de precisão mínima** antes de permitir que o modelo tome decisões automáticas:

```bash
/app/.venv/bin/python -m src.train_fit \
  --train-file data/processed/train.jsonl \
  --test-file data/processed/test.jsonl \
  --calibration-dir models/minha_nova_calibracao \
  --target-precision 0.99 \
  --alpha 0.05 \
  --folds 5
```
*O que acontece:* O sistema recalculará a temperatura e elevará o limiar $q$-hat. Mais casos ambíguos serão roteados para `review` (auditoria humana), garantindo que as decisões marcadas como `auto` atinjam $\ge 99\%$ de precisão.

#### 2. Fine-Tuning Completo de Pesos com RLCD (`src/train_rlcd.py`):
Se você estiver em um servidor com GPU e desejar fazer o ajuste fino de todos os pesos do modelo leitor via gradiente supervisionado (Soft-Cross-Entropy / RLCD):

```bash
# Validar tokenização e sintaxe sem treinar (Dry-Run):
/app/.venv/bin/python -m src.train_rlcd \
  --data data/processed/train.jsonl \
  --eval data/processed/test.jsonl \
  --dry-run

# Executar fine-tuning real (recomendado em máquina com GPU):
/app/.venv/bin/python -m src.train_rlcd \
  --data data/processed/train.jsonl \
  --eval data/processed/test.jsonl \
  --output-dir models/meu_checkpoint_rlcd \
  --epochs 4 \
  --micro-batch 2 \
  --grad-accum 8
```

---

### Módulo E: Servidor Web e API REST (`server.py`)

A API expõe o decisor através de um servidor FastAPI assíncrono com validação via Pydantic.

#### 1. Inicializando o Servidor:
```bash
# Iniciar na porta 8000 com recarga automática para desenvolvimento:
/app/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Variáveis de Ambiente Suportadas:
- `CALIBRATION_DIR`: Caminho do diretório de artefatos (padrão: `models/vega_anonymizer_calibration`).  
  *Dica:* Se definir `export CALIBRATION_DIR=""`, a API subirá em modo **Zero-Shot** sem usar o modelo treinado.
- `LAYA_API_KEY`: Se definida (ex: `export LAYA_API_KEY="minha-chave-secreta"`), todas as requisições para `/v1/...` exigirão o cabeçalho `X-API-Key: minha-chave-secreta`. Se não for definida, o acesso é público.

#### 3. Exemplos de Chamadas HTTP:

##### A. Verificação de Saúde (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
**Resposta:**
```json
{
  "status": "healthy",
  "service": "laya-anonymization-decider",
  "laya_version": "0.4.24",
  "vegaml_version": "0.15.0",
  "calibration_loaded": true,
  "calibration_dir": "models/vega_anonymizer_calibration"
}
```

##### B. Decisão para uma Coluna (`POST /v1/anonymization/decide-column`)
```bash
curl -X POST http://localhost:8000/v1/anonymization/decide-column \
  -H "Content-Type: application/json" \
  -H "X-API-Key: minha-chave-secreta" \
  -d '{
    "dataset": {
      "dataset_name": "base_pacientes",
      "domain": "saúde",
      "purpose": "prontuário clínico",
      "num_rows": 10000,
      "num_cols": 8
    },
    "column": {
      "name": "nome_do_paciente",
      "data_type": "string",
      "null_percentage": 0.0,
      "cardinality": 9980,
      "detectors": {"person_name": true}
    },
    "threshold": 0.5
  }'
```
**Resposta:**
```json
{
  "column_name": "nome_do_paciente",
  "score_float": 0.994112,
  "should_anonymize": true,
  "confidence": 0.994112,
  "route": "auto",
  "route_reason": "",
  "abstain": false,
  "unbound": false,
  "prediction_set": ["true"],
  "latency_ms": 142.3
}
```

##### C. Auditoria de um Dataset Completo (`POST /v1/anonymization/evaluate-dataset`)
```bash
curl -X POST http://localhost:8000/v1/anonymization/evaluate-dataset \
  -H "Content-Type: application/json" \
  -H "X-API-Key: minha-chave-secreta" \
  -d '{
    "dataset_name": "log_acessos",
    "domain": "segurança",
    "purpose": "auditoria",
    "columns": [
      {
        "name": "ip_origem",
        "data_type": "string",
        "cardinality": 5000,
        "detectors": {"location": true}
      },
      {
        "name": "codigo_status_http",
        "data_type": "integer",
        "cardinality": 5,
        "detectors": {}
      }
    ]
  }'
```

---

## 3. Guia Rápido de Solução de Problemas (Troubleshooting)

| Problema | Causa Provável | Solução |
| :--- | :--- | :--- |
| `HTTP 401 Unauthorized` na API | A variável `LAYA_API_KEY` está ativa e a requisição não enviou o header `X-API-Key`. | Inclua o cabeçalho `-H "X-API-Key: <sua_chave>"` na requisição ou desative a variável com `unset LAYA_API_KEY`. |
| `Operating in zero-shot mode` no log | O caminho de calibração não contém os arquivos `heads.npz` e `calibration.json`. | Execute `python -m src.train_fit` para gerar os artefatos ou aponte `CALIBRATION_DIR` para o diretório correto. |
| Incerteza / Rota `"review"` frequente | A coluna possui dados com forte ambiguidade de PII indireta ou a meta de precisão foi configurada muito alta. | O comportamento é intencional para proteção regulatória (LGPD). Se desejar maior automação, reduza `--target-precision` para `0.90` ou ajuste o limiar de decisão (`threshold`). |
| Processamento em CPU lento para grandes lotes | O leitor Vega roda múltiplos passos sequenciais sem GPU. | Aumente o parâmetro `batch_size=32` no `predict_dataset` ou habilite múltiplos workers no `uvicorn`. |

