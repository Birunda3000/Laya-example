# Motor de Decisão Laya para Anonimização de Colunas (Bilíngue PT/EN)

Este projeto implementa um decisor automatizado bilíngue (Português e Inglês) para governança e anonimização de dados usando o motor **Laya** (`laya==0.4.24`) e **VegaML** (`vegaml==0.15.0`).

A partir do contexto do dataset (`domain`, `purpose`, `num_rows`, `num_cols`), metadados da coluna (`name`, `description`, `data_type`, `semantic_type`), estatísticas descritivas (`null_percentage`, `cardinality`, `quantiles`) e detectores de PII (`detectors`), o modelo produz:
1. **`score_float` ($P(\text{true}) \in [0.0, 1.0]$):** probabilidade calibrada de sensibilidade via primitiva `noul`.
2. **`should_anonymize` (`bool`):** decisão binária baseada no limiar de corte (`threshold`, padrão `0.5`).
3. **Governança Conformal (`route`, `route_reason`, `abstain`, `unbound`, `prediction_set`):** roteamento automático seguro (`route == "auto"`) ou escalonamento auditável para revisão humana (`route == "review"`), garantindo precisão alvo $\ge 95\%$.

---

## Estrutura do Repositório

```text
/app
├── AGENTS/
│   ├── GUIDE.md                          # Especificação técnica e guia de implementação
│   └── TIPS.md                           # Documentação e melhores práticas oficiais do Laya
├── data/
│   ├── anonymization_benchmark_pt.json   # Benchmark em Português (20 datasets, 373 colunas)
│   ├── anonymization_benchmark_en.json   # Benchmark em Inglês (20 datasets, 373 colunas)
│   └── processed/                        # Gerado por prepare_data.py
│       ├── train.jsonl                   # 628 colunas (16 datasets agrupados)
│       ├── test.jsonl                    # 118 colunas (4 datasets agrupados)
│       └── split_summary.json            # Metadados e distribuição do split
├── models/
│   └── vega_anonymizer_calibration/      # Gerado por train_fit.py
│       ├── heads.npz                     # Cabeças logísticas ajustadas (TTTHead)
│       ├── calibration.json              # Temperatura, q-hat conformal e banda de precisão
│       └── evaluation_report.json        # Relatório de métricas Before vs. After Fit
├── src/
│   ├── __init__.py
│   ├── schema.py                         # Constante QUESTIONS e construtor de estado limpo
│   ├── prepare_data.py                   # Processamento, remoção de leakage e split agrupado
│   ├── train_fit.py                      # Avaliação Zero-Shot + VegaAgent.fit() + Avaliação Pós-Fit
│   ├── train_rlcd.py                     # Pipeline RLCD para fine-tuning completo / validação dry-run
│   └── predictor.py                      # Wrapper AnonymizationDecider (coluna e dataset completo)
├── server.py                             # API REST FastAPI (/health, /decide-column, /evaluate-dataset)
└── README_PIPELINE.md                    # Esta documentação
```

---

## 1. Preparação dos Dados e Prevenção de Data Leakage

O script `src/prepare_data.py`:
- Achata cada coluna em um registro de decisão independente (`1 coluna = 1 state`).
- **Remove rigorosamente alvos de vazamento:** `should_anonymize`, `reason` e `dataset_id` nunca são injetados no `state`.
- **Group Split por `dataset_id`:** Mantém os datasets de teste estritamente isolados dos datasets de treino, garantindo alinhamento idêntico entre as versões em Português e Inglês.
- Gera formato unificado com chaves `expected` e `gold` compatíveis com `VegaAgent.fit` e `laya.train.finetune`.

### Execução CLI:
```bash
/app/.venv/bin/python -m src.prepare_data --lang both --output-dir data/processed --test-size 0.2 --seed 42
```

Opções adicionais:
- `--lang [pt|en|both]`: Idiomas a incluir (padrão `both`).
- `--drop-semantic-type`: Remove o campo `semantic_type` do `state` para avaliar robustez sem classificação semântica prévia.

---

## 2. Validação Dry-Run (Tokenização e Compatibilidade RLCD)

Valida a tokenização e o formato das linhas JSONL sem custos de treinamento de pesos:
```bash
/app/.venv/bin/python -m src.train_rlcd --data data/processed/train.jsonl --eval data/processed/test.jsonl --dry-run
```

Para treinamento completo de pesos via RLCD / Soft-CE em ambientes com GPU:
```bash
/app/.venv/bin/python -m src.train_rlcd --data data/processed/train.jsonl --output-dir models/laya_rlcd_checkpoint --epochs 4 --device cuda
```

---

## 3. Treinamento e Calibração Vega (`agent.fit`)

Executa o pipeline em CPU:
1. Avalia o modelo pré-treinado no conjunto de teste em modo Zero-Shot.
2. Executa `agent.fit(train_rows, questions=QUESTIONS, folds=5, calibrate=True, target_precision=0.95)`, ajustando cabeças convexas sobre o leitor universal congelado (`nandakishorm/vega-U`, 0.8B) e calibrando a temperatura, limiar conformal $q$-hat e banda de revisão.
3. Salva os artefatos `heads.npz` e `calibration.json` em `models/vega_anonymizer_calibration/`.
4. Avalia novamente o conjunto de teste e exibe tabela comparativa Before vs. After Fit.

### Execução CLI:
```bash
/app/.venv/bin/python -m src.train_fit --train-file data/processed/train.jsonl --test-file data/processed/test.jsonl
```

Flags úteis:
- `--skip-zero-shot`: Pula a avaliação zero-shot antes do fit para treino rápido.
- `--target-precision 0.95`: Precisão certificada desejada na banda de roteamento automático.
- `--alpha 0.10`: Cobertura conformal ($1 - \alpha = 90\%$).
- `--folds 5`: Número de folds de validação cruzada.

---

## 4. Uso Programático em Python (`AnonymizationDecider`)

```python
from src.predictor import AnonymizationDecider

# Instancia o decider e carrega calibração automaticamente
decider = AnonymizationDecider(calibration_dir="models/vega_anonymizer_calibration")

dataset_meta = {
    "dataset_name": "Cadastro de Clientes",
    "domain": "Varejo / E-commerce",
    "purpose": "Marketing e entrega de pedidos",
    "num_rows": 50000,
    "num_cols": 12,
}

column_data = {
    "name": "cpf_cliente",
    "description": "Documento de identificação individual do cliente",
    "data_type": "string",
    "semantic_type": "identifier",
    "null_percentage": 0.0,
    "cardinality": 49800,
    "quantiles": None,
    "detectors": {"cpf": True, "cnpj": False, "email": False, "person_name": False},
}

# Decisão de coluna única
decision = decider.predict_column(dataset_meta, column_data)
print(decision)
# {
#   "column_name": "cpf_cliente",
#   "score_float": 0.9982,
#   "should_anonymize": True,
#   "route": "auto",
#   "route_reason": "conformal singleton, in band",
#   "prediction_set": ["true"],
#   "latency_ms": 32.5
# }
```

---

## 5. Servidor HTTP FastAPI (`server.py`)

Inicie o servidor de inferência:
```bash
/app/.venv/bin/python server.py
# ou com uvicorn:
/app/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
```

### Endpoints Disponíveis:

#### `GET /health`
Verifica saúde do serviço e se os artefatos de calibração estão ativos.
```bash
curl -s http://localhost:8000/health
```

#### `POST /v1/anonymization/decide-column`
Avalia uma única coluna:
```bash
curl -s -X POST http://localhost:8000/v1/anonymization/decide-column \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {
      "dataset_name": "Employee Records",
      "domain": "HR",
      "purpose": "Payroll and benefits management",
      "num_rows": 1000,
      "num_cols": 15
    },
    "column": {
      "name": "SocialSecurityNum",
      "description": "Government-issued SSN identifier",
      "data_type": "string",
      "null_percentage": 0.0,
      "cardinality": 1000,
      "quantiles": null,
      "detectors": {"ssn": true, "cpf": false, "email": false}
    },
    "threshold": 0.5
  }'
```

#### `POST /v1/anonymization/evaluate-dataset`
Avalia todas as colunas de uma tabela em lote com contadores consolidados de governança:
```bash
curl -s -X POST http://localhost:8000/v1/anonymization/evaluate-dataset \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "Inventory",
    "domain": "Logistics",
    "purpose": "Warehouse stock tracking",
    "num_rows": 5000,
    "num_cols": 2,
    "columns": [
      {
        "name": "product_sku",
        "description": "Stock keeping unit random code",
        "data_type": "string",
        "null_percentage": 0.0,
        "cardinality": 5000,
        "detectors": {"cpf": false, "ssn": false}
      },
      {
        "name": "vendor_contact_email",
        "description": "Direct email of sales rep",
        "data_type": "string",
        "null_percentage": 0.02,
        "cardinality": 450,
        "detectors": {"email": true}
      }
    ]
  }'
```

---

## 6. Governança e Roteamento Conformal

Seguindo as diretrizes oficiais do Laya / VegaML:
* **Nunca filtre decisões apenas por `confidence` bruto:** A probabilidade isolada pode ser superconfiante.
* **Filtre por `route`:**
  * `route == "auto"`: A predição atende simultaneamente ao singleton conformal, à banda de precisão calibrada $\ge 95\%$ e não apresenta detecção de out-of-distribution (`unbound == False`) nem abstenção (`abstain == False`).
  * `route == "review"`: A predição possui ambiguidade, incerteza epistêmica ou viola a banda estatística, devendo ser encaminhada para auditoria humana ou revisão manual.
