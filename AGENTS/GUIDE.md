# Especificação Técnica e Guia de Implementação: Motor de Decisão Laya para Anonimização de Colunas

> **Instrução Crítica para o Agente de Código (Pi / AI Coding Agent):**
> 1. **Economia de Tokens (NÃO LEIA OS JSONs BRUTOS):** Os arquivos `data/anonymization_benchmark_pt.json` (~361 KB, >10.700 linhas) e `data/anonymization_benchmark_en.json` (~355 KB, >10.700 linhas) são extensos. **NUNCA** abra ou leia esses arquivos diretamente no contexto do LLM. Utilize o arquivo `/app/estrutura_resumo.txt` como referência oficial da estrutura e processe os dados exclusivamente por meio de scripts Python.
> 2. **Ambiente e Hardware:** Este projeto roda em container Linux (CPU-only, Python 3.13, `uv`). **Não tente instalar pacotes CUDA da NVIDIA nem alterar a configuração do PyTorch no `pyproject.toml`.**
> 3. **Versões Instaladas e Validadas no `.venv`:** `laya==0.4.24` e `vegaml==0.15.0`. Siga estritamente as assinaturas de código documentadas neste guia.

---

## 1. Objetivo do Projeto

Construir um **decisor automatizado bilíngue (PT/EN)** para uma plataforma de anonimização de dados utilizando o motor **Laya** (`laya`).
Para **cada coluna** de uma tabela/dataset — avaliando o contexto do dataset (`domain`, `purpose`, `num_rows`, `num_cols`), metadados da coluna (`name`, `description`, `data_type`, `semantic_type`), estatísticas descritivas (`null_percentage`, `cardinality`, `quantiles`) e detectores de PII (`detectors`) — o modelo deve retornar:

1. **Score Contínuo (`score_float`: `float` de `0.0` a `1.0`):** A probabilidade calibrada $P(\text{true})$ (campo `noul`) de que a coluna deve ser anonimizada.
2. **Decisão Booleana (`should_anonymize`: `bool`):** Sugestão direta (`True` ou `False`) se a coluna deve ou não ser anonimizada.
3. **Metadados de Governança e Roteamento (`route`, `route_reason`, `abstain`, `unbound`):** Indicação automática se a decisão pode seguir sem intervenção (`route == "auto"`) ou se deve ser escalada para revisão humana (`route == "review"`).

---

## 2. Estado Atual do Repositório e Restrições do Ambiente

### 2.1. Arquivos Existentes em `/app`
* `data/anonymization_benchmark_pt.json`: Dataset de benchmark em **Português** (20 datasets, 373 colunas: 266 sensíveis e 107 não sensíveis).
* `data/anonymization_benchmark_en.json`: Dataset de benchmark em **Inglês**, espelhando exatamente os mesmos 20 datasets e 373 colunas da versão em português.
* `estrutura_resumo.txt`: Esquema estrutural completo dos JSONs (gerado por `json_config.py`).
* `pyproject.toml` / `uv.lock`: Dependências gerenciadas pelo `uv` (`laya[serve,mcp,langchain]==0.4.24`, `vegaml==0.15.0`, `fastapi`, `uvicorn`, `torch` CPU).
* *Nota sobre `teste_laya.py`:* O script legado na raiz faz referência a `laya.schema` (que não existe na versão `0.4.24`). Todo código novo deve usar a API real do `laya==0.4.24` detalhada abaixo.

### 2.2. Restrições de Execução (DevContainer)
* **Hardware:** Execução em **CPU** (Linux Debian Bookworm Slim, Python 3.13).
* **Variáveis Críticas:** `TORCH_DISABLE_NATIVE_JIT=1` e `USE_TF=0` ativas.
* **Execução de Comandos:** Sempre utilize `/app/.venv/bin/python` ou `uv run --active python ...`.
* **Estratégia Primária de Treinamento (CPU):** Usar `laya.Agent(vegaml=True)` (`VegaAgent`), que carrega o leitor universal congelado multilíngue (`nandakishorm/vega-U`, 0.8B, 73k tokens de contexto) e executa **`agent.fit()`** para ajustar cabeças logísticas (`TTTHead`), temperatura, limiar conformal (`qhat`) e banda de precisão (`band`) em CPU.
* **Estratégia Secundária de Fine-Tuning (RLCD / GPU Opcional):** Manter script compatível com `laya.train.finetune` e `laya.train.TrainConfig` sobre o checkpoint `convaiinnovations/laya-multilingual` (ou `convaiinnovations/laya` para EN), permitindo treino completo de pesos caso rodado em ambiente com GPU.

---

## 3. Esquema dos Dados de Entrada (`estrutura_resumo.txt`)

Ambos os arquivos `data/anonymization_benchmark_pt.json` e `data/anonymization_benchmark_en.json` seguem o esquema `2.0.0`:

```json
{
    "schema_version": "str",
    "total_datasets": "int",
    "total_columns": "int",
    "total_sensitive_columns": "int",
    "total_non_sensitive_columns": "int",
    "datasets": [
        {
            "dataset_id": "str",
            "dataset_name": "str",
            "domain": "str",
            "purpose": "str",
            "num_rows": "int",
            "num_cols": "int",
            "columns": [
                {
                    "name": "str",
                    "description": "str",
                    "data_type": "str",
                    "semantic_type": "str",
                    "null_percentage": "float",
                    "cardinality": "int",
                    "quantiles": {
                        "min": "float", "p25": "float", "p50": "float",
                        "p75": "float", "max": "float", "mean": "float", "std": "float"
                    },
                    "detectors": {
                        "cpf": "bool", "cnpj": "bool", "cep": "bool",
                        "email": "bool", "phone": "bool", "ssn": "bool",
                        "zipcode": "bool", "date": "bool", "person_name": "bool",
                        "location": "bool"
                    },
                    "should_anonymize": "bool",
                    "reason": "str"
                }
            ]
        }
    ]
}
```

> **Atenção:** Colunas categóricas ou textuais podem ter valores `null` (`None`) dentro do objeto `quantiles`. O código de preparação de dados deve preservar ou tratar `None` normalmente na serialização JSON sem falhar.

---

## 4. Regras de Engenharia de Dados e Prevenção de Data Leakage

### 4.1. Achatamento (Flattening): 1 Coluna = 1 `state`
O Laya decide sobre um único `state` por chamada. **Não passe o dataset inteiro com todas as colunas de uma só vez.**
Achate a estrutura para que **cada coluna** gere um registro independente contendo os metadados do dataset pai + os metadados daquela coluna:

```python
state = {
    "dataset": {
        "dataset_name": dataset_meta["dataset_name"],
        "domain": dataset_meta["domain"],
        "purpose": dataset_meta["purpose"],
        "num_rows": dataset_meta["num_rows"],
        "num_cols": dataset_meta["num_cols"],
    },
    "column": {
        "name": column_data["name"],
        "description": column_data["description"],
        "data_type": column_data["data_type"],
        # "semantic_type" incluído por padrão, removido se drop_semantic_type=True
        "null_percentage": column_data["null_percentage"],
        "cardinality": column_data["cardinality"],
        "quantiles": column_data["quantiles"],
        "detectors": column_data["detectors"],
    }
}
```

### 4.2. Prevenção Estrita de Vazamento de Dados (Data Leakage)
1. **NUNCA inclua `"should_anonymize"` no `state`:** Este é o rótulo alvo (`gold` / `expected`).
2. **NUNCA inclua `"reason"` no `state`:** Contém a justificativa pós-fato da anotação humana; não existe em produção e causaria vazamento direto da resposta.
3. **NUNCA inclua `"dataset_id"` no `state`:** É um identificador de catálogo usado apenas para controle de split.
4. **Flag configurável para `"semantic_type"`:** Incluir suporte ao argumento `--drop-semantic-type` (padrão `False`) para permitir avaliar o modelo sem depender de classificação semântica prévia.
5. **Group Split por `dataset_id` (Invariante ao Idioma):**
   * A divisão entre Treino (`train.jsonl`) e Teste (`test.jsonl`) **deve ser agrupada por `dataset_id`** (ex: 16 datasets para treino e 4 datasets para teste com `--test-size 0.2` e `seed` fixa).
   * **Regra Bilíngue Crítica:** Como `anonymization_benchmark_pt.json` e `anonymization_benchmark_en.json` possuem exatamente os mesmos 20 `dataset_id`s, **o split por `dataset_id` deve ser idêntico entre PT e EN**. Se o modo bilíngue (`--lang both`) for usado, tanto a versão PT quanto a versão EN de um `dataset_id` devem cair juntas no treino ou juntas no teste. Jamais coloque a versão PT de um dataset no treino e a versão EN do mesmo dataset no teste.

### 4.3. Formato Unificado do JSONL (Compatível com `VegaAgent.fit` e `laya.train.finetune`)
No `laya==0.4.24`:
* `VegaAgent.fit()` lê `row["gold"]` ou `row["expected"]` aceitando booleano ou dicionário de probabilidades.
* `laya.train.items_from_rows()` exige que `row["gold"]["should_anonymize"]` contenha `{"probabilities": {"false": ..., "true": ...}}`, **ou** que a linha use `row["expected"]["should_anonymize"] = True/False`.

Para garantir **100% de compatibilidade com ambos os motores**, salve cada linha de `train.jsonl` e `test.jsonl` com a seguinte estrutura:

```json
{
  "dataset_id": "folktables_acs_income",
  "lang": "pt",
  "column_name": "AGEP",
  "state": { ... },
  "questions": { ... },
  "expected": {"should_anonymize": true},
  "gold": {
    "should_anonymize": {
      "label": true,
      "probabilities": {"false": 0.0, "true": 1.0}
    }
  }
}
```

---

## 5. Modelagem da Pergunta no Laya (Primitiva `noul`)

Utilizamos a primitiva **`noul`**, que avalia proposições binárias e retorna diretamente `P(true)` no campo `"noul"` (`0.0` a `1.0`).

### Regras da Primitiva `noul` no `laya==0.4.24`:
1. `criteria` deve ser um dicionário mapeado estritamente pelas chaves `"true"` e `"false"` (qualquer outra chave levanta `ValueError` em `Agent._check_question`).
2. `labels` deve usar `{"true": "A", "false": "B"}` para evitar viés lexical de checkpoints base nas palavras literais `true/false`, mantendo o cálculo matemático de `"noul"` como $P(\text{true})$. No `VegaAgent`, `_to_vega_questions` converte automaticamente `type: "noul"` para `boolean` preservando `criteria`.

### Definição Oficial de `QUESTIONS` (`src/schema.py`):
```python
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
        "labels": {
            "true": "A",
            "false": "B",
        },
    }
}
```

---

## 6. Arquitetura de Arquivos a Implementar

Implemente a seguinte estrutura modular dentro de `/app`:

```text
/app
├── AGENTS/
│   └── GUIDE.md            # Este documento guia atualizado
├── data/
│   ├── anonymization_benchmark_pt.json   # (Existente) 20 datasets / 373 colunas em PT
│   ├── anonymization_benchmark_en.json   # (Existente) 20 datasets / 373 colunas em EN
│   └── processed/                        # (Gerado por prepare_data.py)
│       ├── train.jsonl
│       ├── test.jsonl
│       └── split_summary.json
├── models/
│   └── vega_anonymizer_calibration/      # (Gerado por train_fit.py: heads.npz + calibration.json)
├── src/
│   ├── __init__.py
│   ├── schema.py           # Constante QUESTIONS e build_column_state()
│   ├── prepare_data.py     # Lê os JSONs PT/EN, remove leakage, faz Group Split por dataset_id e gera JSONLs
│   ├── train_fit.py        # Avaliação Zero-Shot + Treinamento CPU via VegaAgent.fit() + Avaliação Pós-Fit
│   ├── train_rlcd.py       # Script para fine-tuning completo via laya.train.finetune (RLCD / Soft-CE)
│   └── predictor.py        # Wrapper AnonymizationDecider para inferência de coluna ou dataset completo
├── server.py               # API FastAPI (/health, /v1/anonymization/decide-column, /v1/anonymization/evaluate-dataset)
└── README_PIPELINE.md      # Documentação de uso e comandos CLI
```

---

## 7. Detalhamento Técnico dos Módulos

### A. `src/schema.py`
* Exportar `QUESTIONS`.
* Exportar `build_column_state(dataset_meta: dict, column_data: dict, drop_semantic_type: bool = False) -> dict`:
  * Filtra estritamente `should_anonymize`, `reason` e `dataset_id`.
  * Remove `semantic_type` somente se `drop_semantic_type=True`.

### B. `src/prepare_data.py`
* Argumentos CLI (`argparse`):
  * `--lang`: choices `["pt", "en", "both"]`, padrão `"both"`.
  * `--data-dir`: padrão `"data"`.
  * `--output-dir`: padrão `"data/processed"`.
  * `--test-size`: proporção de datasets no teste, padrão `0.2` (4 datasets de 20 no teste, 16 no treino).
  * `--seed`: padrão `42`.
  * `--drop-semantic-type`: flag booleana (`action="store_true"`).
* Lógica:
  * Identifica a lista ordenada de `dataset_id`s únicos e sorteia o conjunto de treino e teste usando `random.Random(seed)` no nível de `dataset_id`.
  * Valida que a contagem de linhas de treino supera `laya.VegaAgent.min_rows(QUESTIONS)` (que exige no mínimo 20 linhas para uma questão `noul` com `folds=5` e `calibrate=True`; nosso dataset terá ~298 linhas em um idioma ou ~596 linhas no modo `"both"`).
  * Salva `train.jsonl`, `test.jsonl` e um resumo `split_summary.json` (contendo quais `dataset_id`s ficaram no treino e quais ficaram no teste, além da distribuição de classes).

### C. `src/train_fit.py`
* Argumentos CLI:
  * `--train-file`: padrão `"data/processed/train.jsonl"`.
  * `--test-file`: padrão `"data/processed/test.jsonl"`.
  * `--calibration-dir`: padrão `"models/vega_anonymizer_calibration"`.
  * `--target-precision`: padrão `0.95`.
  * `--alpha`: padrão `0.10`.
  * `--folds`: padrão `5`.
  * `--skip-zero-shot`: flag opcional para pular a avaliação antes do fit caso queira apenas treinar rápido.
* Fluxo de Execução:
  1. Instanciar `agent = laya.Agent(vegaml=True, alpha=args.alpha, target_precision=args.target_precision, folds=args.folds)`.
  2. **(Se não `--skip-zero-shot`)** Avaliar `test_rows` no modo Zero-Shot (Before Fit), calculando:
     * Accuracy, Precision, Recall, F1-Score
     * Brier Score, ROC-AUC (ou log-loss)
     * Taxa de roteamento automático (`route == "auto"`) vs revisão (`route == "review"`) e precisão dentro do subconjunto `route == "auto"`.
  3. Executar `fit_report = agent.fit(train_rows, questions=QUESTIONS, folds=args.folds, calibrate=True, target_precision=args.target_precision)`.
     * Imprimir `fit_report` (`n_rows`, `n_fit`, `n_calibrate`, `kept`, `serving`, `calibration`).
  4. Salvar artefatos com `agent.save_calibration(args.calibration_dir)` (gera `heads.npz` e `calibration.json`).
  5. Avaliar novamente `test_rows` (After Fit) e exibir tabela comparativa **Before Fit vs. After Fit**.

### D. `src/train_rlcd.py` (Opcional / GPU ou Fine-Tuning de Pesos)
* Utiliza `from laya.train import TrainConfig, dry_run, finetune`.
* Permite rodar `--dry-run` na CPU para validar a tokenização e o formato dos dados sem custo pesado, ou rodar o `finetune(...)` completo apontando para `--base convaiinnovations/laya-multilingual` e salvando em `models/laya_rlcd_checkpoint`.

### E. `src/predictor.py`
* Implementar a classe `AnonymizationDecider`:
  * No `__init__(self, calibration_dir="models/vega_anonymizer_calibration", use_vega=True, threshold=0.5, ...)`:
    * Instancia `self.agent = laya.Agent(vegaml=True, ...)` (ou checkpoint padrão se `use_vega=False`).
    * Se `calibration_dir` existir e contiver `heads.npz`, executa `self.agent.load_calibration(calibration_dir)`.
  * Método `predict_column(self, dataset_meta: dict, column_data: dict, threshold: float = 0.5) -> dict`:
    * Monta o `state` limpo via `build_column_state(dataset_meta, column_data)`.
    * Chama `out = self.agent.predict(state, QUESTIONS)`.
    * Extrai `ans = out["answers"]["should_anonymize"]` e retorna:
      ```python
      score_float = float(ans["noul"])
      return {
          "column_name": column_data.get("name"),
          "score_float": round(score_float, 6),
          "should_anonymize": bool(score_float >= threshold),
          "answer_confidence": float(ans.get("answer_confidence") or 0.0),
          "confidence": float(ans.get("confidence") or 0.0),
          "route": ans.get("route", "auto"),          # Regra oficial Laya/Vega: gate por "route", nunca por "confidence"
          "route_reason": ans.get("route_reason", ""),
          "abstain": bool(ans.get("abstain") or False),
          "unbound": bool(ans.get("unbound") or False),
          "prediction_set": ans.get("prediction_set"),
          "latency_ms": out.get("latency_ms", 0.0),
      }
      ```
  * Método `predict_dataset(self, dataset_payload: dict, threshold: float = 0.5) -> dict`:
    * Itera sobre `dataset_payload["columns"]`, coleta a decisão de cada coluna e retorna o resumo consolidado do dataset (total de colunas, quantas marcadas para anonimizar, quantas em `route == "auto"` vs `route == "review"`, e a lista detalhada por coluna).

### F. `server.py` (FastAPI)
* Endpoints:
  * `GET /health`: retorna status do serviço, versão do `laya` e se a calibração (`heads.npz` / `calibration.json`) está carregada.
  * `POST /v1/anonymization/decide-column`: recebe os metadados do dataset + 1 coluna e retorna a decisão estruturada.
  * `POST /v1/anonymization/evaluate-dataset`: recebe um dataset completo com sua lista de `columns` (aceitando payloads com ou sem `should_anonymize`/`reason`, ignorando-os na inferência) e retorna as decisões para todas as colunas.
* Autenticação opcional via header `X-API-Key` caso `LAYA_API_KEY` esteja definida no ambiente.

---

## 8. Checklist de Execução e Validação Passo a Passo

1. **Verificação de Ambiente:**
   ```bash
   /app/.venv/bin/python -c "import laya, vegaml; print('Laya:', laya.__version__, '| VegaML:', vegaml.__version__)"
   ```
2. **Preparação dos Dados (Bilíngue PT + EN sem Leakage):**
   ```bash
   /app/.venv/bin/python -m src.prepare_data --lang both --output-dir data/processed
   ```
3. **Validação Dry-Run do Pipeline de Treino Laya:**
   ```bash
   /app/.venv/bin/python -m src.train_rlcd --data data/processed/train.jsonl --eval data/processed/test.jsonl --dry-run
   ```
4. **Treinamento e Calibração Vega (`agent.fit`) + Benchmark no Conjunto de Teste:**
   ```bash
   /app/.venv/bin/python -m src.train_fit --train-file data/processed/train.jsonl --test-file data/processed/test.jsonl
   ```
5. **Teste do Predictor e Servidor FastAPI:**
   * Verificar se `models/vega_anonymizer_calibration/heads.npz` e `calibration.json` foram gerados.
   * Testar `AnonymizationDecider` em uma coluna real e validar as rotas do `server.py`.
