import json
import os

# ==========================================
# VARIÁVEIS DE CONFIGURAÇÃO
# ==========================================
CAMINHO_JSON = "/app/data/anonymization_benchmark_en.json"  # <-- Coloque o caminho do seu JSON aqui
CAMINHO_TXT = "estrutura_resumo.txt" # <-- Nome do arquivo de saída
# ==========================================

def extrair_estrutura(dados):
    """
    Analisa recursivamente os dados e retorna apenas a estrutura.
    Unifica dicionários dentro de listas para mostrar todas as chaves possíveis.
    """
    if isinstance(dados, dict):
        # Se for um dicionário, processa cada chave
        return {chave: extrair_estrutura(valor) for chave, valor in dados.items()}
        
    elif isinstance(dados, list):
        if not dados:
            return ["Lista Vazia"]
        
        estrutura_unificada = {}
        tipos_simples = set()
        
        for item in dados:
            estrutura_item = extrair_estrutura(item)
            
            # Se o item da lista for um objeto (dict), unificamos as chaves
            if isinstance(estrutura_item, dict):
                for k, v in estrutura_item.items():
                    # Guarda a chave se for nova. 
                    if k not in estrutura_unificada:
                        estrutura_unificada[k] = v
            
            # Se for lista aninhada ou primitivo, agrupa pelo nome do tipo
            elif isinstance(estrutura_item, list):
                tipos_simples.add("Lista")
            else:
                tipos_simples.add(str(estrutura_item))
        
        # Retorna a estrutura unificada se a lista contiver objetos
        if estrutura_unificada:
            return [estrutura_unificada]
        else:
            return [f"Lista de {', '.join(tipos_simples)}"]
            
    else:
        # Substitui o valor real pelo nome do seu tipo (ex: 'str', 'int', 'float')
        return type(dados).__name__

def gerar_resumo():
    print(f"Lendo: {CAMINHO_JSON}...")
    
    try:
        with open(CAMINHO_JSON, 'r', encoding='utf-8') as f:
            json_original = json.load(f)
            
        estrutura_resumida = extrair_estrutura(json_original)
        
        with open(CAMINHO_TXT, 'w', encoding='utf-8') as f:
            # Salva o esqueleto formatado com indentação no arquivo TXT
            json.dump(estrutura_resumida, f, indent=4, ensure_ascii=False)
            
        print(f"Sucesso! Estrutura resumida salva em: {CAMINHO_TXT}")
        
    except FileNotFoundError:
        print(f"Erro: O arquivo '{CAMINHO_JSON}' não foi encontrado.")
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{CAMINHO_JSON}' não é um JSON válido.")

if __name__ == "__main__":
    gerar_resumo()