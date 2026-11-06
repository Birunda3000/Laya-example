import asyncio
from laya import Router
from laya.schema import choice, score, noul

async def run_laya_test():
    print("Iniciando Motor de Decisão Laya (Prefill-Only)...")
    
    # O preload=True evita latência de cold-start pré-alocando os tensores
    router = Router(preload=True)
    
    # Estado/Ticket de entrada
    state = "The system is completely down and I cannot access my dashboard! I will cancel my subscription if this is not fixed now."
    print(f"\nTicket Recebido:\n'{state}'\n")
    
    print("Processando 3 decisões simultâneas em 1 única passagem (aprox 33ms)...")
    
    # Mapeamento do Schema usando as 3 Primitivas Tipadas
    schema = {
        "department": choice(
            "Which department should handle this?",
            {
                "billing": "refunds, cancellations, payments",
                "technical": "bugs, outages, system errors, downtime",
                "other": "general inquiries"
            }
        ),
        "urgency": score(
            "Rate the urgency of this ticket.",
            ["not urgent", "soon", "blocking"]
        ),
        "churn_risk": noul("Does the user threaten to cancel or leave?")
    }
    
    # Executa a predição (inference)
    result = await router.predict(state, schema)
    
    print("\n=== RESULTADO ESTRUTURADO ===")
    print(f"✅ Departamento: {result['department']['choice']} (Confiança calibrada RLCD: {result['department']['answer_confidence']:.2f})")
    print(f"✅ Nível de Urgência: {result['urgency']['score']:.4f}")
    print(f"✅ Risco de Cancelamento: {result['churn_risk']['score']:.4f}")
    print(f"⚡ Tokens Gerados: {result.get('output_tokens', 0)} (Arquitetura não-autorregressiva)")
    print(f"🌍 Motivo Roteamento de Idioma: {result.get('routing', {}).get('reason', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(run_laya_test())
