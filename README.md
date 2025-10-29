#  Odd Calculator

**Odd Calculator** é um sistema inteligente de análise esportiva desenvolvido em **Python**, que utiliza o modelo **Gemini (Google Generative AI)** para calcular **probabilidades, odds e previsões contextuais** de partidas de futebol.  

Além de gerar estatísticas e odds de forma automatizada, o projeto conta com:
- Um **simulador de confrontos** entre times;
- Um **chatbot inteligente** (Gemini IA) com **contexto da simulação**;
- E um **dashboard interativo (Streamlit)** para visualização e interação em tempo real.

---

## 🚀 Funcionalidades

### 🔢 Cálculo e Análise
- Cálculo automático de **taxas de vitória, empate e derrota**  
- Estimativas de **odds de gols e cartões amarelos**  
- Análise contextual de desempenho de cada time via IA  

### 🧠 Inteligência Artificial
- Integração com **Gemini API** (via LangChain)  
- Geração de **análises automáticas curtas e objetivas**  
- **Chatbot contextual** que responde perguntas com base nas estatísticas e simulações  

### ⚔️ Simulador de Confrontos
- Compare **dois times** e receba:
  - Probabilidade de vitória de cada um  
  - Probabilidade de empate  
  - Odds correspondentes  
  - Probalidade de ambos times marcarem na partida. 

### 💻 Interface e Automação
- **Menu CLI** interativo (com Rich)  
- **Painel visual** em Streamlit com KPIs e gráficos  
- **Execução automática** agendada via `schedule`  
- Exportação de **CSV filtrado** com resultados e análise
