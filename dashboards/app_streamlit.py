import math
import os
import sys
from datetime import datetime
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analisador_odds import executar_analise_completa

DATA_FILE = os.path.join(PROJECT_ROOT, "brasileirao_2025_odds.csv")
EXPECTED_COLUMNS = [
    "Time",
    "gols_marcados",
    "gols_sofridos",
    "cartões",
    "vitórias",
    "empates",
    "derrotas",
    "partidas",
    "taxa_vitoria",
    "taxa_empate",
    "taxa_derrota",
    "prob_vitoria",
    "prob_empate",
    "prob_derrota",
    "odd_vitoria",
    "odd_empate",
    "odd_derrota",
    "analise_ia",
]
NUMERIC_COLUMNS = [
    "taxa_vitoria",
    "taxa_empate",
    "taxa_derrota",
    "prob_vitoria",
    "prob_empate",
    "prob_derrota",
    "odd_vitoria",
    "odd_empate",
    "odd_derrota",
]
ODD_COLUMNS = ["odd_vitoria", "odd_empate", "odd_derrota"]
MAX_POISSON_GOALS = 7
MAX_POISSON_CARTOES = 12
DEFAULT_TOP_N = 10
CARTOES_WEIGHT = 0.05

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_CHAT_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        GEMINI_CHAT_MODEL = None
else:
    GEMINI_CHAT_MODEL = None


def _poisson_probability(lmbda: float, k: int) -> float:
    if lmbda <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lmbda) * (lmbda**k) / math.factorial(k)


def _normalize_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def probability_to_odds(prob: float) -> float:
    prob = max(prob, 1e-6)
    return round(1.0 / prob, 2)


def gerar_resposta_chatbot(contexto: str, pergunta: str) -> str:
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "Digite uma pergunta para o assistente."
    if GEMINI_CHAT_MODEL is None:
        return "Gemini não está configurado. Defina a variável de ambiente GEMINI_API_KEY."

    prompt = (
        "Você é um analista de apostas esportivas que responde perguntas sobre um confronto de futebol.\n"
        "Utilize apenas os dados fornecidos e mantenha respostas objetivas, com até 3 parágrafos curtos.\n"
        "Contexto do confronto:\n"
        f"{contexto.strip()}\n\n"
        f"Pergunta: {pergunta}\n"
        "Resposta:"
    )

    try:
        resposta = GEMINI_CHAT_MODEL.generate_content(prompt)
        texto = getattr(resposta, "text", "") or ""
        texto = texto.strip()
        if not texto:
            return "O Gemini não retornou uma resposta compreensível. Tente reformular a pergunta."
        return texto
    except Exception as exc:
        return f"Não foi possível obter resposta do Gemini: {exc}"


def calcular_metricas_liga(df: pd.DataFrame) -> Tuple[float, float, float]:
    df_valid = df.copy()
    if "partidas" not in df_valid.columns or df_valid["partidas"].eq(0).all():
        return 1.4, 1.4, 4.5

    df_valid = df_valid[df_valid["partidas"] > 0]
    if df_valid.empty:
        return 1.4, 1.4, 4.5

    gols_marcados_pg = (df_valid["gols_marcados"] / df_valid["partidas"]).dropna()
    gols_sofridos_pg = (df_valid["gols_sofridos"] / df_valid["partidas"]).dropna()
    cartoes_pg = (
        df_valid["cartões"] / df_valid["partidas"]
        if "cartões" in df_valid.columns
        else pd.Series(dtype=float)
    )

    media_marcados = float(gols_marcados_pg.mean()) if not gols_marcados_pg.empty else 1.4
    media_sofridos = float(gols_sofridos_pg.mean()) if not gols_sofridos_pg.empty else 1.4
    media_cartoes = float(cartoes_pg.dropna().mean()) if not cartoes_pg.empty else 4.5

    return media_marcados, media_sofridos, media_cartoes


def calcular_probabilidade_confronto(
    mandante: pd.Series,
    visitante: pd.Series,
    media_gols_marcados: float,
    media_gols_sofridos: float,
    media_cartoes: float,
    vantagem_mandante: float = 0.25,
    peso_cartoes: float = CARTOES_WEIGHT,
) -> Tuple[dict, dict, float, float, dict, float]:
    partidas_mandante = max(float(mandante.get("partidas", 0)), 1.0)
    partidas_visitante = max(float(visitante.get("partidas", 0)), 1.0)

    gols_mandante_pg = float(mandante.get("gols_marcados", 0)) / partidas_mandante
    gols_visitante_pg = float(visitante.get("gols_marcados", 0)) / partidas_visitante
    sofridos_mandante_pg = float(mandante.get("gols_sofridos", 0)) / partidas_mandante
    sofridos_visitante_pg = float(visitante.get("gols_sofridos", 0)) / partidas_visitante
    cartoes_mandante_pg = float(mandante.get("cartões", 0)) / partidas_mandante
    cartoes_visitante_pg = float(visitante.get("cartões", 0)) / partidas_visitante

    media_gols_marcados = max(media_gols_marcados, 0.1)
    media_gols_sofridos = max(media_gols_sofridos, 0.1)
    media_cartoes = max(media_cartoes, 0.1)

    fator_ataque_mandante = gols_mandante_pg / media_gols_marcados
    fator_ataque_visitante = gols_visitante_pg / media_gols_marcados
    fator_defesa_mandante = sofridos_mandante_pg / media_gols_sofridos
    fator_defesa_visitante = sofridos_visitante_pg / media_gols_sofridos

    base_mandante = media_gols_marcados * (1.0 + vantagem_mandante)
    base_visitante = media_gols_marcados * (1.0 - vantagem_mandante)

    expected_home_goals = base_mandante * fator_ataque_mandante * fator_defesa_visitante
    expected_away_goals = base_visitante * fator_ataque_visitante * fator_defesa_mandante

    taxa_vitoria_mandante = float(mandante.get("taxa_vitoria", 0.0)) / 100.0
    taxa_vitoria_visitante = float(visitante.get("taxa_vitoria", 0.0)) / 100.0
    delta_forma = (taxa_vitoria_mandante - taxa_vitoria_visitante) * 0.1

    delta_disciplina_mandante = (media_cartoes - cartoes_mandante_pg) / media_cartoes
    delta_disciplina_visitante = (media_cartoes - cartoes_visitante_pg) / media_cartoes

    expected_home_goals *= max(0.5, 1.0 + delta_forma)
    expected_away_goals *= max(0.5, 1.0 - delta_forma)

    if peso_cartoes:
        expected_home_goals *= max(0.6, 1.0 + peso_cartoes * delta_disciplina_mandante)
        expected_away_goals *= max(0.6, 1.0 + peso_cartoes * delta_disciplina_visitante)

    expected_home_goals = max(expected_home_goals, 0.05)
    expected_away_goals = max(expected_away_goals, 0.05)

    prob_casa = 0.0
    prob_empate = 0.0
    prob_visitante = 0.0

    for gols_casa in range(MAX_POISSON_GOALS + 1):
        prob_gols_casa = _poisson_probability(expected_home_goals, gols_casa)
        for gols_visitante in range(MAX_POISSON_GOALS + 1):
            prob_gols_visitante = _poisson_probability(expected_away_goals, gols_visitante)
            prob_conjunta = prob_gols_casa * prob_gols_visitante
            if gols_casa > gols_visitante:
                prob_casa += prob_conjunta
            elif gols_casa == gols_visitante:
                prob_empate += prob_conjunta
            else:
                prob_visitante += prob_conjunta

    total = prob_casa + prob_empate + prob_visitante
    if total > 0.0:
        prob_casa /= total
        prob_empate /= total
        prob_visitante /= total

    prob_casa = _normalize_probability(prob_casa)
    prob_empate = _normalize_probability(prob_empate)
    prob_visitante = _normalize_probability(prob_visitante)

    odds = {
        "mandante": probability_to_odds(prob_casa),
        "empate": probability_to_odds(prob_empate),
        "visitante": probability_to_odds(prob_visitante),
    }

    probabilidades = {
        "mandante": prob_casa,
        "empate": prob_empate,
        "visitante": prob_visitante,
    }

    expected_cards_home = max(cartoes_mandante_pg, 0.05)
    expected_cards_away = max(cartoes_visitante_pg, 0.05)

    prob_cartoes_home = 0.0
    prob_cartoes_empate = 0.0
    prob_cartoes_visitante = 0.0

    for cards_home in range(MAX_POISSON_CARTOES + 1):
        prob_cards_home = _poisson_probability(expected_cards_home, cards_home)
        for cards_away in range(MAX_POISSON_CARTOES + 1):
            prob_cards_away = _poisson_probability(expected_cards_away, cards_away)
            prob_conjunta = prob_cards_home * prob_cards_away
            if cards_home > cards_away:
                prob_cartoes_home += prob_conjunta
            elif cards_home == cards_away:
                prob_cartoes_empate += prob_conjunta
            else:
                prob_cartoes_visitante += prob_conjunta

    total_cartoes = prob_cartoes_home + prob_cartoes_empate + prob_cartoes_visitante
    if total_cartoes > 0.0:
        prob_cartoes_home /= total_cartoes
        prob_cartoes_empate /= total_cartoes
        prob_cartoes_visitante /= total_cartoes

    prob_cartoes_home = _normalize_probability(prob_cartoes_home)
    prob_cartoes_empate = _normalize_probability(prob_cartoes_empate)
    prob_cartoes_visitante = _normalize_probability(prob_cartoes_visitante)

    cartas_info = {
        "probabilidades": {
            "mandante": prob_cartoes_home,
            "empate": prob_cartoes_empate,
            "visitante": prob_cartoes_visitante,
        },
        "odds": {
            "mandante": probability_to_odds(prob_cartoes_home),
            "empate": probability_to_odds(prob_cartoes_empate),
            "visitante": probability_to_odds(prob_cartoes_visitante),
        },
        "esperados_mandante": expected_cards_home,
        "esperados_visitante": expected_cards_away,
    }

    prob_ambos_marcam = 0.0
    for gols_casa in range(1, MAX_POISSON_GOALS + 1):
        prob_gols_casa = _poisson_probability(expected_home_goals, gols_casa)
        for gols_visitante in range(1, MAX_POISSON_GOALS + 1):
            prob_gols_visitante = _poisson_probability(expected_away_goals, gols_visitante)
            prob_ambos_marcam += prob_gols_casa * prob_gols_visitante

    prob_ambos_marcam = _normalize_probability(prob_ambos_marcam)

    return (
        probabilidades,
        odds,
        expected_home_goals,
        expected_away_goals,
        cartas_info,
        prob_ambos_marcam,
    )


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [col.strip().lstrip("\ufeff") for col in df.columns]
    for coluna in NUMERIC_COLUMNS:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str)
    return df


def rodar_analise_agora() -> None:
    with st.spinner("Executando análise completa. Isso pode levar alguns instantes..."):
        try:
            executar_analise_completa()
        except Exception as exc:
            st.error(f"Falha ao executar a análise: {exc}")
            return
    st.success("Análise concluída com sucesso! Dados atualizados.")
    st.cache_data.clear()
    st.experimental_rerun()


def kpi_cards(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    total_times = df["Time"].nunique() if "Time" in df.columns else 0
    media_vitoria = df["taxa_vitoria"].mean() if "taxa_vitoria" in df.columns else 0.0
    media_empate = df["taxa_empate"].mean() if "taxa_empate" in df.columns else 0.0
    media_derrota = df["taxa_derrota"].mean() if "taxa_derrota" in df.columns else 0.0

    col1.metric("Times analisados", f"{total_times}")
    col2.metric("Vitória média (%)", f"{media_vitoria:.1f}")
    col3.metric("Empate médio (%)", f"{media_empate:.1f}")
    col4.metric("Derrota média (%)", f"{media_derrota:.1f}")


def tabela_top_n(df: pd.DataFrame, top_n: int) -> None:
    st.subheader("Ranking de desempenho (Top N)")
    if df.empty:
        st.info("Nenhum dado disponível com os filtros atuais.")
        return

    colunas = [col for col in ["Time", "taxa_vitoria", "taxa_empate", "taxa_derrota"] if col in df.columns]
    if len(colunas) < 4:
        st.warning("Colunas necessárias ausentes para montar o ranking.")
        return

    df_top = df.sort_values(by="taxa_vitoria", ascending=False).head(top_n)
    st.dataframe(
        df_top[colunas].style.format(
            {
                "taxa_vitoria": "{:.1f}",
                "taxa_empate": "{:.1f}",
                "taxa_derrota": "{:.1f}",
            }
        ),
        use_container_width=True,
    )


def charts_overview(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Aplique filtros mais amplos para visualizar os gráficos.")
        return

    metrica_legendas = [
        ("taxa_vitoria", "Taxa de vitória (%)"),
        ("taxa_empate", "Taxa de empate (%)"),
        ("taxa_derrota", "Taxa de derrota (%)"),
    ]
    cols = st.columns(3)
    for col, (metrica, titulo) in zip(cols, metrica_legendas):
        if metrica not in df.columns:
            col.warning(f"Coluna {metrica} ausente.")
            continue

        fig = px.bar(
            df.sort_values(by=metrica, ascending=False),
            x="Time",
            y=metrica,
            title=titulo,
            labels={"Time": "Time", metrica: titulo},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=0), xaxis_title=None)
        col.plotly_chart(fig, use_container_width=True)


def simular_confronto_tab(df: pd.DataFrame) -> None:
    st.subheader("Simular confronto direto")

    colunas_necessarias = {
        "Time",
        "gols_marcados",
        "gols_sofridos",
        "partidas",
        "taxa_vitoria",
        "cartões",
    }
    if not colunas_necessarias.issubset(df.columns):
        faltantes = ", ".join(sorted(colunas_necessarias - set(df.columns)))
        st.warning(
            "Dados insuficientes para simulação. Colunas ausentes: "
            + faltantes
        )
        return

    times_disponiveis = sorted(df["Time"].dropna().unique().tolist())
    if len(times_disponiveis) < 2:
        st.info("São necessários pelo menos dois times cadastrados para simular um confronto.")
        return

    media_gols_marcados, media_gols_sofridos, media_cartoes = calcular_metricas_liga(df)

    col1, col2 = st.columns(2)
    with col1:
        time_casa = st.selectbox(
            "Time mandante",
            options=times_disponiveis,
            index=0,
            key="sim_time_casa",
        )
    with col2:
        opcoes_visitante = [time for time in times_disponiveis if time != time_casa] or times_disponiveis
        indice_padrao = 0 if opcoes_visitante[0] != time_casa else (1 if len(opcoes_visitante) > 1 else 0)
        time_visitante = st.selectbox(
            "Time visitante",
            options=opcoes_visitante,
            index=min(indice_padrao, len(opcoes_visitante) - 1),
            key="sim_time_visitante",
        )

    vantagem_mandante = 0.25

    if time_casa == time_visitante:
        st.warning("Selecione times diferentes para executar a simulação.")
        return

    serie_mandante = df[df["Time"] == time_casa].iloc[0]
    serie_visitante = df[df["Time"] == time_visitante].iloc[0]

    (
        probabilidades,
        odds,
        gols_esperados_casa,
        gols_esperados_visitante,
        info_cartoes,
        prob_ambos_marcam,
    ) = calcular_probabilidade_confronto(
        serie_mandante,
        serie_visitante,
        media_gols_marcados,
        media_gols_sofridos,
        media_cartoes,
        vantagem_mandante,
    )

    resultados_df = pd.DataFrame(
        {
            "Resultado": ["Vitória mandante", "Empate", "Vitória visitante"],
            "Probabilidade (%)": [
                round(probabilidades["mandante"] * 100, 1),
                round(probabilidades["empate"] * 100, 1),
                round(probabilidades["visitante"] * 100, 1),
            ],
            "Expectativa de odd": [
                odds["mandante"],
                odds["empate"],
                odds["visitante"],
            ],
        }
    )

    st.dataframe(resultados_df, use_container_width=True, hide_index=True)

    st.caption(
        "Modelagem baseada na média de gols e cartões por partida, vantagem de mando fixa em 25% e forma recente (taxa de vitória)."
    )

    col_gols_casa, col_gols_visitante = st.columns(2)
    col_gols_casa.metric("Gols esperados mandante", f"{gols_esperados_casa:.2f}")
    col_gols_visitante.metric("Gols esperados visitante", f"{gols_esperados_visitante:.2f}")

    odd_ambos_marcam = probability_to_odds(prob_ambos_marcam)

    st.metric(
        "Ambos marcam (probabilidade / expectativa de odd)",
        f"{prob_ambos_marcam * 100:.1f}%",
        f"Odd {odd_ambos_marcam:.2f}",
    )

    st.subheader("Cartões")

    cartoes_df = pd.DataFrame(
        {
            "Cenário": ["Mandante mais cartões", "Empate em cartões", "Visitante mais cartões"],
            "Probabilidade (%)": [
                round(info_cartoes["probabilidades"]["mandante"] * 100, 1),
                round(info_cartoes["probabilidades"]["empate"] * 100, 1),
                round(info_cartoes["probabilidades"]["visitante"] * 100, 1),
            ],
            "Expectativa de odd": [
                info_cartoes["odds"]["mandante"],
                info_cartoes["odds"]["empate"],
                info_cartoes["odds"]["visitante"],
            ],
        }
    )

    st.dataframe(cartoes_df, use_container_width=True, hide_index=True)

    col_cartoes_casa, col_cartoes_visitante = st.columns(2)
    col_cartoes_casa.metric("Cartões esperados mandante", f"{info_cartoes['esperados_mandante']:.2f}")
    col_cartoes_visitante.metric(
        "Cartões esperados visitante", f"{info_cartoes['esperados_visitante']:.2f}"
    )

    prob_cartoes = info_cartoes["probabilidades"]
    odds_cartoes = info_cartoes["odds"]

    contexto_chat = (
        f"Confronto: {time_casa} (mandante) x {time_visitante} (visitante).\n"
        f"Probabilidades de resultado: vitória mandante {probabilidades['mandante'] * 100:.1f}% (odd {odds['mandante']:.2f}), "
        f"empate {probabilidades['empate'] * 100:.1f}% (odd {odds['empate']:.2f}), "
        f"vitória visitante {probabilidades['visitante'] * 100:.1f}% (odd {odds['visitante']:.2f}).\n"
        f"Gols esperados: mandante {gols_esperados_casa:.2f}, visitante {gols_esperados_visitante:.2f}.\n"
        f"Probabilidade de ambos marcarem: {prob_ambos_marcam * 100:.1f}% (odd {odd_ambos_marcam:.2f}).\n"
        f"Cartões esperados: mandante {info_cartoes['esperados_mandante']:.2f}, visitante {info_cartoes['esperados_visitante']:.2f}.\n"
        f"Probabilidades relacionadas a cartões: mandante {prob_cartoes['mandante'] * 100:.1f}% (odd {odds_cartoes['mandante']:.2f}), "
        f"empate {prob_cartoes['empate'] * 100:.1f}% (odd {odds_cartoes['empate']:.2f}), "
        f"visitante {prob_cartoes['visitante'] * 100:.1f}% (odd {odds_cartoes['visitante']:.2f})."
    )

    st.subheader("Chatbot (Gemini)")
    st.caption("Faça pergunta ao nosso analisador GEMINI!")

    with st.form("chatbot_gemini_form"):
        pergunta = st.text_area(
            "Pergunta para o modelo",
            placeholder="Ex.: Qual mercado parece mais seguro? Ou: Quais riscos devo considerar?",
        )
        enviar = st.form_submit_button("Perguntar ao Gemini")

    if enviar:
        resposta = gerar_resposta_chatbot(contexto_chat, pergunta)
        st.markdown(resposta)


def validar_colunas(df: pd.DataFrame) -> None:
    faltantes = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if faltantes:
        st.warning(
            "Algumas colunas esperadas não foram encontradas no arquivo: "
            + ", ".join(faltantes)
        )


def main() -> None:
    st.set_page_config(page_title="Odd Calculator — Dashboard", layout="wide")
    st.title("Odd Calculator — Dashboard")
    st.caption("Painel interativo para acompanhar probabilidades e odds projetadas.")

    st.sidebar.header("Filtros & Ações")
    if st.sidebar.button("Limpar cache / Recarregar dados", key="sidebar_clear_cache"):
        st.cache_data.clear()
        st.experimental_rerun()

    if st.sidebar.button("Rodar análise agora", type="primary", key="sidebar_run_analysis"):
        rodar_analise_agora()

    if not os.path.exists(DATA_FILE):
        st.warning(
            "Arquivo de resultados não encontrado. Clique em 'Rodar análise agora' para gerar o CSV."
        )
        if st.button("Rodar análise agora", type="primary", key="main_run_analysis"):
            rodar_analise_agora()
        st.stop()

    try:
        df = load_data(DATA_FILE)
    except FileNotFoundError:
        st.error("Não foi possível localizar o arquivo de dados. Execute a análise novamente.")
        if st.button("Rodar análise agora", type="primary", key="error_run_analysis"):
            rodar_analise_agora()
        st.stop()
    except Exception as exc:
        st.error(f"Erro ao carregar o CSV: {exc}")
        st.stop()

    validar_colunas(df)

    df_filtrado = df.copy()

    kpi_cards(df_filtrado)

    tabs = st.tabs(["Visão Geral", "Simular Confronto"])

    with tabs[0]:
        tabela_top_n(df_filtrado, DEFAULT_TOP_N)
        charts_overview(df_filtrado)

    with tabs[1]:
        simular_confronto_tab(df)
    st.sidebar.markdown("---")
    st.sidebar.caption("Use os botões para atualizar a base sempre que necessário.")


if __name__ == "__main__":
    main()
