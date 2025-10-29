import pandas as pd
import os
import re
import unicodedata
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configura o Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

DATA_PATH = "brasileirao_2025_parcial.csv"
OUTPUT_PATH = "brasileirao_2025_odds.csv"


def carregar_dados():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def calcular_probabilidades(df):
    df["taxa_vitoria"] = (df["vitórias"] / df["partidas"]) * 100
    df["taxa_empate"] = (df["empates"] / df["partidas"]) * 100
    df["taxa_derrota"] = (df["derrotas"] / df["partidas"]) * 100

    df["prob_vitoria"] = df["taxa_vitoria"] / 100
    df["prob_empate"] = df["taxa_empate"] / 100
    df["prob_derrota"] = df["taxa_derrota"] / 100

    for coluna in ["prob_vitoria", "prob_empate", "prob_derrota"]:
        df[f"odd_{coluna.split('_')[1]}"] = df[coluna].apply(lambda x: 1 / x if x > 0 else 0)

    return df


def _normalizar_nome(nome: str) -> str:
    if not isinstance(nome, str):
        return ""
    texto = unicodedata.normalize("NFKD", nome)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", texto.lower())


def _extrair_analises_por_time(texto: str, nomes_times):
    """Mapeia a resposta livre da IA para cada time conhecido."""
    if not texto:
        return ["" for _ in nomes_times]

    nomes_norm = {_normalizar_nome(nome): nome for nome in nomes_times}
    analises = {chave: "" for chave in nomes_norm.keys()}

    corrente = None
    buffer = []

    linhas = [linha.strip() for linha in texto.splitlines()]

    for linha in linhas:
        if not linha:
            continue

        match = re.match(
            r"(?:[-•*]\s*)?(?:\*\*(?P<nome_bold>.+?)\*\*|(?P<nome_simples>[^:–-]+))\s*[:–-]\s*(?P<conteudo>.+)",
            linha,
        )

        if match:
            nome = match.group("nome_bold") or match.group("nome_simples") or ""
            chave = _normalizar_nome(nome.strip())

            if corrente and buffer and corrente in analises:
                analises[corrente] = " ".join(buffer).strip()

            if chave in analises:
                corrente = chave
                buffer = [match.group("conteudo").strip()]
            else:
                corrente = None
                buffer = []
            continue

        if corrente:
            buffer.append(linha.lstrip("-•* ").strip())

    if corrente and buffer and corrente in analises:
        analises[corrente] = " ".join(buffer).strip()

    if all(not valor for valor in analises.values()):
        linhas_validas = [linha for linha in linhas if linha and not linha.startswith(("Vamos analisar", "* "))]
        for chave, linha in zip(analises.keys(), linhas_validas):
            analises[chave] = linha

    return [analises.get(_normalizar_nome(nome), "") for nome in nomes_times]


def gerar_analises_em_lote(df):
    prompt = (
        "Atue como analista de futebol e apostador esportivo. Gere análises curtas "
        "(até 2 linhas) sobre cada time, citando estratégias com base nas probabilidades.\n"
        "Respeite a ordem fornecida e inicie cada análise com o nome do time seguido de dois-pontos.\n\n"
    )
    for _, r in df.iterrows():
        prompt += f"- {r['Time']}: vitória {r['taxa_vitoria']:.1f}%, empate {r['taxa_empate']:.1f}%, derrota {r['taxa_derrota']:.1f}%\n"

    try:
        resposta = model.generate_content(prompt)
        texto = getattr(resposta, "text", "") or ""
    except Exception as exc:
        print(f"[AVISO] Falha ao gerar análises com IA: {exc}")
        texto = ""

    df["analise_ia"] = _extrair_analises_por_time(texto, df["Time"])

    return df


def salvar_resultados(df):
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Resultados salvos em {OUTPUT_PATH}")


def executar_analise_completa():
    print("Executando análise de odds através de IA...")

    df = carregar_dados()
    df = calcular_probabilidades(df)
    df = gerar_analises_em_lote(df)
    salvar_resultados(df)

    print("Análise concluída com sucesso!")
