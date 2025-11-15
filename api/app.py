# api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

###########################
# Setup de paths e ambiente
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
ML_MODELS_DIR = BASE_DIR / "modelo_ml" / "models"
ENV_PATH = BASE_DIR / ".env"
###########################

load_dotenv(ENV_PATH)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Carrega modelo
try:
    model = joblib.load(ML_MODELS_DIR / "eduscore_modelo.pkl")
    scaler = joblib.load(ML_MODELS_DIR / "eduscore_scaler.pkl")

    with open(ML_MODELS_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    print("Modelo EduScore carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    model = None
    scaler = None
    feature_columns = []


# Modelos pydantic
class DadosAluno(BaseModel):
    serie: int
    disciplina: str
    perfil_base: str = "mediano"

    nota_prova1: float
    nota_prova2: float
    nota_projeto: float
    nota_listas: float

    presenca_percentual: float
    estudos_semanais_horas: float
    acessos_plataforma: int
    simulados_feitos: int
    dificuldade_disciplina: float


class RespostaPrevisaoML(BaseModel):
    nota_prevista: float
    prob_aprovacao: float
    risco: str
    vestibular_score: float
    nivel_preparacao: str


class RespostaAnaliseLLM(BaseModel):
    ml_predicao: Dict[str, Any]
    analise_llm: Dict[str, str]
    insights_combinados: str


# Funções auxiliares
def preprocessamento_aluno(dict_alunos: dict) -> np.ndarray:
    df = pd.DataFrame([dict_alunos])

    df = pd.get_dummies(
        df,
        columns=["disciplina", "perfil_base"],
        drop_first=True,
    )
    df = df.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df.values)
    return X_scaled


def calcular_probabilidade_aprovacao(nota_prevista: float) -> float:
    x = nota_prevista - 6.0
    prob = 1.0 / (1.0 + np.exp(-x))
    return float(np.clip(prob, 0.0, 1.0))


def classificar_risco(prob: float) -> str:
    if prob >= 0.85:
        return "baixo"
    if prob >= 0.60:
        return "moderado"
    return "alto"


def calcular_vestibular_score(aluno: dict, nota_prevista: float) -> float:
    serie = aluno["serie"]
    presenca = aluno["presenca_percentual"]
    estudos = aluno["estudos_semanais_horas"]
    simulados = aluno["simulados_feitos"]

    base_nota = nota_prevista * 10.0
    bonus_presenca = (presenca - 75.0) * 0.6
    bonus_estudos = estudos * 3.0
    bonus_simulados = simulados * 2.5
    bonus_serie = (serie - 1) * 5.0

    score = base_nota + bonus_presenca + bonus_estudos + bonus_simulados + bonus_serie
    return float(np.clip(score, 0.0, 100.0))


def classificar_preparacao(score: float) -> str:
    if score >= 80:
        return "alta"
    if score >= 60:
        return "moderada"
    if score >= 40:
        return "baixa"
    return "crítica"


# Conteúdo programático por disciplina para LLM
CURRICULO_DISCIPLINAS = {
    "Matemática": [
        "funções (afim, quadrática, exponencial, logarítmica)",
        "geometria plana e espacial",
        "probabilidade e estatística básica",
        "progressões aritméticas e geométricas",
        "trigonometria",
    ],
    "Português": [
        "interpretação de textos",
        "gramática normativa (concordância, regência, crase)",
        "semântica e figuras de linguagem",
        "tipos textuais",
        "literatura brasileira",
    ],
    "Física": [
        "cinemática",
        "leis de Newton",
        "trabalho e energia",
        "eletrodinâmica",
        "óptica geométrica",
    ],
    "Química": [
        "ligações químicas",
        "estequiometria",
        "termoquímica",
        "equilíbrios químicos",
        "eletroquímica",
    ],
    "Biologia": [
        "citologia",
        "genética",
        "ecologia",
        "evolução",
        "fisiologia humana",
    ],
    "História": [
        "Brasil Colônia",
        "Brasil Império e República",
        "Idade Moderna",
        "Idade Contemporânea",
        "ditadura militar no Brasil",
    ],
    "Geografia": [
        "geografia física (relevo, clima, vegetação)",
        "geopolítica",
        "população e urbanização",
        "economia e globalização",
        "meio ambiente",
    ],
    "Inglês": [
        "interpretação de textos em inglês",
        "vocabulário básico para ENEM/vestibulares",
        "estruturas gramaticais mais cobradas",
    ],
    "Redação": [
        "estrutura de texto dissertativo-argumentativo",
        "tese e argumentos",
        "coesão e coerência",
        "proposta de intervenção",
        "competências do ENEM",
    ],
}


def montar_prompt_estudante(
    dados_estudante: Dict[str, Any],
    nota_prevista: float,
    prob_aprovacao: float,
    risco: str,
    vestibular_score: float,
    nivel_preparacao: str,
) -> str:
    disciplina = dados_estudante.get("disciplina", "Disciplina não informada")
    serie = dados_estudante.get("serie", None)

    topicos_disciplina = CURRICULO_DISCIPLINAS.get(disciplina, [])
    topicos_texto = (
        "\n".join(f"- {t}" for t in topicos_disciplina)
        if topicos_disciplina
        else "Sem tópicos cadastrados para essa disciplina."
    )

    dados_formatados = "\n".join(
        f"- {chave}: {valor}" for chave, valor in dados_estudante.items()
    )

    prompt = f"""
Você é um orientador pedagógico e coach para vestibular, especializado em ensino médio.

Você recebeu dados de um aluno em uma disciplina específica e resultados de um modelo
de Machine Learning que estima o desempenho dele.

DADOS DO ALUNO (ENTRADA DO SISTEMA):
{dados_formatados}

INFORMAÇÕES DA DISCIPLINA:
- Disciplina: {disciplina}
- Série: {serie}º ano do ensino médio

RESULTADOS DO MODELO:
- Nota prevista na disciplina (0 a 10): {nota_prevista:.2f}
- Probabilidade de aprovação: {prob_aprovacao:.2%}
- Nível de risco de reprovação: {risco.upper()}

INDICADOR ESPECÍFICO DE VESTIBULAR:
- Score de preparação para vestibular (0 a 100): {vestibular_score:.1f}
- Nível de preparação: {nivel_preparacao.upper()}

CONTEÚDOS MAIS IMPORTANTES PARA VESTIBULAR NESSA DISCIPLINA:
{topicos_texto}

Sua tarefa é devolver uma resposta ESTRUTURADA em 3 blocos:

[EXPLICACAO]
Explique de forma clara:
- por que o aluno está com essa nota prevista e score de preparação;
- quais fatores mais pesam (presença, horas de estudo, simulados, notas parciais);
- o que isso significa na prática para a disciplina e para o vestibular.

[RECOMENDACOES_PROFESSOR]
Liste recomendações específicas para o PROFESSOR dessa disciplina:
- estratégias em sala ou online;
- tipos de atividades e revisões;
- uso de simulados, listas, projetos;
- como apoiar esse aluno considerando o nível de risco e a série.

[RECOMENDACOES_ALUNO]
Liste recomendações específicas para o ALUNO, com foco em vestibular:
- tópicos da disciplina que ele deveria priorizar (use os tópicos listados acima);
- plano de estudo semanal sugerido (de forma simples);
- sugestão de como usar simulados e revisão;
- tom motivador, mas realista.

IMPORTANTE:
- Use linguagem simples e direta.
- Não invente notas ou dados que não existem.
- Deixe os blocos bem separados visualmente.
"""
    return prompt


def extrair_blocos(texto: str) -> Dict[str, str]:
    blocos = {
        "explicacao": "",
        "recomendacoes_professor": "",
        "recomendacoes_aluno": "",
    }

    texto_normalizado = (
        texto.replace("[Explicacao]", "[EXPLICACAO]")
        .replace("[Explicação]", "[EXPLICACAO]")
        .replace("[Recomendacoes_Professor]", "[RECOMENDACOES_PROFESSOR]")
        .replace("[Recomendações_Professor]", "[RECOMENDACOES_PROFESSOR]")
        .replace("[Recomendacoes_Aluno]", "[RECOMENDACOES_ALUNO]")
        .replace("[Recomendações_Aluno]", "[RECOMENDACOES_ALUNO]")
    )

    partes = texto_normalizado.split("[EXPLICACAO]")
    if len(partes) == 1:
        blocos["explicacao"] = texto.strip()
        return blocos

    depois_explicacao = partes[1]
    partes_prof = depois_explicacao.split("[RECOMENDACOES_PROFESSOR]")
    blocos["explicacao"] = partes_prof[0].strip()

    if len(partes_prof) > 1:
        depois_prof = partes_prof[1]
        partes_aluno = depois_prof.split("[RECOMENDACOES_ALUNO]")

        blocos["recomendacoes_professor"] = partes_aluno[0].strip()

        if len(partes_aluno) > 1:
            blocos["recomendacoes_aluno"] = partes_aluno[1].strip()

    return blocos


def chamar_llm(prompt: str) -> Dict[str, str]:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você ajuda professores e alunos do ensino médio a entender "
                    "previsões de desempenho e a se prepararem melhor para vestibulares."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
        temperature=0.7,
    )
    texto = response.choices[0].message.content
    return extrair_blocos(texto)



# FastAPI app
app = FastAPI(
    title="EduScore IA - Previsão de Notas e Análise com LLM",
    description="API para prever desempenho escolar por disciplina usando ML + LLM",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "EduScore IA API funcionando",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "erro",
        "modelo_carregado": model is not None,
        "scaler_carregado": scaler is not None,
        "qtd_features": len(feature_columns),
        "llm_configurado": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/predict", response_model=RespostaPrevisaoML)
def predict(aluno: DadosAluno):
    if model is None or scaler is None or not feature_columns:
        raise HTTPException(
            status_code=500,
            detail="Modelo ou artefatos não carregados. Rode primeiro o treino.",
        )

    try:
        dict_alunos = aluno.dict()
        X = preprocessamento_aluno(dict_alunos)
        nota_prevista = float(model.predict(X)[0])

        prob = calcular_probabilidade_aprovacao(nota_prevista)
        risco = classificar_risco(prob)
        vestibular_score = calcular_vestibular_score(dict_alunos, nota_prevista)
        nivel_preparacao = classificar_preparacao(vestibular_score)

        return RespostaPrevisaoML(
            nota_prevista=nota_prevista,
            prob_aprovacao=prob,
            risco=risco,
            vestibular_score=vestibular_score,
            nivel_preparacao=nivel_preparacao,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint híbrido: ML + LLM
@app.post("/analisar-com-llm", response_model=RespostaAnaliseLLM)
def analisar_com_llm(aluno: DadosAluno):

    # Primeiro, faz predição com ML
    ml_pred = predict(aluno)
    dict_alunos = aluno.dict()

    # Monta prompt e chama LLM
    prompt = montar_prompt_estudante(
        dados_estudante=dict_alunos,
        nota_prevista=ml_pred.nota_prevista,
        prob_aprovacao=ml_pred.prob_aprovacao,
        risco=ml_pred.risco,
        vestibular_score=ml_pred.vestibular_score,
        nivel_preparacao=ml_pred.nivel_preparacao,
    )

    try:
        blocos = chamar_llm(prompt)
        return RespostaAnaliseLLM(
            ml_predicao=ml_pred.dict(),
            analise_llm=blocos,
            insights_combinados="Análise híbrida (ML + LLM) concluída com sucesso.",
        )
    except Exception as e:
        # fallback: retorna só ML
        return RespostaAnaliseLLM(
            ml_predicao=ml_pred.dict(),
            analise_llm={
                "explicacao": f"Análise LLM indisponível: {str(e)}",
                "recomendacoes_professor": "",
                "recomendacoes_aluno": "",
            },
            insights_combinados="Análise realizada apenas com o modelo de ML.",
        )


if __name__ == "__main__":
    import uvicorn

    # Rodar diretamente com: python api/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
