# modelo_ml/treinamento_modelo.py

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
DATA_DIR = THIS_DIR / "data"
MODELS_DIR = THIS_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DISCIPLINAS = [
    "Matemática",
    "Português",
    "Física",
    "Química",
    "Biologia",
    "História",
    "Geografia",
    "Inglês",
    "Redação",
]


# Gerar com cada aluno com registros em várias disciplinas, considera serie, etc
def gerar_dados_sinteticos(n_alunos: int = 400, seed: int = 42) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    registros = []

    for aluno_id in range(1, n_alunos + 1):
        # Série do ensino médio (1, 2, 3)
        serie = rng.integers(1, 4)

        # "Perfil" base do aluno (bom, médio, em risco)
        perfil_base = rng.choice(["forte", "mediano", "em_risco"], p=[0.3, 0.5, 0.2])

        # Parâmetros globais do aluno que impactam todas as disciplinas
        if perfil_base == "forte":
            presenca_media = rng.normal(92, 4)
            estudos_semanais_base = rng.normal(8, 2)
            simulados_base = rng.integers(4, 9)
        elif perfil_base == "mediano":
            presenca_media = rng.normal(85, 7)
            estudos_semanais_base = rng.normal(5, 2)
            simulados_base = rng.integers(2, 7)
        else:
            presenca_media = rng.normal(75, 10)
            estudos_semanais_base = rng.normal(2.5, 1.5)
            simulados_base = rng.integers(0, 4)

        for disciplina in DISCIPLINAS:
            # Ajuste de dificuldade por disciplina
            if disciplina in ["Matemática", "Física", "Química"]:
                dificuldade_media = 3.5  # mais difícil
            elif disciplina in ["Biologia", "História", "Geografia"]:
                dificuldade_media = 3.0
            else:
                dificuldade_media = 2.8

            dificuldade_disciplina = np.clip(
                rng.normal(dificuldade_media, 0.7), 1.0, 5.0
            )

            # Engajamento específico por disciplina
            presenca_percentual = np.clip(
                rng.normal(presenca_media, 5), 50, 100
            )

            estudos_semanais_horas = max(
                0.0, rng.normal(estudos_semanais_base, 1.5)
            )

            acessos_plataforma = max(
                0, int(rng.normal(30 + estudos_semanais_horas * 3, 10))
            )

            simulados_feitos = int(
                np.clip(rng.normal(simulados_base, 2), 0, 10)
            )

            # Notas parciais variam com engajamento + dificuldade
            base_nivel = {
                "forte": 7.2,
                "mediano": 6.0,
                "em_risco": 4.8,
            }[perfil_base]

            ajuste_engajamento = (presenca_percentual - 80) / 15.0
            ajuste_estudo = (estudos_semanais_horas - 4) * 0.3
            ajuste_simulados = simulados_feitos * 0.2
            ajuste_dificuldade = -(dificuldade_disciplina - 3) * 0.4

            centro = (
                base_nivel
                + ajuste_engajamento
                + ajuste_estudo
                + ajuste_simulados
                + ajuste_dificuldade
            )

            nota_prova1 = np.clip(
                rng.normal(centro, 1.5), 0, 10
            )
            nota_prova2 = np.clip(
                rng.normal(centro + 0.3, 1.5), 0, 10
            )
            nota_projeto = np.clip(
                rng.normal(centro + 0.5, 1.2), 0, 10
            )
            nota_listas = np.clip(
                rng.normal(centro + 0.2, 1.0), 0, 10
            )

            # Nota final com mais regras
            nota_final_raw = (
                0.35 * nota_prova1
                + 0.35 * nota_prova2
                + 0.15 * nota_projeto
                + 0.15 * nota_listas
            )

            # bônus por engajamento e simulados
            nota_final_raw += (presenca_percentual - 75) / 20.0
            nota_final_raw += estudos_semanais_horas * 0.15
            nota_final_raw += simulados_feitos * 0.1

            # penalidade se dificuldade alta e engajamento baixo
            if dificuldade_disciplina > 3.5 and estudos_semanais_horas < 4:
                nota_final_raw -= 0.6

            # ruído final
            nota_final = np.clip(
                rng.normal(nota_final_raw, 0.8), 0, 10
            )

            registros.append(
                {
                    "aluno_id": aluno_id,
                    "serie": serie,
                    "disciplina": disciplina,
                    "perfil_base": perfil_base,
                    "presenca_percentual": presenca_percentual,
                    "estudos_semanais_horas": estudos_semanais_horas,
                    "acessos_plataforma": acessos_plataforma,
                    "simulados_feitos": simulados_feitos,
                    "dificuldade_disciplina": dificuldade_disciplina,
                    "nota_prova1": nota_prova1,
                    "nota_prova2": nota_prova2,
                    "nota_projeto": nota_projeto,
                    "nota_listas": nota_listas,
                    "nota_final": nota_final,
                }
            )

    df = pd.DataFrame(registros)
    return df

# Cria dataset de usuários do sistema EduScoreIA (professores e equipe pedagógica)
def criar_dados_usuarios():

    dados_usuarios = {
        "usuario_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        
        "nome": [
            "Priscila Andrade",
            "Carlos Menezes",
            "Mariana Duarte",
            "Fernando Albuquerque",
            "Ana Lúcia Campos",
            "João Pedro Martins",
            "Camila Ribeiro",
            "Sofia Ferreira",
            "Rodrigo Vasconcelos",
            "Bianca Moreira"
        ],
        
        "cargo": [
            "Professora de Matemática",
            "Coordenador Pedagógico",
            "Orientadora Educacional",
            "Professor de Física",
            "Professora de Português",
            "Professor de Química",
            "Professora de Biologia",
            "Professora de História",
            "Professor de Geografia",
            "Professora de Inglês"
        ],
        
        "experiencia_anos": [3, 10, 6, 8, 12, 4, 5, 7, 9, 3],

        "series_responsaveis": [
            "1º e 2º ano",   # matemática
            "1º ao 3º ano",  # coordenação
            "1º ao 3º ano",  # orientação
            "2º e 3º ano",   # física
            "1º e 3º ano",   # português
            "1º ano",        # química
            "1º e 2º ano",   # biologia
            "1º e 2º ano",   # história
            "1º ao 3º ano",  # geografia
            "1º e 2º ano"    # inglês
        ],

        "especialidade": [
            "Álgebra e Geometria",
            "Currículo e Avaliação",
            "Acompanhamento socioemocional",
            "Mecânica e Eletricidade",
            "Redação e Interpretação",
            "Reações e Termoquímica",
            "Genética e Fisiologia",
            "Brasil Colônia e República",
            "Geopolítica e Meio Ambiente",
            "Reading & Grammar"
        ],

        # experiência com acompanhamento individual (0 a 100)
        "eficacia_acompanhamento": [82, 90, 88, 75, 92, 70, 78, 85, 80, 74]
    }

    return pd.DataFrame(dados_usuarios)


# Treina vários modelos de regressão
def treinar_modelo(df: pd.DataFrame):
    
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        mean_absolute_error,
        make_scorer,
    )
    import matplotlib.pyplot as plt

    ################
    # Cria dados
    df_usuarios = criar_dados_usuarios()
    
    # Salva dados
    df_usuarios.to_csv('data/dados_usuarios.csv', index=False)
    #########################

    # Preparação dos dados
    df_features = df.drop(columns=["nota_final"])
    y = df["nota_final"]

    # One-hot em disciplina e perfil_base
    df_features = pd.get_dummies(
        df_features,
        columns=["disciplina", "perfil_base"],
        drop_first=True,
    )

    feature_columns = list(df_features.columns)
    X = df_features.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definição dos modelos utilizados
    candidatos = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=42,
        ),
        "LinearRegression": LinearRegression(),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_score = make_scorer(mean_absolute_error, greater_is_better=False)

    resultados_cv = []

    print("\n=== AVALIAÇÃO POR CROSS-VALIDATION (apenas treino) ===")
    for nome, modelo in candidatos.items():
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", modelo),
            ]
        )

        cv_mse = cross_val_score(
            pipe, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
        )
        cv_mae = cross_val_score(pipe, X_train, y_train, cv=kf, scoring=mae_score)

        mse_mean = -cv_mse.mean()
        rmse_mean = np.sqrt(mse_mean)
        mae_mean = -cv_mae.mean()

        resultados_cv.append(
            {
                "modelo": nome,
                "mse_mean": mse_mean,
                "rmse_mean": rmse_mean,
                "mae_mean": mae_mean,
            }
        )

        print(f"\nModelo: {nome}")
        print(f"  RMSE médio (cv=5): {rmse_mean:.4f}")
        print(f"  MAE  médio (cv=5): {mae_mean:.4f}")

    # Escolhe o melhor modelo pelo menor RMSE médio
    resultados_cv_sorted = sorted(resultados_cv, key=lambda d: d["rmse_mean"])
    melhor = resultados_cv_sorted[0]
    melhor_nome = melhor["modelo"]

    print("\n=== MELHOR MODELO PELA CROSS-VAL ===")
    print(
        f"Modelo escolhido: {melhor_nome} "
        f"(RMSE médio={melhor['rmse_mean']:.4f}, MAE médio={melhor['mae_mean']:.4f})"
    )

    # Treino final com o melhor modelo
    # 1) cria scaler separado, o que será salvo e utilizado
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    # 2) instancia o mesmo tipo de modelo novamente
    if melhor_nome == "RandomForest":
        modelo_final = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif melhor_nome == "GradientBoosting":
        modelo_final = GradientBoostingRegressor(random_state=42)
    elif melhor_nome == "HistGradientBoosting":
        modelo_final = HistGradientBoostingRegressor(random_state=42)
    else:
        modelo_final = LinearRegression()

    modelo_final.fit(X_train_scaled, y_train)

    # Avaliação no conjunto de teste (hold-out)
    y_pred = modelo_final.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== MÉTRICAS NO CONJUNTO DE TESTE ===")
    print(f"Modelo final: {melhor_nome}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")

    # Importância das features, se disponível
    importances = None
    if hasattr(modelo_final, "feature_importances_"):
        importances = modelo_final.feature_importances_
    elif hasattr(modelo_final, "coef_"):
        coef = modelo_final.coef_
        importances = np.abs(coef)  # usa módulo dos coeficientes

    if importances is not None:
        ranking = sorted(
            zip(feature_columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        print("\nTop 5 features mais importantes:")
        for nome, imp in ranking:
            print(f"- {nome}: {imp:.3f}")
    else:
        ranking = []
        print("\nModelo não fornece importância de features diretamente.")

    # Salvando previsões
    resultados = pd.DataFrame(
        {
            "y_real": y_test,
            "y_pred": y_pred,
            "erro": y_test - y_pred,
        }
    )
    resultados_path = MODELS_DIR / "predicoes_teste.csv"
    resultados.to_csv(resultados_path, index=False)

    # Salvando métricas em JSON
    report_json = {
        "modelo_escolhido": melhor_nome,
        "cv_resultados": resultados_cv_sorted,
        "teste_mse": float(mse),
        "teste_rmse": float(rmse),
        "teste_mae": float(mae),
        "teste_r2": float(r2),
    }
    with open(MODELS_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=4)

    # Relatório em Markdown
    cv_md = ""
    for r in resultados_cv_sorted:
        cv_md += (
            f"- {r['modelo']}: RMSE (cv)={r['rmse_mean']:.4f}, "
            f"MAE (cv)={r['mae_mean']:.4f}\n"
        )

    if ranking:
        features_md = "".join([f"- {nome}: {imp:.4f}\n" for nome, imp in ranking])
    else:
        features_md = "_Não disponível para este modelo._\n"

    md = f"""# Relatório de Treinamento – EduScoreIA

    
## Modelos avaliados (cross-validation, apenas treino)
{cv_md}
## Modelo escolhido
- **Nome:** {melhor_nome}

## Métricas no conjunto de teste (hold-out)
- **MSE:** {mse:.4f}
- **RMSE:** {rmse:.4f}
- **MAE:** {mae:.4f}
- **R²:** {r2:.4f}

## Importância das Features (Top 5)
{features_md}
## Arquivos Gerados
- `predicoes_teste.csv`: resultados de previsão no teste
- `metricas.json`: métricas + resumo da cross-validation
- `eduscore_modelo.pkl`: modelo treinado final
- `eduscore_scaler.pkl`: scaler do treino
- `feature_columns.json`: ordem das features
"""

    with open(MODELS_DIR / "modelo_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # Gráficos de desempenho
    # Real vs previsto
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Nota Real")
    plt.ylabel("Nota Prevista")
    plt.title(f"Real vs Previsto – {melhor_nome}")
    plt.grid(True)
    plt.savefig(MODELS_DIR / "dispersao_real_vs_pred.png", dpi=300)
    plt.close()

    # Distribuição do erro
    plt.figure(figsize=(6, 4))
    plt.hist(resultados["erro"], bins=30, alpha=0.7)
    plt.title("Distribuição do Erro (y_real - y_pred)")
    plt.xlabel("Erro")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.savefig(MODELS_DIR / "erro_histograma.png", dpi=300)
    plt.close()

    return modelo_final, scaler_final, feature_columns



def salvar_artefatos(model, scaler, feature_columns, df: pd.DataFrame):
    model_path = MODELS_DIR / "eduscore_modelo.pkl"
    scaler_path = MODELS_DIR / "eduscore_scaler.pkl"
    features_path = MODELS_DIR / "feature_columns.json"
    data_path = DATA_DIR / "dados_sinteticos_alunos.csv"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    df.to_csv(data_path, index=False)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {model_path}")
    print(f"Scaler salvo em: {scaler_path}")
    print(f"Feature columns em: {features_path}")
    print(f"Dataset sintético em: {data_path}")


# Função principal, fluxo principal
def main():
    print("Gerando dados sintéticos (multi-disciplinas)...")
    df = gerar_dados_sinteticos()
    print(f"Total de registros: {len(df)}")

    print("\nTreinando modelo de previsão de notas...")
    model, scaler, feature_columns = treinar_modelo(df)

    print("\nSalvando artefatos...")
    salvar_artefatos(model, scaler, feature_columns, df)


if __name__ == "__main__":
    main()
