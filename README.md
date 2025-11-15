## EduScoreIA – Sistema Inteligente de Previsão de Notas com IA Híbrida

Sistema que combina Machine Learning (regressão) com Large Language Models (LLMs) para prever o desempenho de alunos em disciplinas do ensino médio e gerar recomendações pedagógicas personalizadas para professores, coordenação e estudantes.

### Visão Geral

O EduScoreIA integra um modelo de regressão para predições objetivas (nota prevista, probabilidade de aprovação, score de vestibular) com o GPT-4o-mini para análise conversacional contextual, criando uma experiência completa de acompanhamento de alunos:

- Previsão de notas

- Avaliação de risco de reprovação

- Nível de preparação para vestibular

- Recomendações específicas para professor e aluno


### Problema Resolvido

Apoiar decisões pedagógicas com base em dados e não apenas em percepção

Identificar risco de reprovação cedo (1º, 2º, 3º ano do EM)

Personalizar o plano de estudos por disciplina (Matemática, Português, Física etc.)

Conectar o boletim à preparação para vestibular (score de 0 a 100)

Explicar resultados de forma compreensível para educadores e estudantes

### Resultados Alcançados (exemplo com dados sintéticos)

- Modelo de regressão com R² alto (bom ajuste entre notas reais e previstas)
- API REST híbrida funcional: ML + LLM integrados
- Chatbot inteligente em modo CLI totalmente operacional
- Relatório automático de treino com métricas, gráficos e ranking de features
- Análise contextual por disciplina (Matemática, Física, Redação etc.)
- Sistema end-to-end: da geração de dados sintéticos até a conversa com o chatbot

### Principais Funcionalidades

- Autenticação personalizada com 10 usuários fictícios do contexto escolar
(professores, coordenador pedagógico, orientadora educacional)

- Coleta conversacional dos dados do aluno (notas, presença, estudo, simulados)

- Cálculo automático de:

  - nota prevista (0 a 10)

  - probabilidade de aprovação

  - risco (baixo / moderado / alto)

  - score de preparação para vestibular (0 a 100)

- Análise híbrida: modelo de ML + GPT-4o-mini explicando o caso

- Recomendações específicas:

  - bloco para o professor

  - bloco para o aluno (foco em vestibular e tópicos da disciplina)

- Interface CLI intuitiva, barras de progresso e textos bem formatados

### Especificações Técnicas
| Componente     | Tecnologia                                    | Destaque                                   |
| -------------- | --------------------------------------------- | ------------------------------------------ |
| **Modelo ML**  | Regressão                                     | R² alto + bons erros (MSE/RMSE/MAE)        |
| **LLM**        | GPT-4o-mini via OpenAI API                    | Análise pedagógica contextual              |
| **API**        | FastAPI + Uvicorn                             | Endpoints `/prever` e `/analisar-com-llm` |
| **Chatbot**    | Python CLI + OpenAI Client                    | Coleta, resumo e exibição dos resultados   |
| **Dataset**    | Alunos sintéticos gerados em `treinamento_modelo.py` | ensino médio + disciplinas                 |
| **Relatórios** | `modelo_report.md`, gráficos `.png`            | avaliação visual do modelo                 |


### Início Rápido (5 minutos)
Pré-requisitos

- Python 3.10+

- Chave da OpenAI (OPENAI_API_KEY)

- pip instalado

1. Clonar o projeto

````bash
git clone https://github.com/sua-conta/EduScoreIA.git
cd EduScoreIA
````

2. Configurar a chave da OpenAI
````bash
echo "OPENAI_API_KEY=sua_chave_aqui" > .env
# opcional: modelinho
echo "OPENAI_MODEL=gpt-4o-mini" >> .env
````

3. Instalar dependências (na raiz do projeto)
````bash
pip install -r modelo_ml/requirements.txt
pip install -r api/requirements.txt
pip install -r chatbot/requirements.txt
````

4. Treinar o modelo (gera dados + métricas + artefatos)
````bash
cd modelo_ml
python treinamento_modelo.py
````

Isso gera em modelo_ml/models/:

- eduscore_modelo.pkl – modelo treinado

- eduscore_scaler.pkl – scaler usado no treino

- feature_columns.json – ordem das features

- metricas.json – métricas (MSE, RMSE, MAE, R² etc.)

- modelo_report.md – relatório em markdown

- erro_histograma.png, dispersao_real_vs_pred.png – gráficos


5. Subir a API híbrida
````bash
cd api
python app.py
# ou
# uvicorn api.app:app --reload
````

Verificar se está tudo ok:
````bash
curl http://localhost:8000/health
````

6. Rodar o chatbot EduScore
````bash
cd chatbot
python chatbot_llm.py
````

Fluxo:

  1. escolhe um usuário (ex.: Professora de Matemática)

  2. informa os dados do aluno

  3. o chatbot mostra o resumo

  4. você escolhe analisar com IA

  5. ele exibe os resultados + recomendações

### Componente 1: Modelo de Machine Learning
Características

- Tarefa: regressão de nota final em uma disciplina (0–10)

- Entradas (exemplos):

  - série (1º, 2º, 3º ano)

  - disciplina (Matemática, Português, Física, etc.)

  - perfil global (forte, mediano, em_risco)

  - notas parciais (prova 1, prova 2, projeto, listas)

  - presença (%)

  - horas de estudo semanais

  - nº de acessos à plataforma

  - nº de simulados

  - dificuldade percebida (1–5)


A partir da nota prevista, o sistema calcula:

- prob_aprovacao – probabilidade de aprovação na disciplina

- risco – baixo / moderado / alto

- vestibular_score – 0 a 100, agregando nota, presença, estudos, simulados e série

- nivel_preparacao – alta / moderada / baixa / crítica

### Componente 2: Large Language Model (LLM)
Função do LLM (GPT-4o-mini)

Para cada caso de aluno, o LLM recebe:

- dados brutos do aluno (entrada)

- resultados do modelo (nota prevista, risco, score, nível de preparação)

- tópicos importantes daquela disciplina (currículo do ensino médio)

E responde com 3 blocos:

[EXPLICACAO] – interpreta os números de forma humana

[RECOMENDACOES_PROFESSOR] – estratégias de aula, reforço, acompanhamento

[RECOMENDACOES_ALUNO] – plano de estudo, tópicos prioritários, foco em vestibular

O chatbot exibe esses blocos separadamente, deixando claro:

- o que é saída do modelo numérico

- o que é análise qualitativa da IA

### Componente 3: API REST Híbrida
Tecnologia

- Framework: FastAPI

- Servidor: Uvicorn

- Validação de dados: Pydantic

- Documentação automática: Swagger UI (/docs)

Endpoints principais
| Endpoint            | Método | Descrição                              |
| ------------------- | ------ | -------------------------------------- |
| `/`                 | GET    | Status básico da API                   |
| `/health`           | GET    | Health check (modelo + scaler + LLM)   |
| `/prever`          | POST   | Previsão numérica (nota, risco, score) |
| `/analisar-com-llm` | POST   | **Análise híbrida** (ML + LLM)         |
| `/docs`             | GET    | Interface Swagger interativa           |


Exemplo de chamada híbrida
````bash
curl -X POST "http://localhost:8000/analisar-com-llm" \
  -H "Content-Type: application/json" \
  -d '{
    "serie": 1,
    "disciplina": "Matemática",
    "perfil_base": "forte",
    "nota_prova1": 7,
    "nota_prova2": 8,
    "nota_projeto": 9,
    "nota_listas": 8,
    "presenca_percentual": 90,
    "estudos_semanais_horas": 2,
    "acessos_plataforma": 2,
    "simulados_feitos": 1,
    "dificuldade_disciplina": 3
  }'
````

### Componente 4: Chatbot Inteligente (CLI)
O que ele faz

- Autentica um usuário (professores, coordenação, orientação)

- Coleta os dados do aluno de forma conversacional

- Gera um resumo inteligente usando o LLM

- Pergunta se você quer rodar a análise com IA

- Chama a API /analisar-com-llm e exibe:

  - Nota prevista

  - Probabilidade de aprovação (com barra visual)

  - Risco na disciplina

  - Score de vestibular e nível de preparação

  - Explicação do caso

  - Recomendações para professor

  - Recomendações para o aluno


### Tecnologias Utilizadas
````text
IA & ML
- scikit-learn      – modelos de regressão e métricas
- pandas / numpy    – manipulação de dados
- joblib            – salvamento de modelos

LLM
- openai (client v1) – GPT-4o-mini via API

API
- fastapi
- uvicorn
- pydantic
- python-dotenv
- requests
````

### Conclusão

O EduScoreIA mostra como combinar:

- Modelos preditivos numéricos (ML)

- IA generativa conversacional (LLM)

- Contexto educacional real (disciplinas, séries, vestibular)

para criar uma ferramenta que:

- Ajuda professores e coordenação a tomar decisões melhores

- Dá feedback compreensível para alunos

- Conecta o dia a dia da sala de aula ao objetivo final: aprovação e vestibular


### Autor

Projeto desenvolvido para estudo e demonstração de competências em Machine Learning e Large Language Models aplicada à Educação.
| [<img src="https://avatars.githubusercontent.com/u/55546267?v=4" width=115><br><sub>Priscila Miranda</sub>](https://github.com/priscilafraser) |
| :---: |



