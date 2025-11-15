import os
import time
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Carrega .env (na raiz do projeto)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatbotEstudanteLLM:

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.usuario_atual = None
        self.dados_aluno = {}
        self.perguntas_feitas = 0  # evitar ficar dando muitos "olá"


    # Usa GPT-4o-mini (ou outro modelo definido) para respostas inteligentes durante a conversa
    def obter_resposta_llm(self, msg_do_usuario: str, contexto: str = "") -> str:

        prompt_sistema = f"""
        Você é o EduScore, um assistente de IA especializado em educação básica
        e preparação para vestibular.

        Seu principal objetivo é ajudar professores, coordenadores e estudantes
        a entenderem o desempenho em disciplinas do ensino médio, com foco em:
        - notas previstas
        - engajamento (presença, estudo, simulados)
        - preparação para ENEM e vestibulares

        Contexto: {contexto}

        Regras:
        - Seja conversacional, claro e profissional.
        - Pode usar emojis de forma moderada.
        - Não repita cumprimentos como "Olá" ou "Oi" toda hora.
        - Quando fizer perguntas sobre dados, explique POR QUE essa informação é importante.
        - Nunca invente números, notas ou diagnósticos clínicos.
        """

        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": msg_do_usuario},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception:
            return "Estou com uma instabilidade técnica agora, mas podemos continuar com o fluxo padrão."


    def msg_boas_vindas(self):
        print("=" * 60)
        print("CHATBOT INTELIGENTE DE PREVISÃO DE NOTAS & VESTIBULAR")
        print("=" * 60)

        msg = """
        Seja bem-vindo! Sou o EduScore, seu assistente inteligente.

        Vou te ajudar a analisar o desempenho de um aluno em uma disciplina
        do ensino médio, prevendo nota, risco de reprovação e nível de preparação
        para vestibular, usando:
        • Modelos de Machine Learning treinados com dados sintéticos
        • Análise de IA (LLM) focada na disciplina
        • Recomendações específicas para professor e estudante

        Primeiro, vou entender quem está usando o sistema.
        """

        response = self.obter_resposta_llm(msg)
        print(f"\n: {response}")

    #Autenticação simples de usuário a partir de um CSV
    def autenticar_usuario(self):

        try:
            usuario_path = BASE_DIR / "modelo_ml" / "data" / "dados_usuarios.csv"
            usuarios_df = pd.read_csv(usuario_path)

            print("\nUsuários disponíveis:")
            for _, user in usuarios_df.iterrows():
                print(f"  {user['usuario_id']}. {user['nome']} ({user['cargo']})")

            while True:
                try:
                    usuario_id = int(input("\nDigite seu ID de usuário: "))
                    user = usuarios_df[usuarios_df["usuario_id"] == usuario_id]

                    if not user.empty:
                        self.usuario_atual = user.iloc[0].to_dict()

                        contexto = (
                            f"Usuário: {self.usuario_atual['nome']}, "
                            f"{self.usuario_atual['cargo']}, "
                            f"{self.usuario_atual['experiencia_anos']} anos de experiência."
                        )
                        greeting_msg = (
                            "O usuário acabou de fazer login. "
                            "Cumprimente-o e comente brevemente como a experiência dele "
                            "pode ajudar na análise de alunos e disciplinas."
                        )

                        greeting = self.obter_resposta_llm(greeting_msg, contexto)
                        print(f"\n >> {greeting}")
                        return True
                    else:
                        print("❌ Usuário não encontrado. Tente novamente.")
                except ValueError:
                    print("❌ Digite um número válido.")
        except Exception as e:
            print(f"❌ Erro ao carregar usuários: {e}")
            return False


    # Gera perguntas específicas para cada campo, aproveitando LLM ou perguntas fixas
    def obter_pergunta(self, campo: str, config: dict) -> str:

        # Perguntas pré-definidas (evita sobrecarregar o LLM e mantém consistência)
        perguntas_predefinidas = {
            "serie": (
                "Em que série do ensino médio o aluno está (1º, 2º ou 3º ano)? "
                "Isso é importante porque a cobrança de conteúdo e o foco em vestibular mudam por série."
            ),
            "disciplina": (
                "Qual disciplina você quer analisar (por exemplo, Matemática, Português, Física...)? "
                "A disciplina define quais conteúdos são mais críticos para avaliação e vestibular."
            ),
            "perfil_base": (
                "Como você classificaria o perfil geral desse aluno nas matérias, "
                "pensando no histórico recente? (forte, mediano ou em_risco). "
                "Isso ajuda a ajustar as expectativas e recomendações."
            ),
            "nota_prova1": (
                "Qual foi a nota aproximada do aluno na primeira prova dessa disciplina (0 a 10)? "
                "Ela nos dá um primeiro termômetro do desempenho."
            ),
            "nota_prova2": (
                "E na segunda prova, qual foi a nota aproximada (0 a 10)? "
                "Comparar provas ajuda a ver evolução ou queda."
            ),
            "nota_projeto": (
                "Qual a nota aproximada em trabalhos/projetos nessa disciplina (0 a 10)? "
                "Projetos costumam medir aplicação prática e envolvimento."
            ),
            "nota_listas": (
                "E nas listas de exercícios/atividades, que nota você atribuiria (0 a 10)? "
                "Listas mostram consistência no estudo ao longo do tempo."
            ),
            "presenca_percentual": (
                "Qual a presença aproximada do aluno nessa disciplina (%)? "
                "A presença impacta muito a aprendizagem e explica várias dificuldades."
            ),
            "estudos_semanais_horas": (
                "Em média, quantas horas por semana o aluno dedica à disciplina fora da aula? "
                "Isso ajuda a medir o esforço real de estudo."
            ),
            "acessos_plataforma": (
                "Quantos acessos à plataforma/AVA da disciplina o aluno teve nas últimas semanas (aprox.)? "
                "Isso mostra engajamento com materiais digitais."
            ),
            "simulados_feitos": (
                "Quantos simulados relacionados à disciplina (ENEM/vestibular) o aluno fez recentemente? "
                "Simulados são fundamentais para avaliar preparo para provas externas."
            ),
            "dificuldade_disciplina": (
                "Em uma escala de 1 a 5, qual o nível de dificuldade que o aluno sente nessa disciplina "
                "(1=muito fácil, 5=muito difícil)? Isso ajuda a calibrar as recomendações."
            ),
        }

        if campo in perguntas_predefinidas:
            return perguntas_predefinidas[campo]

        # Fallback: gerar pergunta com LLM
        if self.perguntas_feitas > 0:
            prompt_de_pergunta = (
                f"Faça uma pergunta direta sobre: {config['pergunta']}. "
                "Explique brevemente por que é importante. NÃO use 'Olá' ou 'Oi'."
            )
        else:
            prompt_de_pergunta = (
                f"Faça uma pergunta natural sobre: {config['pergunta']}. "
                "Explique brevemente por que essa informação é importante."
            )

        contexto = (
            f"Coletando '{campo}' para análise de desempenho escolar. "
            f"Usuário: {self.usuario_atual['nome'] if self.usuario_atual else 'desconhecido'}. "
            f"Pergunta número: {self.perguntas_feitas + 1}."
        )

        return self.obter_resposta_llm(prompt_de_pergunta, contexto)


    # Coleta dados do aluno + disciplina via conversação
    def coletar_dados_aluno(self):

        print("\n" + "=" * 50)
        print("COLETA DE DADOS DO ALUNO")
        print("=" * 50)

        disciplinas = [
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

        mapa_de_campos = {
            "serie": {"pergunta": "Série do ensino médio (1, 2 ou 3)?", "tipo": "int"},
            "disciplina": {
                "pergunta": "Disciplina do ensino médio",
                "tipo": "escolha",
                "opcoes": disciplinas,
            },
            "perfil_base": {
                "pergunta": "Perfil global de desempenho",
                "tipo": "escolha",
                "opcoes": ["forte", "mediano", "em_risco"],
            },
            "nota_prova1": {
                "pergunta": "Nota da prova 1 (0 a 10)?",
                "tipo": "float",
            },
            "nota_prova2": {
                "pergunta": "Nota da prova 2 (0 a 10)?",
                "tipo": "float",
            },
            "nota_projeto": {
                "pergunta": "Nota de trabalhos/projetos (0 a 10)?",
                "tipo": "float",
            },
            "nota_listas": {
                "pergunta": "Nota em listas/atividades (0 a 10)?",
                "tipo": "float",
            },
            "presenca_percentual": {
                "pergunta": "Presença na disciplina (%)",
                "tipo": "float",
            },
            "estudos_semanais_horas": {
                "pergunta": "Horas de estudo por semana na disciplina",
                "tipo": "float",
            },
            "acessos_plataforma": {
                "pergunta": "Acessos à plataforma da disciplina (últimas semanas)",
                "tipo": "int",
            },
            "simulados_feitos": {
                "pergunta": "Simulados feitos relacionados à disciplina",
                "tipo": "int",
            },
            "dificuldade_disciplina": {
                "pergunta": "Dificuldade sentida (1 a 5)",
                "tipo": "float",
            },
        }

        self.perguntas_feitas = 0
        self.dados_aluno = {}

        for campo, config in mapa_de_campos.items():
            pergunta = self.obter_pergunta(campo, config)
            print(f"\n🤖: {pergunta}")
            self.perguntas_feitas += 1

            while True:
                entrada_usuario = input("👤: ").strip()
                try:
                    if config["tipo"] == "int":
                        valor = int(entrada_usuario)
                        self.dados_aluno[campo] = valor
                        print(f">> Registrado: {valor}")
                        break
                    elif config["tipo"] == "float":
                        limpar = (
                            entrada_usuario.replace("R$", "")
                            .replace(".", "")
                            .replace(",", ".")
                        )
                        valor = float(limpar)
                        self.dados_aluno[campo] = valor
                        print(f">> Registrado: {valor}")
                        break
                    elif config["tipo"] == "escolha":
                        escolha_do_usuario = entrada_usuario.strip().lower()
                        opcoes = [opt.lower() for opt in config["opcoes"]]
                        if escolha_do_usuario in opcoes:
                            original = config["opcoes"][opcoes.index(escolha_do_usuario)]
                            self.dados_aluno[campo] = original
                            print(f">> Selecionado: {original}")
                            break
                        else:
                            print(
                                f"❌ Escolha uma das opções: {', '.join(config['opcoes'])}"
                            )
                            print(
                                "Dica: você pode digitar em minúsculas, "
                                "eu faço o ajuste."
                            )
                    else:
                        self.dados_aluno[campo] = entrada_usuario
                        break
                except ValueError:
                    print("❌ Formato inválido. Tente novamente.")

        # Preenche um aluno_id fake para manter compatibilidade com o modelo, se precisar
        self.dados_aluno.setdefault("aluno_id", 0)

        return True


    # Aqui é somente para mostrar um resumo do projeto
    def exibir_resumo_aluno(self):
        print("\n" + "=" * 50)
        print("RESUMO DO CASO DO ALUNO")
        print("=" * 50)

        resumo_contexto = (
            f"Dados coletados do aluno para análise de disciplina e vestibular: "
            f"{self.dados_aluno}. "
            f"Usuário responsável: {self.usuario_atual['nome'] if self.usuario_atual else 'Desconhecido'}."
        )

        resumo_prompt = (
            "Faça um resumo amigável e conciso do caso do aluno, destacando: "
            "disciplina, série, presença, esforço de estudo e uso de simulados. "
            "Não repita cumprimentos como 'Olá'. Seja objetivo e motivador."
        )

        resumo = self.obter_resposta_llm(resumo_prompt, resumo_contexto)
        print(f"\n >> {resumo}")

        print("\n DADOS TÉCNICOS:")
        print(f"   Série: {self.dados_aluno['serie']}º ano do EM")
        print(f"   Disciplina: {self.dados_aluno['disciplina']}")
        print(f"   Perfil base: {self.dados_aluno['perfil_base']}")
        print(f"   Prova 1: {self.dados_aluno['nota_prova1']}")
        print(f"   Prova 2: {self.dados_aluno['nota_prova2']}")
        print(f"   Projeto: {self.dados_aluno['nota_projeto']}")
        print(f"   Listas: {self.dados_aluno['nota_listas']}")
        print(f"   Presença: {self.dados_aluno['presenca_percentual']}%")
        print(
            f"   Estudo semanal: {self.dados_aluno['estudos_semanais_horas']} h/sem"
        )
        print(f"   Acessos AVA: {self.dados_aluno['acessos_plataforma']}")
        print(f"   Simulados: {self.dados_aluno['simulados_feitos']}")
        print(
            f"   Dificuldade sentida: {self.dados_aluno['dificuldade_disciplina']} (1 a 5)"
        )


    # É onde é acontece a análise completa ML + LLM
    def obter_analise_ia(self):

        print("\nANALISANDO COM IA...")
        print("-" * 30)

        for i in range(3):
            print("Processando" + "." * (i + 1), end="\r")
            time.sleep(0.7)
        print("Análise concluída!    ")

        try:
            resp = requests.post(
                f"{self.api_url}/analisar-com-llm",
                json=self.dados_aluno,
                timeout=60,
            )

            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"❌ Erro na API: {resp.status_code} - {resp.text}")
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Erro: API não está rodando.")
            print("Execute: cd api && python app.py")
            return None
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            return None


    # Exibe o resultado da análise ML + LLM
    def exibir_resultados(self, analise: dict):

        print("\n" + "=" * 60)
        print("RESULTADO DA ANÁLISE INTELIGENTE (EduScore IA)")
        print("=" * 60)

        ml_pred = analise.get("ml_predicao", {})
        analise_llm = analise.get("analise_llm", {})
        nota = ml_pred.get("nota_prevista", 0.0)
        prob_aprov = ml_pred.get("prob_aprovacao", 0.0)
        risco = ml_pred.get("risco", "desconhecido")
        vestibular_score = ml_pred.get("vestibular_score", 0.0)
        nivel_prep = ml_pred.get("nivel_preparacao", "desconhecido")

        emoji_status = "✅" if risco == "baixo" else "⚠️" if risco == "moderado" else "❌"

        print(f"\n{emoji_status} RISCO NA DISCIPLINA: {risco.upper()}")
        print(f"Nota prevista: {nota:.2f} / 10")
        print(f"Probabilidade de aprovação: {prob_aprov:.1%}")
        print(f"Score de preparação para vestibular: {vestibular_score:.1f} / 100")
        print(f"Nível de preparação: {nivel_prep}")

        barra = 30
        preenchimento = int(prob_aprov * barra)
        bar = "█" * preenchimento + "░" * (barra - preenchimento)
        print(f"\nProbabilidade visual: [{bar}] {prob_aprov:.1%}")

        print("\nEXPLICAÇÃO DO CASO:")
        print("-" * 40)
        print(analise_llm.get("explicacao", "(sem explicação do LLM)"))

        print("\nRECOMENDAÇÕES PARA O PROFESSOR:")
        print("-" * 40)
        print(analise_llm.get("recomendacoes_professor", "(sem recomendações)"))

        print("\nRECOMENDAÇÕES PARA O ALUNO (VESTIBULAR):")
        print("-" * 40)
        print(analise_llm.get("recomendacoes_aluno", "(sem recomendações)"))

        print("\n" + analise.get("insights_combinados", ""))

    
    # Fluxo principal, executa o chatbot conversacional de ponta a ponta
    def run(self):

        try:
            # Verifica se API está de pé
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code != 200:
                    print("❌ API não está saudável!")
                    print(response.text)
                    return
            except Exception:
                print("❌ API não está rodando!")
                print("Execute: cd api && python app.py")
                return

            self.msg_boas_vindas()

            if not self.autenticar_usuario():
                return

            while True:
                print("\n" + "=" * 60)
                print("NOVA ANÁLISE DE ALUNO/DISCIPLINA")
                print("=" * 60)

                # reset do contador
                self.perguntas_feitas = 0

                if not self.coletar_dados_aluno():
                    break

                self.exibir_resumo_aluno()

                confirmar = (
                    input("\nDeseja analisar esse caso com IA? (s/n): ")
                    .strip()
                    .lower()
                )
                if confirmar == "s":
                    analises = self.obter_analise_ia()
                    if analises:
                        self.exibir_resultados(analises)
                    else:
                        print("❌ Não foi possível realizar a análise.")

                nova_analise = (
                    input("\nDeseja analisar outro aluno/disciplina? (s/n): ")
                    .strip()
                    .lower()
                )
                if nova_analise != "s":
                    break

                self.dados_aluno = {}

            # Hora da despedida
            msg_despedida = (
                "O usuário está encerrando a sessão no EduScore. "
                "Faça uma despedida amigável, reconhecendo o esforço em usar dados para "
                "ajudar alunos a melhorarem seu desempenho."
            )
            despedida = self.obter_resposta_llm(msg_despedida)
            print(f"\n >> {despedida}")

        except KeyboardInterrupt:
            print("\n\nSessão encerrada pelo usuário.")
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")


if __name__ == "__main__":
    chatbot = ChatbotEstudanteLLM()
    chatbot.run()
