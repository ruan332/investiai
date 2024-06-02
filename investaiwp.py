import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import Tool


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("A API key do Google não foi encontrada. Certifique-se de que o arquivo .env está configurado corretamente e contém 'GOOGLE_API_KEY'.")

if GROQ_API_KEY is None:
    raise ValueError("A API key do Groq não foi encontrada. Certifique-se de que o arquivo .env está configurado corretamente e contém 'GROQ_API_KEY'.")

# Cria uma instância do modelo Gemini
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-pro", #testar o 1.5 flash
    verbose=True,
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)

# Cria uma instância do modelo Llama 3
llm_llama3 = ChatGroq(
    api_key=GROQ_API_KEY,
    model='llama3-70b-8192'
)

# Cria uma instância da ferramenta DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Agente Analista Técnico
technical_analyst = Agent(
    role='Analista Técnico',
    goal='Analisar gráficos e indicadores técnicos para identificar sinais de compra de ações',
    verbose=True,
    memory=True,
    backstory=(
        "Você é um especialista em análise técnica, com anos de experiência em identificar padrões gráficos e conversa em Português pt-br"
        "e usar indicadores para prever movimentos de preços."
    ),
    llm=llm_gemini,
    tools=[search_tool],
    allow_delegation=True
)

# Agente Analista Fundamentalista
fundamental_analyst = Agent(
    role='Analista Fundamentalista',
    goal='Analisar notícias e tendências de mercado para identificar ações promissoras',
    verbose=True,
    memory=True,
    backstory=(
        "Você é um analista fundamentalista com uma profunda compreensão de notícias de mercado e conversa em Português pt-br"
        "tendências econômicas que impactam o valor das ações."
    ),
    llm=llm_gemini,
    tools=[search_tool],  # Adiciona a ferramenta de busca ao agente
    allow_delegation=True
)

# Agente Moderador
moderator = Agent(
    role='Moderador',
    goal='Facilitar a discussão entre os analistas para chegar a um consenso sobre as melhores ações',
    verbose=True,
    memory=True,
    backstory=(
        "Você é um moderador experiente, especializado em facilitar discussões e ajudar grupos a chegarem a um consenso e conversa em Português pt-br."
    ),
    llm=llm_gemini,
    tools=[],  # O moderador não precisa de ferramentas específicas
    allow_delegation=False
)

# Agente Analista de Investimentos
investment_analyst = Agent(
    role='Analista de Investimentos',
    goal='Avaliar recomendações dos analistas técnico e fundamentalista e criar um relatório detalhado',
    verbose=True,
    memory=True,
    backstory=(
        "Você é um analista de investimentos experiente, especializado em sintetizar análises técnicas e fundamentalistas para fazer recomendações de investimento e conversa em Português pt-br."
    ),
    llm=llm_gemini,
    tools=[],  # O analista de investimentos não precisa de ferramentas específicas
    allow_delegation=False
)

# Definir as tarefas para cada agente
technical_task = Task(
    description=(
        "Pesquisas ações brasileiras listadas na B3 com maior liquidês e Analisar os gráficos e preços para identificar até 5 papéis com os melhores sinais de compra."
        "Utilize indicadores técnicos e padrões gráficos para fazer suas recomendações. Liste os nomes das ações recomendadas e explique brevemente o motivo da recomendação para cada uma."
    ),
    expected_output='Uma lista de até 5 papéis recomendados com base em análise técnica, incluindo os nomes das ações e uma breve explicação de cada recomendação.',
    llm=llm_gemini,
    agent=technical_analyst,
)

fundamental_task = Task(
    description=(
        "Avaliar notícias e tendências de mercado do site br.investing.com para identificar até 5 papéis promissores."
        "Considere fatores econômicos, notícias corporativas e tendências de mercado. Utilize o site br.investing.com para encontrar as notícias mais recentes sobre o cenário econômico. Liste os nomes das ações recomendadas e explique brevemente o motivo da recomendação para cada uma."
    ),
    expected_output='Uma lista de até 5 papéis recomendados com base em análise fundamentalista, incluindo os nomes das ações e uma breve explicação de cada recomendação.',
    llm=llm_gemini,
    agent=fundamental_analyst,
)

discussion_task = Task(
    description=(
        "Facilitar uma discussão entre os analistas técnico e fundamentalista para chegarem a um consenso sobre as melhores ações."
        "A discussão deve considerar as recomendações de ambos os analistas e chegar a uma lista final de 5 ações com suas respectivas siglas na B3 e uma breve explicação para cada escolha."
    ),
    expected_output='Uma lista final de 5 ações recomendadas com base na análise técnica e fundamentalista, incluindo o preço de entrada e saída da operação, as siglas da B3 e uma breve explicação de cada recomendação.',
    llm=llm_gemini,
    agent=moderator,
)

investment_task = Task(
    description=(
        "Receber as recomendações dos analistas técnico e fundamentalista, avaliá-las e criar um relatório detalhado."
        "O relatório deve incluir as 5 ações recomendadas, os motivos para cada recomendação fornecidos pelos analistas técnico e fundamentalista, e uma avaliação final de cada recomendação."
    ),
    expected_output='Um relatório detalhado com as 5 ações recomendadas, incluindo os motivos fornecidos pelos analistas técnico e fundamentalista, e uma avaliação final de cada recomendação.',
    llm=llm_gemini,
    agent=investment_analyst,
)

# Formar a equipe de analistas
crew = Crew(
    agents=[technical_analyst, fundamental_analyst, moderator, investment_analyst],
    tasks=[technical_task, fundamental_task, discussion_task, investment_task],
    process=Process.sequential  # Execução sequencial das tarefas
)

# Adicionar uma verificação antes de iniciar o processo de análise
if not crew.agents or not crew.tasks:
    raise ValueError("A equipe ou as tarefas não estão definidas corretamente.")

# Iniciar o processo de análise
result = crew.kickoff(inputs={})
print(result)

# Função para enviar mensagem via API WhatsApp (adaptada para enviar o relatório)
def send_whatsapp_message(message):
    url = "API_WHATSAPP"
    headers = {
        "access-token": "",
        "Content-Type": ""
    }
    payload = {
        "number": "",
        "message": message
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("Mensagem enviada com sucesso!")
    else:
        print("Falha ao enviar mensagem. Código de status:", response.status_code)

# Enviar o relatório final via WhatsApp
send_whatsapp_message(result)
