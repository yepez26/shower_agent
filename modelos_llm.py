# modelos_llm.py


#%%

from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

import openai
openai.api_key = openai_key

#%%
import sys
sys.path.append("C:/Users/Jesus Yepez Rojas/Documents/jesus/DIGITAL_TWINS/DIGITAL_TWINS_agents")

import os 
os.chdir("C:/Users/Jesus Yepez Rojas/Documents/jesus/DIGITAL_TWINS/DIGITAL_TWINS_agents")

from pipeline import chatbot_chuveiro_v2


#%%

from pipeline import chatbot_chuveiro_v2

def _call_openai(prompt, model):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro com modelo {model}: {str(e)}"


def gpt_turbo(pergunta, dados):
    prompt = f"""
Você é um assistente técnico que responde de forma breve e direta sobre configurações de chuveiro.

Dados do sistema:
- Registro Frio: {dados['Xfrio']:.2f}
- Registro Quente: {dados['Xquente']:.2f}
- Temp Fria: {dados['Tfrio']} °C
- Temp Quente: {dados['Tquente']} °C
- Vazão: {dados['Fsaida']:.2f} L/min
- Temperatura saída: {dados['Tsaida']:.2f} °C

Pergunta do usuário:
{pergunta}

Responda em até 3 frases, de forma prática.
"""
    return _call_openai(prompt, "gpt-3.5-turbo")


def gpt_4(pergunta, dados):
    prompt = f"""
Você é um especialista em controle térmico de sistemas hidráulicos.

Analise os dados a seguir:
- Abertura Frio: {dados['Xfrio']:.2f}
- Abertura Quente: {dados['Xquente']:.2f}
- Temperatura Entrada Fria: {dados['Tfrio']} °C
- Temperatura Entrada Quente: {dados['Tquente']} °C
- Vazão final: {dados['Fsaida']:.2f} L/min
- Temperatura final da água: {dados['Tsaida']:.2f} °C

Pergunta técnica:
{pergunta}

Forneça uma resposta analítica, com explicação causal.
"""
    return _call_openai(prompt, "gpt-4")


def agente_v2(pergunta, dados):
    prompt = f"""
    Dados do sistema:
    - Registro Frio: {dados['Xfrio']:.2f}
    - Registro Quente: {dados['Xquente']:.2f}
    - Temp Fria: {dados['Tfrio']} °C
    - Temp Quente: {dados['Tquente']} °C
    - Vazão: {dados['Fsaida']:.2f} L/min
    - Temperatura saída: {dados['Tsaida']:.2f} °C

    Pergunta do usuário:
    {pergunta}

    Responda de forma prática."""
    
    return chatbot_chuveiro_v2(prompt)
