# modelos_llm.py

import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

import openai
openai.api_key = openai_key

import sys
sys.path.append("C:/Users/Jesus Yepez Rojas/Documents/jesus/DIGITAL_TWINS/DIGITAL_TWINS_agents")

import os 
os.chdir("C:/Users/Jesus Yepez Rojas/Documents/jesus/DIGITAL_TWINS/DIGITAL_TWINS_agents")

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
Você é um assistente técnico que compara o estado de um chuveiro real com seu digital twin.

Dados do Chuveiro Real:
- Registro Frio: {dados['Xfrio']:.2f}
- Registro Quente: {dados['Xquente']:.2f}
- Temp Fria: {dados['Tfrio']} °C
- Temp Quente: {dados['Tquente']} °C
- Vazão: {dados['Fsaida_real']:.2f} L/min
- Temperatura saída: {dados['Tsaida_real']:.2f} °C

Dados do Digital Twin:
- Vazão: {dados['Fsaida_twin']:.2f} L/min
- Temperatura saída: {dados['Tsaida_twin']:.2f} °C

Pergunta do usuário:
{pergunta}

Responda comparando os dois estados, em até 3 frases práticas.
"""
    return _call_openai(prompt, "gpt-3.5-turbo")

def gpt_4(pergunta, dados):
    prompt = f"""
Você é um especialista em controle térmico de sistemas hidráulicos. Analise o comportamento comparado de um chuveiro físico e seu gêmeo digital.

Chuveiro Real:
- Abertura Frio: {dados['Xfrio']:.2f}
- Abertura Quente: {dados['Xquente']:.2f}
- Temp Entrada Fria: {dados['Tfrio']} °C
- Temp Entrada Quente: {dados['Tquente']} °C
- Vazão: {dados['Fsaida_real']:.2f} L/min
- Temp Saída: {dados['Tsaida_real']:.2f} °C

Digital Twin:
- Vazão: {dados['Fsaida_twin']:.2f} L/min
- Temp Saída: {dados['Tsaida_twin']:.2f} °C

Pergunta técnica:
{pergunta}

Ofereça uma explicação técnica comparando os comportamentos.
"""
    return _call_openai(prompt, "gpt-4")

def agente_v2(pergunta, dados):
    prompt = f"""
Dados do Chuveiro Real:
- Registro Frio: {dados['Xfrio']:.2f}
- Registro Quente: {dados['Xquente']:.2f}
- Temp Fria: {dados['Tfrio']} °C
- Temp Quente: {dados['Tquente']} °C
- Vazão: {dados['Fsaida_real']:.2f} L/min
- Temperatura saída: {dados['Tsaida_real']:.2f} °C

Digital Twin:
- Vazão: {dados['Fsaida_twin']:.2f} L/min
- Temperatura saída: {dados['Tsaida_twin']:.2f} °C

Pergunta do usuário:
{pergunta}

Responda de forma prática, destacando se há divergência.
"""
    return chatbot_chuveiro_v2(prompt)
