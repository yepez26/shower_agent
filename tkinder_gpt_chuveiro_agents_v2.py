#%%
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import time
import os
os.chdir(r'C:\Users\Jesus Yepez Rojas\Documents\jesus\DIGITAL_TWINS\Tkinter_gpt\tkinder_gpt_chuveiro_agents_v2')

import modelos_llm  # Módulo externo com os LLMs

# Modelos disponíveis (nome, função)
modelos_disponiveis = [
    ("GPT 3.5 Turbo", modelos_llm.gpt_turbo),
    ("GPT-4", modelos_llm.gpt_4),
    ("Agente Local v2", modelos_llm.agente_v2),
]
modelo_atual_index = [0]
modelo_nome_label = None

#%%
fila_perguntas = queue.Queue()

def modchuv(Xs=[0.5, 0.5], Ps=[2, 2], Ts=[20, 60]):
    Xfrio, Xquente = Xs
    Tfrio, Tquente = Ts
    Pfrio, Pquente = Ps
    Patm = 1
    K = 10
    Ffrio = Xfrio * K * np.sqrt(Pfrio - Patm)
    Fquente = Xquente * K * np.sqrt(Pquente - Patm)
    Fsaida = Ffrio + Fquente
    Tsaida = (Ffrio * Tfrio + Fquente * Tquente) / Fsaida
    Fsaida = max(0, Fsaida + np.random.normal(0, 0.2))
    Tsaida += np.random.normal(0, 0.5)
    return Fsaida, Tsaida

# Inicialização do Tkinter
root = tk.Tk()
root.title("Chuveiro Interativo com IA")
root.geometry("1200x600")

# Layout com frame principal dividido em 2 colunas
frame_principal = tk.Frame(root)
frame_principal.pack(fill=tk.BOTH, expand=True)

# Frame da esquerda: gráficos e controles
frame_esquerda = tk.Frame(frame_principal)
frame_esquerda.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Frame da direita: perguntas e respostas da IA
frame_direita = tk.Frame(frame_principal, width=400)
frame_direita.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

# Gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_esquerda)
canvas.get_tk_widget().pack()

def criar_slider(frame, label, from_, to, resolution, row, col, default):
    slider = tk.Scale(frame, from_=from_, to=to, resolution=resolution,
                      orient=tk.HORIZONTAL, label=label, length=250)
    slider.set(default)
    slider.grid(row=row, column=col, padx=5, pady=5)
    return slider

# Controles
frame_controles = tk.Frame(frame_esquerda)
frame_controles.pack(pady=5)

frame_sliders = tk.Frame(frame_controles)
frame_sliders.pack()

slider_Xfrio = criar_slider(frame_sliders, 'Abertura Registro Frio', 0.0, 1.0, 0.05, 0, 0, 0.5)
slider_Xquente = criar_slider(frame_sliders, 'Abertura Registro Quente', 0.0, 1.0, 0.05, 0, 1, 0.5)
slider_Tfrio = criar_slider(frame_sliders, 'Temp. Linha Fria (°C)', 10, 30, 0.5, 1, 0, 20)
slider_Tquente = criar_slider(frame_sliders, 'Temp. Linha Quente (°C)', 30, 80, 0.5, 1, 1, 60)

frame_botoes = tk.Frame(frame_controles)
frame_botoes.pack(pady=10)

vazoes, temperaturas, tempos = [], [], []
rodando = False

def atualizar_grafico():
    global rodando
    rodando = True
    tempo = 0
    while rodando:
        Fsaida, Tsaida = modchuv([slider_Xfrio.get(), slider_Xquente.get()],
                                 Ts=[slider_Tfrio.get(), slider_Tquente.get()])
        temperaturas.append(Tsaida)
        vazoes.append(Fsaida)
        tempos.append(tempo)

        ax1.clear()
        ax2.clear()
        ax1.plot(tempos, temperaturas, color='red', label='Temperatura (°C)')
        ax2.plot(tempos, vazoes, color='blue', label='Vazão (L/min)')
        ax1.legend()
        ax2.legend()
        ax1.set_ylabel("Temperatura (°C)")
        ax2.set_ylabel("Vazão (L/min)")
        ax2.set_xlabel("Tempo (s)")
        canvas.draw()

        tempo += 1
        time.sleep(1)

tk.Button(frame_botoes, text="Iniciar Simulação",
          command=lambda: threading.Thread(target=atualizar_grafico, daemon=True).start()).grid(row=0, column=0, padx=5)

tk.Button(frame_botoes, text="Parar Simulação",
          command=lambda: globals().update(rodando=False)).grid(row=0, column=1, padx=5)

# Pergunta (lado direito)
frame_pergunta = tk.Frame(frame_direita)
frame_pergunta.pack(padx=10, pady=5, anchor="w")

pergunta_entry = tk.Entry(frame_pergunta, width=40)
pergunta_entry.pack(side=tk.LEFT, padx=5)

tk.Button(frame_pergunta, text="Perguntar para IA", command=lambda: enviar_pergunta()).pack(side=tk.LEFT, padx=5)

# Resposta (lado direito, grande)
resposta_label = tk.Label(frame_direita, text="Resposta da IA aparecerá aqui.", 
                          wraplength=380, justify="left", anchor="nw")
resposta_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Botão de troca de modelo
tk.Button(frame_controles, text="Trocar Modelo", command=lambda: trocar_modelo()).pack()
modelo_nome_label = tk.Label(frame_controles, text=f"Modelo atual: {modelos_disponiveis[modelo_atual_index[0]][0]}")
modelo_nome_label.pack(pady=5)

# Funções
def enviar_pergunta():
    if not vazoes or not temperaturas:
        resposta_label.config(text="Inicie a simulação primeiro.")
        return

    dados = {
        'Xfrio': slider_Xfrio.get(),
        'Xquente': slider_Xquente.get(),
        'Tfrio': slider_Tfrio.get(),
        'Tquente': slider_Tquente.get(),
        'Fsaida': vazoes[-1],
        'Tsaida': temperaturas[-1]
    }

    resposta_label.config(text="Consultando IA...")
    fila_perguntas.put((pergunta_entry.get(), dados))

def trocar_modelo():
    modelo_atual_index[0] = (modelo_atual_index[0] + 1) % len(modelos_disponiveis)
    modelo_nome_label.config(text=f"Modelo atual: {modelos_disponiveis[modelo_atual_index[0]][0]}")

def loop_ia():
    while True:
        pergunta, dados = fila_perguntas.get()
        modelo_fn = modelos_disponiveis[modelo_atual_index[0]][1]
        resposta = modelo_fn(pergunta, dados)
        root.after(0, lambda r=resposta: resposta_label.config(text=r))

threading.Thread(target=loop_ia, daemon=True).start()
root.mainloop()
# %%
