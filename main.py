# Versão modificada do tkinder_gpt_chuveiro_agents_v2.py com simulação duplicada (Chuveiro Real + Digital Twin)

#%%

import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import time
import modelos_llm

# Modelos disponíveis
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

root = tk.Tk()
root.title("Chuveiro Interativo com IA + Digital Twin")
root.geometry("1400x700")

frame_principal = tk.Frame(root)
frame_principal.pack(fill=tk.BOTH, expand=True)

frame_esquerda = tk.Frame(frame_principal)
frame_esquerda.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frame_direita = tk.Frame(frame_principal, width=400)
frame_direita.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

fig, axs = plt.subplots(2, 2, figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_esquerda)
canvas.get_tk_widget().pack()

frame_controles = tk.Frame(frame_esquerda)
frame_controles.pack(pady=5)

frame_sliders = tk.Frame(frame_controles)
frame_sliders.pack()

# Sliders
slider_Xfrio = tk.Scale(frame_sliders, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, label='Abertura Frio', length=200)
slider_Xfrio.set(0.5)
slider_Xfrio.grid(row=0, column=0)

slider_Xquente = tk.Scale(frame_sliders, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, label='Abertura Quente', length=200)
slider_Xquente.set(0.5)
slider_Xquente.grid(row=0, column=1)

slider_Tfrio = tk.Scale(frame_sliders, from_=10, to=30, resolution=0.5, orient=tk.HORIZONTAL, label='Temp. Frio', length=200)
slider_Tfrio.set(20)
slider_Tfrio.grid(row=1, column=0)

slider_Tquente = tk.Scale(frame_sliders, from_=30, to=80, resolution=0.5, orient=tk.HORIZONTAL, label='Temp. Quente', length=200)
slider_Tquente.set(60)
slider_Tquente.grid(row=1, column=1)

frame_botoes = tk.Frame(frame_controles)
frame_botoes.pack(pady=10)

# Estados
real_tempos, real_vazoes, real_temperaturas = [], [], []
twin_tempos, twin_vazoes, twin_temperaturas = [], [], []
rodando = False

def atualizar_grafico():
    global rodando
    rodando = True
    tempo = 0
    while rodando:
        Xs = [slider_Xfrio.get(), slider_Xquente.get()]
        Ts = [slider_Tfrio.get(), slider_Tquente.get()]

        F_real, T_real = modchuv(Xs, Ts)
        F_twin, T_twin = modchuv(Xs, Ts)  # Digital Twin replica condições

        real_temperaturas.append(T_real)
        real_vazoes.append(F_real)
        twin_temperaturas.append(T_twin)
        twin_vazoes.append(F_twin)
        real_tempos.append(tempo)
        twin_tempos.append(tempo)

        axs[0][0].clear()
        axs[1][0].clear()
        axs[0][1].clear()
        axs[1][1].clear()

        axs[0][0].plot(real_tempos, real_temperaturas, color='red')
        axs[1][0].plot(real_tempos, real_vazoes, color='blue')
        axs[0][1].plot(twin_tempos, twin_temperaturas, color='orange')
        axs[1][1].plot(twin_tempos, twin_vazoes, color='green')

        axs[0][0].set_title("Chuveiro Real - Temperatura")
        axs[1][0].set_title("Chuveiro Real - Vazão")
        axs[0][1].set_title("Digital Twin - Temperatura")
        axs[1][1].set_title("Digital Twin - Vazão")

        canvas.draw()
        tempo += 1
        time.sleep(1)

# Botões
btn_iniciar = tk.Button(frame_botoes, text="Iniciar Simulação", command=lambda: threading.Thread(target=atualizar_grafico, daemon=True).start())
btn_iniciar.grid(row=0, column=0, padx=5)

tk.Button(frame_botoes, text="Parar Simulação", command=lambda: globals().update(rodando=False)).grid(row=0, column=1, padx=5)

# Seção direita: IA
frame_pergunta = tk.Frame(frame_direita)
frame_pergunta.pack(padx=10, pady=5, anchor="w")

pergunta_entry = tk.Entry(frame_pergunta, width=40)
pergunta_entry.pack(side=tk.LEFT, padx=5)

tk.Button(frame_pergunta, text="Perguntar para IA", command=lambda: enviar_pergunta()).pack(side=tk.LEFT, padx=5)

resposta_label = tk.Label(frame_direita, text="Resposta da IA aparecerá aqui.", wraplength=380, justify="left", anchor="nw")
resposta_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

tk.Button(frame_controles, text="Trocar Modelo", command=lambda: trocar_modelo()).pack()
modelo_nome_label = tk.Label(frame_controles, text=f"Modelo atual: {modelos_disponiveis[modelo_atual_index[0]][0]}")
modelo_nome_label.pack(pady=5)

def enviar_pergunta():
    if not real_vazoes or not real_temperaturas or not twin_vazoes or not twin_temperaturas:
        resposta_label.config(text="Inicie a simulação primeiro.")
        return

    dados = {
        'Xfrio': slider_Xfrio.get(),
        'Xquente': slider_Xquente.get(),
        'Tfrio': slider_Tfrio.get(),
        'Tquente': slider_Tquente.get(),
        'Fsaida_real': real_vazoes[-1],
        'Tsaida_real': real_temperaturas[-1],
        'Fsaida_twin': twin_vazoes[-1],
        'Tsaida_twin': twin_temperaturas[-1]
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

        try:
            resposta = modelo_fn(pergunta, dados)
        except Exception as e:
            resposta = f"Ocorreu um erro: {e}"

        def atualizar_resposta(r=resposta):
            resposta_label.config(text=r)

        root.after(0, atualizar_resposta)

threading.Thread(target=loop_ia, daemon=True).start()
root.mainloop()
# %%
