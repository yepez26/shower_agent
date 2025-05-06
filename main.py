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

#%%

plt.rcParams.update({'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})

modelos_disponiveis = [
    ("GPT 3.5 Turbo", modelos_llm.gpt_turbo),
    ("GPT-4", modelos_llm.gpt_4),
    ("Agente Local v2", modelos_llm.agente_v2),
]
modelo_atual_index = [0]
modelo_nome_label = None
fila_perguntas = queue.Queue()

real_tempos, real_vazoes, real_temperaturas, real_iqb = [], [], [], []
twin_tempos, twin_vazoes, twin_temperaturas, twin_iqb = [], [], [], []
rodando = False


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


def aplicar_falhas(Xs, Ps, Ts):
    if falhas["valvula_fria_emperrada"]:
        Xs[0] = 0
    if falhas["falha_aquecedor"]:
        Ts[1] = Ts[0]
    if falhas["agua_fria_mais_fria"]:
        Ts[0] = min(Ts[0], 4.9)
    if falhas["pouco_gas"]:
        Ts[1] = min(Ts[1], 29.9)
    if falhas["falta_agua_fria"]:
        Ps[0] = 1
    if falhas["pouca_agua_fria"]:
        Ps[0] = min(Ps[0], 1.49)
    return Xs, Ps, Ts

def calcular_IQB(T_atual, F_atual, T_des, F_des):
    IQB = 100 - 0.1 * (T_atual - T_des)**2 - 0.1 * (F_atual - F_des)**2
    return max(0, min(100, IQB))


root = tk.Tk()
root.title("Chuveiro Interativo com IA + Digital Twin")
root.geometry("1600x900")

frame_principal = tk.Frame(root)
frame_principal.pack(fill=tk.BOTH, expand=True)

frame_topo = tk.Frame(frame_principal)
frame_topo.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

frame_graficos = tk.Frame(frame_topo)
frame_graficos.pack(side=tk.LEFT, padx=10, pady=10)

frame_col_real = tk.LabelFrame(frame_graficos, text="Chuveiro Real")
frame_col_real.grid(row=0, column=0, padx=5)

frame_col_twin = tk.LabelFrame(frame_graficos, text="Chuveiro Twin")
frame_col_twin.grid(row=0, column=1, padx=5)

fig, axs = plt.subplots(3, 2, figsize=(9, 5))
plt.subplots_adjust(hspace=0.6, wspace=0.4)

canvas = FigureCanvasTkAgg(fig, master=frame_graficos)
canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)

frame_direita = tk.Frame(frame_topo, width=380)
frame_direita.pack(side=tk.RIGHT, fill=tk.Y)

frame_inputs = tk.LabelFrame(frame_principal, text="Parâmetros de Entrada")
frame_inputs.pack(padx=10, pady=5)

for i, (label, from_, to, res, val) in enumerate([
    ('Abertura Frio', 0.0, 1.0, 0.05, 0.5),
    ('Abertura Quente', 0.0, 1.0, 0.05, 0.5),
    ('Temp. Frio', 10, 30, 0.5, 20),
    ('Temp. Quente', 30, 80, 0.5, 60),
    ('Temp. Desejada', 30, 50, 0.5, 37.5),
    ('Vazão Desejada', 1, 50, 1, 5),
]):
    slider = tk.Scale(frame_inputs, label=label, from_=from_, to=to, resolution=res, orient=tk.HORIZONTAL, length=160)
    slider.set(val)
    slider.grid(row=i//3, column=i%3, padx=5, pady=5)
    if 'Abertura Frio' in label:
        slider_xfrio = slider
    elif 'Abertura Quente' in label:
        slider_xquente = slider
    elif 'Temp. Frio' in label:
        slider_tfrio = slider
    elif 'Temp. Quente' in label:
        slider_tquente = slider
    elif 'Temp. Desejada' in label:
        slider_t_des = slider
    elif 'Vazão Desejada' in label:
        slider_f_des = slider

frame_falhas = tk.LabelFrame(frame_principal, text="Falhas Simuladas")
frame_falhas.pack(padx=10, pady=5)

botoes_falha = [
    ("valvula_fria_emperrada", "Válvula Fria Emperrada"),
    ("falha_aquecedor", "Falha no Aquecedor"),
    ("agua_fria_mais_fria", "Água Fria Muito Fria"),
    ("pouco_gas", "Pouco Gás"),
    ("falta_agua_fria", "Falta Água Fria"),
    ("pouca_agua_fria", "Pouca Água Fria"),
]

falhas = {nome: False for nome, _ in botoes_falha}

def toggle_falha(falha):
    falhas[falha] = not falhas[falha]

for i, (nome, label) in enumerate(botoes_falha):
    tk.Button(frame_falhas, text=label, command=lambda n=nome: toggle_falha(n), width=20).grid(row=i//3, column=i%3, padx=5, pady=3)

frame_controles = tk.LabelFrame(frame_principal, text="Controles da Simulação")
frame_controles.pack(padx=10, pady=10)

tk.Button(frame_controles, text="Resetar Gráficos", command=lambda: resetar_graficos()).grid(row=0, column=2, padx=10)

tk.Button(frame_controles, text="Iniciar Simulação", command=lambda: threading.Thread(target=atualizar_grafico, daemon=True).start()).grid(row=0, column=0, padx=10)

tk.Button(frame_controles, text="Parar Simulação", command=lambda: globals().update(rodando=False)).grid(row=0, column=1, padx=10)

frame_modelo = tk.LabelFrame(frame_direita, text="Modelo de Linguagem")
frame_modelo.pack(padx=10, pady=10, fill="x")

btn_trocar_modelo = tk.Button(frame_modelo, text="Trocar Modelo", command=lambda: trocar_modelo())
btn_trocar_modelo.pack(side=tk.LEFT)

modelo_nome_label = tk.Label(frame_modelo, text=f"Modelo atual: {modelos_disponiveis[modelo_atual_index[0]][0]}")
modelo_nome_label.pack(side=tk.LEFT, padx=10)

frame_pergunta = tk.LabelFrame(frame_direita, text="Pergunta para a IA")
frame_pergunta.pack(padx=10, pady=10, fill="x")

pergunta_entry = tk.Entry(frame_pergunta, width=30)
pergunta_entry.pack(side=tk.LEFT, padx=5)

btn_perguntar = tk.Button(frame_pergunta, text="Perguntar", command=lambda: enviar_pergunta())
btn_perguntar.pack(side=tk.LEFT)

resposta_label = tk.Label(frame_direita, text="Resposta da IA aparecerá aqui.", wraplength=360, justify="left", anchor="nw")
resposta_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

def atualizar_grafico():
    global rodando
    rodando = True
    tempo = 0
    while rodando:
        Xs = [slider_xfrio.get(), slider_xquente.get()]
        Ts = [slider_tfrio.get(), slider_tquente.get()]
        T_des = slider_t_des.get()
        F_des = slider_f_des.get()
        Xs_real, Ps_real, Ts_real = aplicar_falhas(Xs.copy(), [2, 2], Ts.copy())
        F_real, T_real = modchuv(Xs_real, Ps_real, Ts_real)
        F_twin, T_twin = modchuv(Xs, [2, 2], Ts)
        real_temperaturas.append(T_real)
        real_vazoes.append(F_real)
        twin_temperaturas.append(T_twin)
        twin_vazoes.append(F_twin)
        real_iqb.append(calcular_IQB(T_real, F_real, T_des, F_des))
        twin_iqb.append(calcular_IQB(T_twin, F_twin, T_des, F_des))
        real_tempos.append(tempo)
        twin_tempos.append(tempo)
        axs[0][0].clear()
        axs[0][1].clear()
        axs[1][0].clear()
        axs[1][1].clear()
        axs[2][0].clear()
        axs[2][1].clear()
        axs[0][0].plot(real_tempos, real_temperaturas, 'red')
        axs[0][0].set_title("Temperatura Real", pad=10)
        axs[0][1].plot(twin_tempos, twin_temperaturas, 'orange')
        axs[0][1].set_title("Temperatura Twin", pad=10)
        axs[1][0].plot(real_tempos, real_vazoes, 'blue')
        axs[1][0].set_title("Vazão Real", pad=10)
        axs[1][1].plot(twin_tempos, twin_vazoes, 'green')
        axs[1][1].set_title("Vazão Twin", pad=10)
        axs[2][0].plot(real_tempos, real_iqb, 'purple')
        axs[2][0].set_title("IQB Real", pad=10)
        axs[2][1].plot(twin_tempos, twin_iqb, 'cyan')
        axs[2][1].set_title("IQB Twin", pad=10)
        canvas.draw()
        tempo += 1
        time.sleep(1)

def enviar_pergunta():
    if not real_vazoes or not real_temperaturas or not twin_vazoes or not twin_temperaturas:
        resposta_label.config(text="Inicie a simulação primeiro.")
        return
    dados = {
        'Xfrio': slider_xfrio.get(),
        'Xquente': slider_xquente.get(),
        'Tfrio': slider_tfrio.get(),
        'Tquente': slider_tquente.get(),
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

def resetar_graficos():
    global real_tempos, real_vazoes, real_temperaturas, real_iqb
    global twin_tempos, twin_vazoes, twin_temperaturas, twin_iqb
    real_tempos.clear()
    real_vazoes.clear()
    real_temperaturas.clear()
    real_iqb.clear()
    twin_tempos.clear()
    twin_vazoes.clear()
    twin_temperaturas.clear()
    twin_iqb.clear()
    for row in axs:
        for ax in row:
            ax.clear()
    canvas.draw()


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
