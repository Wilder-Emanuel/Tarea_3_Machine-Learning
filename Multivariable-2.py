"""
Regresion Lineal Multivariable desde cero: Gradient Descent
Dataset: horas_estudio | horas_suenio | asistencia_pct | ejercicios_resueltos  →  nota_final (Y)
Curso  : CS3061 - Machine Learning - UTEC
Autor  : D.Sc. Manuel Eduardo Loaiza Fernandez
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================
# EXTRAER DATOS
# ===========================================================================

ruta = "/home/bm-003/MachineLearning/caso2_notas.csv"
datos = []

with open(ruta, "r", encoding="utf-8") as f:
    encabezado = f.readline().strip().split(",")

    idx_x1 = encabezado.index("horas_estudio")
    idx_x2 = encabezado.index("horas_suenio")
    idx_x3 = encabezado.index("asistencia_pct")
    idx_x4 = encabezado.index("ejercicios_resueltos")
    idx_y  = encabezado.index("nota_final ( Y )")

    for linea in f:
        partes = linea.strip().split(",")
        x1 = float(partes[idx_x1])   # horas_estudio
        x2 = float(partes[idx_x2])   # horas_suenio
        x3 = float(partes[idx_x3])   # asistencia_pct
        x4 = float(partes[idx_x4])   # ejercicios_resueltos
        y  = float(partes[idx_y])    # nota_final (target)
        datos.append((x1, x2, x3, x4, y))

# ===========================================================================
# NORMALIZACION Z-SCORE
# ===========================================================================

def calcular_media(valores):
    return sum(valores) / len(valores)

def calcular_sigma(valores, mu):
    return math.sqrt(sum((v - mu)**2 for v in valores) / len(valores))

def normalizar_zscore_multivariable(datos, num_features):

    # Separar columnas
    columnas_x = [[datos[i][j] for i in range(len(datos))] for j in range(num_features)]
    columna_y  = [datos[i][num_features] for i in range(len(datos))]  # y REAL

    # Z-score SOLO para X
    params_x = []
    for col in columnas_x:
        mu    = calcular_media(col)
        sigma = calcular_sigma(col, mu)
        params_x.append((mu, sigma))

    # Normalizar SOLO X
    datos_norm = []
    for fila in datos:
        xs_norm = tuple((fila[j] - params_x[j][0]) / params_x[j][1]
                        for j in range(num_features))
        y_real = fila[num_features]   # <-- y sin tocar
        datos_norm.append(xs_norm + (y_real,))

    return datos_norm, params_x

# ===========================================================================
# CONSTRUCCION MATRICIAL DEL DATASET
# ===========================================================================

def construir_matrices(datos_norm, num_features):

    X = [[1.0] + list(fila[:num_features]) for fila in datos_norm]
    y = [fila[num_features] for fila in datos_norm]
    return X, y

# ===========================================================================
# OPERACIONES MATRICIALES DESDE CERO
# ===========================================================================

def predecir_matricial(X, w):

    return [sum(X[i][j] * w[j] for j in range(len(w))) for i in range(len(X))]

def calcular_mse(y, y_hat):

    n = len(y)
    return sum((y[i] - y_hat[i])**2 for i in range(n)) / n

def calcular_gradiente(X, y, w):

    n = len(X)
    d = len(w)

    # e = y - X·w
    y_hat = predecir_matricial(X, w)
    e = [y[i] - y_hat[i] for i in range(n)]

    # ∇wJ = (-2/N) · X^T · e
    grad = [(-2.0 / n) * sum(X[i][j] * e[i] for i in range(n)) for j in range(d)]
    return grad

# ===========================================================================
# INICIALIZACION DE PESOS
# ===========================================================================

def init_ceros(d):

    return [0.0] * d

def init_aleatorio(d, semilla=42):
    random.seed(semilla)
    return [random.uniform(-1.0, 1.0) for _ in range(d)]

def init_historico(historico):
  
    return list(historico)

# ===========================================================================
# GRADIENT DESCENT MULTIVARIABLE — ALGORITMO VECTORIZADO
# ===========================================================================

def gradient_descent_multivariable(X, y, w_init, lr, iteraciones,
                                   epsilon=1e-8, verbose=True,
                                   mostrar_cada=300, nombre="GD"):

    w = list(w_init)
    d = len(w)
    historial = []   # (iteracion, lista_w, mse)

    if verbose:
        separador(f"GD MULTIVARIABLE — {nombre}")
        print(f" lr={lr} | N={len(X)} | d={d-1} features | "
              f"iter_max={iteraciones} | eps={epsilon}")
        etiquetas = ["w0(bias)", "w1(horas_est)", "w2(suenio)", "w3(asistencia)", "w4(ejercicios)"]
        header = f" {'Iter':>6} " + " ".join(f"{etiquetas[j]:>11}" for j in range(d)) \
                 + f" {'MSE':>14} {'||∇w||':>12}"
        print(header)
        print(" " + "-"*6 + " " + " ".join(["-"*11]*d) + " " + "-"*14 + " " + "-"*12)

    iter_final = iteraciones

    for i in range(iteraciones):

        # --- Forward pass: y_hat = X · w    ---
        y_hat = predecir_matricial(X, w)

        # --- Costo: J = (1/N)||y - y_hat||²    ---
        mse = calcular_mse(y, y_hat)
        historial.append((i, list(w), mse))

        if verbose and (i % mostrar_cada == 0 or i < 5):
            w_str  = " ".join(f"{wj:>11.5f}" for wj in w)
            norm_g = math.sqrt(sum(g**2 for g in calcular_gradiente(X, y, w)))
            print(f" {i:>6} {w_str} {mse:>14.6f} {norm_g:>12.2e}")

        # --- Gradiente: ∇w = (-2/N) · X^T · e  ---
        grad = calcular_gradiente(X, y, w)

        # --- Criterio de parada: ||∇w|| < epsilon  ---
        norm_grad = math.sqrt(sum(g**2 for g in grad))
        if norm_grad < epsilon:
            if verbose:
                print(f"\n [Convergencia] iter={i} | ||∇w||={norm_grad:.2e} < eps={epsilon}")
            iter_final = i
            break

        # --- Actualizar: w <- w - α · ∇w ---
        w = [w[j] - lr * grad[j] for j in range(d)]

    # Estado final
    y_hat_f = predecir_matricial(X, w)
    mse_f   = calcular_mse(y, y_hat_f)
    historial.append((iter_final, list(w), mse_f))

    if verbose:
        w_str = " ".join(f"{wj:>11.5f}" for wj in w)
        print(f" {'FINAL':>6} {w_str} {mse_f:>14.6f}")
        print(f"\n Resultado {nombre}:")
        etiquetas = ["w0(bias)", "w1(horas_estudio)", "w2(horas_suenio)",
                     "w3(asistencia_pct)", "w4(ejercicios_resueltos)"]
        for j, wj in enumerate(w):
            print(f"   {etiquetas[j]} = {wj:.6f}")
        print(f" MSE final: {mse_f:.6f}")

    return w, historial

# ===========================================================================
# SOLUCION ANALITICA — NORMAL EQUATION
# ===========================================================================

def solucion_analitica(X, y):

    X_np = np.array(X)
    y_np = np.array(y)
    w_star = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np
    return w_star.tolist()

# ===========================================================================
# UTILIDADES
# ===========================================================================

def separador(titulo):
    linea = "=" * 75
    print(f"\n{linea}")
    print(f"  {titulo}")
    print(linea)

# ===========================================================================
# GRAFICA DE CONVERGENCIA
# ===========================================================================

def graficar_convergencia_3_lr(X, y, w_ceros, w_rand, w_hist, mse_analitico):

    lrs = [0.5, 0.05, 0.001]
    nombres_lr = ["lr = 0.5", "lr = 0.05", "lr = 0.001"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    inits = [
        ("Ceros", w_ceros, '#2196F3'),
        ("Aleatorio", w_rand, '#FF5722'),
        ("Historico", w_hist, '#4CAF50')
    ]

    for ax, lr, titulo in zip(axes, lrs, nombres_lr):

        for nombre, w_init, color in inits:

            _, hist = gradient_descent_multivariable(
                X, y, w_init,
                lr=lr,
                iteraciones=2000,
                epsilon=1e-8,
                verbose=False
            )

            iters = [h[0] for h in hist]
            mses  = [h[2] for h in hist]

            ax.plot(iters, mses, color=color, linewidth=2, label=nombre)

        ax.axhline(y=mse_analitico, color='gold', linestyle='--', label='Analitico')
        ax.set_title(titulo)
        ax.set_xlabel("Iteración")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def barras_w_inicial_vs_final(w_inicial, w_final):
    etiquetas = ["w0(bias)", "w1(horas_est)", "w2(suenio)", "w3(asist)", "w4(ejerc)"]
    
    x = np.arange(len(etiquetas))
    ancho = 0.35

    plt.figure(figsize=(9,5))
    plt.bar(x - ancho/2, w_inicial, width=ancho, label="w inicial (historico)")
    plt.bar(x + ancho/2, w_final,   width=ancho, label="w final (GD)")

    plt.xticks(x, etiquetas)
    plt.ylabel("Valor del peso")
    plt.title("Comparación de pesos: Inicial vs Final")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    NUM_FEATURES = 4   # horas_estudio | horas_suenio | asistencia_pct | ejercicios_resueltos

    print("\n" + "#"*75)
    print("  GD MULTIVARIABLE | Z-score | 3 Inicializaciones de Pesos")
    print("  Features: horas_estudio | horas_suenio | asistencia_pct | ejercicios_resueltos -> nota_final")
    print("#"*75)

    # -----------------------------------------------------------------------
    # 1. NORMALIZAR CON Z-SCORE
    # -----------------------------------------------------------------------
    datos_norm, params_x = normalizar_zscore_multivariable(
        datos, NUM_FEATURES
    )

    nombres_features = ["horas_estudio", "horas_suenio", "asistencia_pct", "ejercicios_resueltos"]
    print(f"\n Dataset: {len(datos)} muestras | {NUM_FEATURES} features + 1 target")
    print(f"\n Z-score — parametros de normalizacion:")
    for j, nombre in enumerate(nombres_features):
        mu_j, sig_j = params_x[j]
        print(f"   {nombre:<15}  mu = {mu_j:>10.4f}  |  sigma = {sig_j:>10.4f}")
    print(f"   {'nota_final':<15}  (NO normalizado)")

    X, y = construir_matrices(datos_norm, NUM_FEATURES)
    d = len(X[0])   # d+1 = 5  (bias + 4 features)

    print(f"\n Dimensiones matriciales:")
    print(f"   X : {len(X)} x {len(X[0])}   [N x (d+1)] con bias trick")
    print(f"   y : {len(y)} x 1")
    print(f"   w : {d} x 1   [w0(bias), w1, w2, w3, w4]")

    w_analitico   = solucion_analitica(X, y)
    mse_analitico = calcular_mse(y, predecir_matricial(X, w_analitico))

    separador("SOLUCION ANALITICA — Normal Equation  w* = (X^T X)^-1 · X^T · y")
    etiquetas = ["w0(bias)", "w1(horas_estudio)", "w2(horas_suenio)",
                 "w3(asistencia_pct)", "w4(ejercicios_resueltos)"]
    for j, wj in enumerate(w_analitico):
        print(f"   {etiquetas[j]} = {wj:.6f}")
    print(f" MSE exacto = {mse_analitico:.6f}")

    # -----------------------------------------------------------------------
    # 4. HIPERPARAMETROS
    # -----------------------------------------------------------------------
    lr       = 0.05
    max_iter = 2000
    epsilon  = 1e-8

    # -----------------------------------------------------------------------
    # 5. INICIALIZACIONES DE PESOS
    # -----------------------------------------------------------------------
    historico = [15.08740, 2.68994, 0.60019, 0.99721, 0.93793]

    w_ceros = init_ceros(d)
    w_rand  = init_aleatorio(d, semilla=42)
    w_hist  = init_historico(historico)

    separador("INICIALIZACIONES DE PESOS")
    print(f" Ceros     : {w_ceros}")
    print(f" Aleatorio : {[round(w, 4) for w in w_rand]}")
    print(f" Historico : {w_hist}")

    # -----------------------------------------------------------------------
    # 6. GRADIENT DESCENT MULTIVARIABLE
    # -----------------------------------------------------------------------
    w_gd_ceros, hist_ceros = gradient_descent_multivariable(
        X, y, w_ceros, lr=lr, iteraciones=max_iter,
        epsilon=epsilon, verbose=True, nombre="Init=Ceros"
    )

    w_gd_rand, hist_rand = gradient_descent_multivariable(
        X, y, w_rand, lr=lr, iteraciones=max_iter,
        epsilon=epsilon, verbose=True, nombre="Init=Aleatorio"
    )

    w_gd_hist, hist_hist = gradient_descent_multivariable(
        X, y, w_hist, lr=lr, iteraciones=max_iter,
        epsilon=epsilon, verbose=True, nombre="Init=Historico"
    )

    # -----------------------------------------------------------------------
    # 7. RESUMEN COMPARACION
    # -----------------------------------------------------------------------
    separador("RESUMEN COMPARACION")
    mse_c = calcular_mse(y, predecir_matricial(X, w_gd_ceros))
    mse_r = calcular_mse(y, predecir_matricial(X, w_gd_rand))
    mse_h = calcular_mse(y, predecir_matricial(X, w_gd_hist))

    fmt  = f" {{:<30}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>14}}"
    fmt2 = f" {{:<30}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>14}}"
    print(fmt.format("Metodo", "w0", "w1", "w2", "w3", "w4", "MSE"))
    print(" " + "-"*30 + " " + " ".join(["-"*10]*5) + " " + "-"*14)

    def fmtw(lst):
        return [f"{v:.5f}" for v in lst]

    def print_row(nombre, w, mse):
        print(fmt2.format(nombre, *fmtw(w), f"{mse:.6f}"))

    print_row("Analitico (exacto)", w_analitico, mse_analitico)
    print_row("GD  Init=Ceros",     w_gd_ceros,  mse_c)
    print_row("GD  Init=Aleatorio", w_gd_rand,   mse_r)
    print_row("GD  Init=Historico", w_gd_hist,   mse_h)

    print("\n Diferencias respecto a la solucion analitica:")
    for nombre_m, w_m in [("Ceros",     w_gd_ceros),
                           ("Aleatorio", w_gd_rand),
                           ("Historico", w_gd_hist)]:
        dw = [abs(w_m[j] - w_analitico[j]) for j in range(d)]
        print(f"   {nombre_m:<12}: " + "  ".join(f"Δw{j}={dw[j]:.2e}" for j in range(d)))

    # -----------------------------------------------------------------------
    # 8. GRAFICA DE CONVERGENCIA
    # -----------------------------------------------------------------------
    graficar_convergencia_3_lr(
    X, y,
    w_ceros, w_rand, w_hist,
    mse_analitico
    )

    barras_w_inicial_vs_final(w_hist, w_gd_hist)