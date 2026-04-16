import pandas as pd
import numpy as np

# ===========================================================================
# 1. CARGA DE DATOS
# ===========================================================================
df = pd.read_csv('caso2_notas.csv')
X = df[['horas_estudio', 'horas_suenio', 'asistencia_pct', 'ejercicios_resueltos']].values
Y = df['nota_final ( Y )'].values

# ===========================================================================
# 2. NORMALIZACIÓN Z-SCORE
# ===========================================================================
def normalizar_datos(X):
    # Normalización Z-Score: (X - media) / desviación_estándar
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = normalizar_datos(X)
print("Datos X normalizados con Z-Score.")

# ===========================================================================
# 3. SOLUCIÓN ANALÍTICA
# ===========================================================================
N = len(X_norm)
columna_unos = np.ones((N, 1))
X_matriz = np.hstack((columna_unos, X_norm))

# Fórmula: W = (X^T * X)^-1 * X^T * Y
X_T_X = np.dot(X_matriz.T, X_matriz)
Inversa = np.linalg.inv(X_T_X)
Inversa_X_T = np.dot(Inversa, X_matriz.T)
W = np.dot(Inversa_X_T, Y)

# ===========================================================================
# RESULTADOS
# ===========================================================================
b_exacto = W[0]
pesos_exactos = W[1:]
print("\n--- RESULTADOS EXACTOS (Z-SCORE) ---")
print(f"Bias (b): {b_exacto:.5f}")
print(f"Pesos (w): {[round(w, 5) for w in pesos_exactos]}")