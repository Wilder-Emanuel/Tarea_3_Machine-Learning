import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FUNCIONES DEL ALGORITMO MULTIVARIABLE

def normalizar_datos(X):
    # Normalización Z-Score: (X - media) / desviación_estándar
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def gradient_descent_multivariable(X, y, lr=0.01, epocas=500, epsilon=1e-6):
    """
    Implementación vectorial del Gradiente Descendente.
    X: Matriz de características
    y: Vector de variable objetivo
    """
    n_muestras, n_features = X.shape
    
    # 1. INICIALIZACIÓN CON CEROS
    w = np.zeros(n_features)
    
    historial_costo = []

    for i in range(epocas):
        # 2. FORWARD: Predicciones (y_hat = X @ w)
        y_pred = np.dot(X, w)
        
        # 3. COSTO (MSE)
        error = y - y_pred
        costo = np.mean(error**2)
        historial_costo.append(costo)

        # 4. GRADIENTE
        gradiente = (-2 / n_muestras) * np.dot(X.T, error)

        # 5. ACTUALIZACIÓN
        w_nuevo = w - lr * gradiente

        # Condición de parada si la mejora es insignificante
        if i > 0 and abs(historial_costo[-2] - historial_costo[-1]) < epsilon:
            break
        
        w = w_nuevo

    return w, historial_costo

# ===========================================================================

if __name__ == "__main__":
    # 1. Cargar el dataset
    
    df = pd.read_csv('caso2_notas.csv')

    features = ['horas_estudio', 'horas_suenio', 'asistencia_pct', 'ejercicios_resueltos']
    target = 'nota_final ( Y )'

    # 2. Preparar matrices (X e y)
    X = df[features].values
    y = df[target].values

    # 3. Normalizar características
    X_norm, mu, sigma = normalizar_datos(X)

    # 4. Añadir columna de UNOS para el Bias (w0)
    X_final = np.c_[np.ones(X_norm.shape[0]), X_norm]

    # 5. Learning Rates diferentes
    learning_rates = [0.5, 0.05, 0.001]
    colores = ['red', 'blue', 'green']
    
    plt.figure(figsize=(12, 6))

    print(f"{'LR':>10} | {'Iteraciones':>12} | {'Costo Final':>12}")
    print("-" * 40)

    for lr_test, color in zip(learning_rates, colores):
        # Entrenar
        w_final, historial = gradient_descent_multivariable(
            X_final, y, lr=lr_test, epocas=4000
        )
        
        # Graficar caída del gradiente
        plt.plot(historial, label=f'LR = {lr_test}', color=color, linewidth=2)
        
        print(f"{lr_test:>10} | {len(historial):>12} | {historial[-1]:>12.6f}")

    # 6. Formato de la gráfica
    plt.title('Comparativa de Caída del Gradiente: Learning Rates', fontsize=14)
    plt.xlabel('Épocas (Iteraciones)')
    plt.ylabel('Costo (MSE)')
    plt.yscale('log') # Escala logarítmica para ver mejor la caída
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    for feature, peso in zip(features, w_final[1:]):
        impacto_absoluto = abs(peso)
        print(f"{feature:>15}: {peso:>10.5f}  |  Impacto (Abs): {impacto_absoluto:.5f}")
    print("="*50)
    
    print("\n[INFO] Gráfica generada. Analiza cuál LR converge más rápido.")
    plt.show()
