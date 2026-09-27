import matplotlib.pyplot as plt

def leer_csv(ruta):
    X = []
    y = []
    try:
        with open(ruta, 'r', encoding='utf-8') as archivo:
            lineas = archivo.readlines()
            if not lineas:
                return X, y
            
            for linea in lineas[1:]:
                valores = linea.strip().split(',')
                if len(valores) > 1:
                    valores_numericos = [float(v) for v in valores]
                    X.append(valores_numericos[:-1])
                    y.append(valores_numericos[-1])
    except FileNotFoundError:
        print(f"No se encontró el archivo: {ruta}")
    return X, y

def hardlim(n):
    return 1.0 if n >= 0 else 0.0

def relu(n):
    return float(max(0.0, n))

def producto_punto_y_sesgo(x, w, b):
    suma = b
    for i in range(len(x)):
        suma += x[i] * w[i]
    return suma

def graficar_resultados(X, y_real, y_pred):
    x1 = [fila[0] for fila in X]
    x2 = [fila[1] if len(fila) > 1 else 0 for fila in X]
    # asi me aseguro de usar solo dos dimensiones en los ejes

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.scatter(x1, x2, c=y_real, cmap='coolwarm', edgecolor='k')
    ax1.set_title("Esperado")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax2.scatter(x1, x2, c=y_pred, cmap='coolwarm', edgecolor='k')
    ax2.set_title("Predicción")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    colores = ['green' if real == pred else 'red' for real, pred in zip(y_real, y_pred)]
    ax3.scatter(x1, x2, c=colores, edgecolor='k')
    ax3.set_title("Aciertos (Verde) / Errores (Rojo)")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")

    plt.tight_layout()
    plt.show()

def main():
    ruta = input("Ruta del CSV: ")
    X, y_real = leer_csv(ruta)
    
    if not X:
        print("No hay datos para procesar.")
        return

    num_entradas = len(X[0])
    
    continuar = 's'
    while continuar.lower() == 's':
        print("\n--- Pesos y bias ---")
        b = float(input("Bias (b): "))
        
        w = []
        for i in range(num_entradas):
            w.append(float(input(f"Peso w{i+1}: ")))
            
        opcion = input("Activación (1: Hardlim [0,1], 2: ReLU [0,∞]): ")
        
        y_pred = []
        for x in X:
            n = producto_punto_y_sesgo(x, w, b)
            a = relu(n) if opcion == '2' else hardlim(n)
            y_pred.append(a)
            
        graficar_resultados(X, y_real, y_pred)
        continuar = input("\nProbar con otros valores? (s/n): ")

if __name__ == "__main__":
    main()