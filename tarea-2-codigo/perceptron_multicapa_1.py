# Nombre del estudiante: Diego Moreno Ricci
# Cédula: [Tu Cédula aquí]

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

class RedMulticapa:
    def __init__(self, entradas, salidas, capas_ocultas, neuronas_capa, epocas=0):
        self.u = int(entradas)       # Neuronas de entrada
        self.v = int(salidas)       # Neuronas de salida / clases
        self.L = int(capas_ocultas) # Capas ocultas
        self.b = int(neuronas_capa) # Neuronas por capa oculta
        e = int(epocas)             # Épocas acumuladas
        
        self.e = e
        self.pesos = []
        self.sesgos = []
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        """Inicialización de pesos usando Xavier/Glorot Uniforme."""
        np.random.seed(None)
        self.pesos = []
        self.sesgos = []
        dim_actual = self.u

        # Capas ocultas
        for _ in range(self.L):
            limite = np.sqrt(6.0 / (dim_actual + self.b))
            w = np.random.uniform(-limite, limite, (dim_actual, self.b))
            b = np.zeros((1, self.b))
            self.pesos.append(w)
            self.sesgos.append(b)
            dim_actual = self.b

        # Capa de salida
        limite_out = np.sqrt(6.0 / (dim_actual + self.v))
        w_out = np.random.uniform(-limite_out, limite_out, (dim_actual, self.v))
        b_out = np.zeros((1, self.v))
        self.pesos.append(w_out)
        self.sesgos.append(b_out)

    @staticmethod
    def _sigmoide(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _derivada_sigmoide(a):
        return a * (1.0 - a)

    def propagacion_adelante(self, X):
        """Realiza el paso hacia delante (Feedforward) guardando activaciones."""
        activaciones = [X]
        entrada_actual = X

        for w, b in zip(self.pesos, self.sesgos):
            z = np.dot(entrada_actual, w) + b
            a = self._sigmoide(z)
            activaciones.append(a)
            entrada_actual = a

        return activaciones

    def predecir(self, X):
        """Predice las clases para un conjunto de entradas X."""
        activaciones = self.propagacion_adelante(X)
        salida = activaciones[-1]

        if self.v == 1:
            return np.where(salida.flatten() >= 0.5, 1, -1)
        else:
            return np.argmax(salida, axis=1) + 1

    def entrenar(self, X, y, epocas_entrenamiento, tasa_aprendizaje=0.1):
        """Entrena la red mediante retropropagación (Backpropagation)."""
        if self.v == 1:
            Y_obj = np.where(y.reshape(-1, 1) == 1, 1.0, 0.0)
        else:
            Y_obj = np.zeros((len(y), self.v))
            for idx, etiqueta in enumerate(y):
                c_idx = int(etiqueta) - 1
                if 0 <= c_idx < self.v:
                    Y_obj[idx, c_idx] = 1.0

        historial_errores = []

        for _ in range(epocas_entrenamiento):
            activaciones = self.propagacion_adelante(X)
            prediccion_final = activaciones[-1]

            # Error Cuadrático Medio (MSE)
            error = np.mean((prediccion_final - Y_obj) ** 2)
            historial_errores.append(error)

            # Delta de la capa de salida
            delta = (prediccion_final - Y_obj) * self._derivada_sigmoide(prediccion_final)

            # Retropropagación por capas
            for capa in range(len(self.pesos) - 1, -1, -1):
                act_entrada = activaciones[capa]
                grad_w = np.dot(act_entrada.T, delta) / len(X)
                grad_b = np.sum(delta, axis=0, keepdims=True) / len(X)

                if capa > 0:
                    delta = np.dot(delta, self.pesos[capa].T) * self._derivada_sigmoide(activaciones[capa])

                self.pesos[capa] -= tasa_aprendizaje * grad_w
                self.sesgos[capa] -= tasa_aprendizaje * grad_b

        self.e += epocas_entrenamiento
        return historial_errores


# MANEJO DE ARCHIVOS CSV
def guardar_red(red, ruta):
    max_c = max(red.u, red.b)
    total_cols = max_c + 4

    with open(ruta, mode='w', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)
        
        def escribir_hp(letra, valor):
            escritor.writerow([letra, str(valor)] + [''] * (total_cols - 2))

        escribir_hp('u', red.u)
        escribir_hp('v', red.v)
        escribir_hp('L', red.L)
        escribir_hp('b', red.b)
        escribir_hp('e', red.e)

        encabezado = ['tipo', 'capa', 'posicion'] + [f'w{i}' for i in range(max_c + 1)]
        escritor.writerow(encabezado)

        # Neuronas ocultas
        for l_idx in range(red.L):
            w_capa = red.pesos[l_idx]
            b_capa = red.sesgos[l_idx]
            for n_pos in range(red.b):
                pesos_neurona = [b_capa[0, n_pos]] + w_capa[:, n_pos].tolist()
                relleno = [''] * (total_cols - len(['h', l_idx + 1, n_pos] + pesos_neurona))
                escritor.writerow(['h', l_idx + 1, n_pos] + pesos_neurona + relleno)

        # Neuronas de salida
        w_salida = red.pesos[-1]
        b_salida = red.sesgos[-1]
        capa_salida_num = red.L + 1
        for n_pos in range(red.v):
            pesos_neurona = [b_salida[0, n_pos]] + w_salida[:, n_pos].tolist()
            relleno = [''] * (total_cols - len(['o', capa_salida_num, n_pos] + pesos_neurona))
            escritor.writerow(['o', capa_salida_num, n_pos] + pesos_neurona + relleno)


def cargar_red(ruta):
    with open(ruta, mode='r', encoding='utf-8') as f:
        filas = list(csv.reader(f))

    u = int(filas[0][1])
    v = int(filas[1][1])
    L = int(filas[2][1])
    b = int(filas[3][1])
    e = int(filas[4][1])

    red = RedMulticapa(u, v, L, b, e)
    filas_neuronas = filas[6:]

    for l_idx in range(len(red.pesos)):
        dim_entrada = red.u if l_idx == 0 else red.b
        tipo_buscado = 'h' if l_idx < red.L else 'o'
        capa_num = l_idx + 1

        filas_capa = [r for r in filas_neuronas if len(r) > 2 and r[0] == tipo_buscado and int(r[1]) == capa_num]

        for r in filas_capa:
            pos = int(r[2])
            red.sesgos[l_idx][0, pos] = float(r[3])
            pesos_neurona = [float(val) for val in r[4:4 + dim_entrada]]
            red.pesos[l_idx][:, pos] = pesos_neurona

    return red


def cargar_dataset(ruta_csv):
    with open(ruta_csv, mode='r', encoding='utf-8') as f:
        primera_linea = f.readline().strip()
        if primera_linea.startswith(('u,', 'v,', 'L,', 'b,', 'e,')) or (len(primera_linea.split(',')) > 0 and primera_linea.split(',')[0] in ['u', 'v', 'L', 'b', 'e']):
            raise ValueError("El archivo proporcionado es un archivo de red neuronal, no un dataset de datos. Debe ingresar un archivo CSV de prueba o entrenamiento válido (ej. 3-d_4-class_test.csv).")

    datos = np.loadtxt(ruta_csv, delimiter=',', skiprows=1)
    if datos.ndim == 1:
        datos = datos.reshape(1, -1)
    X = datos[:, :-1]
    y = datos[:, -1]
    return X, y


# VISUALIZACIÓN
def mostrar_graficos(X, y_real, y_pred, es_entrenamiento, historial_error=None, num_clases=2):
    fig = plt.figure(figsize=(16, 5))
    num_entradas = X.shape[1]

    # Gráfico 1: Dispersión de entradas (2D o 3D)
    if num_entradas >= 3:
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_real, cmap='tab10', edgecolors='k', s=50)
        ax1.set_title("1. Datos de Entrada (3D)")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("x3")
    else:
        ax1 = fig.add_subplot(131)
        ax1.scatter(X[:, 0], X[:, 1], c=y_real, cmap='tab10', edgecolors='k', s=70)
        ax1.set_title("1. Datos de Entrada (2D)")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")

    # Gráfico 2: Aciertos y Errores (Verde vs Rojo)
    ax2 = fig.add_subplot(132)
    aciertos = (y_real == y_pred)
    colores = ['#2ca02c' if ok else '#d62728' for ok in aciertos]
    
    ax2.scatter(X[:, 0], X[:, 1], c=colores, edgecolors='k', s=70)
    ax2.set_title("2. Predicciones (Verde=Correcto, Rojo=Error)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    # Gráfico 3: Curva de Error o Matriz de Confusión
    ax3 = fig.add_subplot(133)
    if es_entrenamiento:
        ax3.plot(historial_error, color='#1f77b4', linewidth=2)
        ax3.set_title("3. Error de la Red por Época")
        ax3.set_xlabel("Épocas")
        ax3.set_ylabel("Error Cuadrático Medio (MSE)")
        ax3.grid(True, linestyle='--', alpha=0.6)
    else:
        clases = np.unique(np.concatenate([y_real, y_pred]))
        n_clases = max(num_clases, len(clases))
        matriz_conf = np.zeros((n_clases, n_clases), dtype=int)
        
        for real, pred in zip(y_real, y_pred):
            i_r = int(real) - 1 if real in [1, 2, 3, 4] else int(real)
            i_p = int(pred) - 1 if pred in [1, 2, 3, 4] else int(pred)
            if 0 <= i_r < n_clases and 0 <= i_p < n_clases:
                matriz_conf[i_r, i_p] += 1

        cax = ax3.matshow(matriz_conf, cmap='Blues', alpha=0.8)
        fig.colorbar(cax, ax=ax3)
        ax3.set_title("3. Matriz de Confusión", pad=15)
        ax3.set_xticks(range(n_clases))
        ax3.set_yticks(range(n_clases))
        ax3.set_xticklabels([f"C{i+1}" for i in range(n_clases)])
        ax3.set_yticklabels([f"C{i+1}" for i in range(n_clases)])
        ax3.set_xlabel("Predicción")
        ax3.set_ylabel("Real")

        for i in range(n_clases):
            for j in range(n_clases):
                ax3.text(j, i, str(matriz_conf[i, j]), va='center', ha='center', color='black', weight='bold')

    plt.tight_layout()
    plt.show()


# INTERFAZ Y MENÚS
def pedir_entero_positivo(mensaje):
    while True:
        try:
            val = int(input(mensaje))
            if val > 0:
                return val
            print("❌ Debe ingresar un número entero mayor a 0.")
        except ValueError:
            print("❌ Entrada inválida. Ingrese un número entero.")


def imprimir_hiperparametros(red):
    print("\n" + "="*45)
    print("      HIPERPARÁMETROS ACTUALES DE LA RED")
    print("="*45)
    print(f" • u (Neuronas de entrada):       {red.u}")
    print(f" • v (Neuronas de salida/clases):  {red.v}")
    print(f" • L (Capas ocultas):            {red.L}")
    print(f" • b (Neuronas por capa oculta):   {red.b}")
    print(f" • e (Épocas acumuladas):        {red.e}")
    print("="*45 + "\n")


def main():
    print("="*55)
    print("   SISTEMA DE PERCEPTRÓN MULTICAPA (TAREA 2)")
    print("="*55)

    red = None

    while True:
        print("\nMENÚ INICIAL:")
        print("1. Crear un nuevo perceptrón multicapa")
        print("2. Cargar perceptrón desde un archivo")
        opc = input("Seleccione una opción (1/2): ").strip()

        if opc == '1':
            u = pedir_entero_positivo("Número de neuronas de entrada (u): ")
            v = pedir_entero_positivo("Número de neuronas de salida/clases (v): ")
            L = pedir_entero_positivo("Número de capas ocultas (L): ")
            b = pedir_entero_positivo("Número de neuronas por capa oculta (b): ")
            red = RedMulticapa(u, v, L, b)
            print("\n✅ Red neuronal creada exitosamente.")
            break
        elif opc == '2':
            ruta = input("Ingrese la ruta del archivo CSV de la red: ").strip()
            if os.path.exists(ruta):
                try:
                    red = cargar_red(ruta)
                    print("\n✅ Red cargada exitosamente.")
                    break
                except Exception as ex:
                    print(f"❌ Error al leer la estructura de la red: {ex}\n")
            else:
                print("❌ El archivo especificado no existe.\n")
        else:
            print("❌ Opción inválida.\n")

    imprimir_hiperparametros(red)

    while True:
        print("\nMENÚ PRINCIPAL:")
        print("1. Entrenar la red")
        print("2. Probar la red")
        print("3. Guardar la red")
        print("4. Salir")

        opcion = input("Seleccione una opción (1-4): ").strip()

        if opcion == '1':
            ruta_train = input("Ruta del archivo CSV de entrenamiento: ").strip()
            if not os.path.exists(ruta_train):
                print("❌ El archivo de entrenamiento no existe.\n")
                continue

            try:
                X, y = cargar_dataset(ruta_train)
            except Exception as ex:
                print(f"❌ Error al cargar el dataset: {ex}\n")
                continue

            if X.shape[1] != red.u:
                print(f"\n❌ ERROR DE CONFIGURACIÓN: El archivo tiene {X.shape[1]} entradas,")
                print(f"pero la red requiere u = {red.u} neuronas de entrada.")
                print(f"Configure o cree una red con u = {X.shape[1]} para usar este archivo.\n")
                continue

            epocas = pedir_entero_positivo("Número de épocas de entrenamiento: ")
            
            print("\n🔄 Entrenando la red con retropropagación...")
            historial = red.entrenar(X, y, epocas)
            print("✨ ¡Entrenamiento finalizado exitosamente!")

            imprimir_hiperparametros(red)
            y_pred = red.predecir(X)
            mostrar_graficos(X, y, y_pred, es_entrenamiento=True, historial_error=historial)

        elif opcion == '2':
            ruta_test = input("Ruta del archivo CSV de prueba: ").strip()
            if not os.path.exists(ruta_test):
                print("❌ El archivo de prueba no existe.\n")
                continue

            try:
                X, y = cargar_dataset(ruta_test)
            except Exception as ex:
                print(f"❌ Error al cargar el dataset: {ex}\n")
                continue

            if X.shape[1] != red.u:
                print(f"\n❌ ERROR DE CONFIGURACIÓN: El archivo tiene {X.shape[1]} entradas,")
                print(f"pero la red requiere u = {red.u} neuronas de entrada.")
                print(f"Configure o cargue una red con u = {X.shape[1]} para usar este archivo.\n")
                continue

            print("\n🔄 Probando la red con alimentación hacia delante...")
            y_pred = red.predecir(X)
            print("✨ ¡Prueba finalizada exitosamente!")

            imprimir_hiperparametros(red)
            mostrar_graficos(X, y, y_pred, es_entrenamiento=False, num_clases=red.v)

        elif opcion == '3':
            ruta_save = input("Ruta o nombre de archivo para guardar la red: ").strip()
            try:
                guardar_red(red, ruta_save)
                print(f"✅ Red guardada exitosamente en '{ruta_save}'.\n")
            except Exception as ex:
                print(f"❌ Error al guardar el archivo: {ex}\n")

        elif opcion == '4':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida. Ingrese un número del 1 al 4.\n")

if __name__ == "__main__":
    main()