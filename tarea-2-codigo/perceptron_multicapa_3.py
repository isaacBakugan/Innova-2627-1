# Nombre del integrante: Jonathan Sarli
# Cédula del integrante: 30496924

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

class PerceptronMulticapa:
    def __init__(self, u, v, L, b, e=0):
        self.u = int(u)  # Entradas
        self.v = int(v)  # Salidas / Clases
        self.L = int(L)  # Capas ocultas
        self.b = int(b)  # Neuronas por capa oculta
        self.e = int(e)  # Épocas acumuladas

        self.weights = []
        self.biases = []
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        """Inicialización de He/Xavier adaptada para evitar saturación de Sigmoide."""
        self.weights = []
        self.biases = []
        in_dim = self.u

        # Capas Ocultas
        for _ in range(self.L):
            std = np.sqrt(2.0 / in_dim)
            w = np.random.randn(in_dim, self.b) * std
            b = np.zeros((1, self.b))
            self.weights.append(w)
            self.biases.append(b)
            in_dim = self.b

        # Capa de Salida
        std_out = np.sqrt(2.0 / in_dim)
        w_out = np.random.randn(in_dim, self.v) * std_out
        b_out = np.zeros((1, self.v))
        self.weights.append(w_out)
        self.biases.append(b_out)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, X):
        """Pase hacia adelante (Feedforward)."""
        activaciones = [X]
        curr_input = X

        for w, b in zip(self.weights, self.biases):
            z = np.dot(curr_input, w) + b
            a = self.sigmoid(z)
            activaciones.append(a)
            curr_input = a

        return activaciones

    def predict(self, X):
        """Predice las clases finales de cada vector."""
        activaciones = self.forward(X)
        salida = activaciones[-1]

        if self.v == 1:
            return np.where(salida.flatten() >= 0.5, 1, -1)
        else:
            return np.argmax(salida, axis=1) + 1

    def train(self, X, y, epocas, lr=0.05):
        #Entrena la red con Retropropagación (Backpropagation).
        if self.v == 1:
            Y_target = np.where(y.reshape(-1, 1) == 1, 1.0, 0.0)
        else:
            Y_target = np.zeros((len(y), self.v))
            for idx, val in enumerate(y):
                c_idx = int(val) - 1
                if 0 <= c_idx < self.v:
                    Y_target[idx, c_idx] = 1.0

        historial_error = []

        for _ in range(epocas):
            activaciones = self.forward(X)
            output = activaciones[-1]

            # MSE Loss
            loss = np.mean((output - Y_target) ** 2)
            historial_error.append(loss)

            # Delta en la capa de salida
            delta = (output - Y_target) * output * (1.0 - output)

            # Propagación hacia atrás
            for l in range(len(self.weights) - 1, -1, -1):
                input_act = activaciones[l]
                grad_w = np.dot(input_act.T, delta) / len(X)
                grad_b = np.sum(delta, axis=0, keepdims=True) / len(X)

                if l > 0:
                    delta = np.dot(delta, self.weights[l].T) * activaciones[l] * (1.0 - activaciones[l])

                self.weights[l] -= lr * grad_w
                self.biases[l] -= lr * grad_b

        self.e += epocas
        return historial_error



# 2. CARGA Y GUARDADO CSV
def guardar_red(red, ruta_archivo):
    max_c = max(red.u, red.b)
    with open(ruta_archivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u', red.u] + [''] * (max_c + 1))
        writer.writerow(['v', red.v] + [''] * (max_c + 1))
        writer.writerow(['L', red.L] + [''] * (max_c + 1))
        writer.writerow(['b', red.b] + [''] * (max_c + 1))
        writer.writerow(['e', red.e] + [''] * (max_c + 1))

        encabezado = ['tipo', 'capa', 'posicion'] + [f'w{i}' for i in range(max_c + 1)]
        writer.writerow(encabezado)

        for l_idx in range(red.L):
            w_layer = red.weights[l_idx]
            b_layer = red.biases[l_idx]
            for n_pos in range(red.b):
                pesos_n = [b_layer[0, n_pos]] + w_layer[:, n_pos].tolist()
                padding = [''] * ((max_c + 1) - len(pesos_n))
                writer.writerow(['h', l_idx + 1, n_pos] + pesos_n + padding)

        w_out = red.weights[-1]
        b_out = red.biases[-1]
        capa_salida_idx = red.L + 1
        for n_pos in range(red.v):
            pesos_n = [b_out[0, n_pos]] + w_out[:, n_pos].tolist()
            padding = [''] * ((max_c + 1) - len(pesos_n))
            writer.writerow(['o', capa_salida_idx, n_pos] + pesos_n + padding)


def cargar_red(ruta_archivo):
    with open(ruta_archivo, mode='r') as f:
        reader = list(csv.reader(f))

    u = int(reader[0][1])
    v = int(reader[1][1])
    L = int(reader[2][1])
    b = int(reader[3][1])
    e = int(reader[4][1])

    red = PerceptronMulticapa(u, v, L, b, e)
    filas_neuronas = reader[6:]

    for l_idx in range(len(red.weights)):
        in_dim = red.u if l_idx == 0 else red.b
        tipo_buscado = 'h' if l_idx < red.L else 'o'
        capa_num = l_idx + 1

        filas_capa = [r for r in filas_neuronas if r[0] == tipo_buscado and int(r[1]) == capa_num]

        for r in filas_capa:
            pos = int(r[2])
            red.biases[l_idx][0, pos] = float(r[3])
            pesos = [float(val) for val in r[4:4 + in_dim]]
            red.weights[l_idx][:, pos] = pesos

    return red


def cargar_dataset(ruta_csv):
    datos = np.loadtxt(ruta_csv, delimiter=',', skiprows=1)
    X = datos[:, :-1]
    y = datos[:, -1]
    return X, y



# 3. VISUALIZACIÓN
def mostrar_graficos(X, y_real, y_pred, es_entrenamiento, historial_error=None, num_clases=2):
    fig = plt.figure(figsize=(16, 5))
    num_entradas = X.shape[1]

    # Subplot 1
    if num_entradas >= 3:
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_real, cmap='tab10', edgecolors='k')
        ax1.set_title("1. Datos de Entrada (Proyección 3D)")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("x3")
    else:
        ax1 = fig.add_subplot(131)
        ax1.scatter(X[:, 0], X[:, 1], c=y_real, cmap='tab10', edgecolors='k', s=70)
        ax1.set_title("1. Datos de Entrada (2D)")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")

    # Subplot 2
    ax2 = fig.add_subplot(132)
    coincidencias = (y_real == y_pred)
    colores = ['forestgreen' if c else 'firebrick' for c in coincidencias]
    
    ax2.scatter(X[:, 0], X[:, 1], c=colores, edgecolors='k', s=70)
    ax2.set_title("2. Salida Predicha (Verde=Correcto, Rojo=Error)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    # Subplot 3
    ax3 = fig.add_subplot(133)
    if es_entrenamiento:
        ax3.plot(historial_error, color='blue', linewidth=2)
        ax3.set_title("3. Error de la Red por Época")
        ax3.set_xlabel("Épocas")
        ax3.set_ylabel("Pérdida (MSE)")
        ax3.grid(True)
    else:
        clases_unicas = np.arange(1, num_clases + 1)
        matriz = np.zeros((num_clases, num_clases), dtype=int)
        
        for r, p in zip(y_real, y_pred):
            i_r = int(r) - 1
            i_p = int(p) - 1
            if 0 <= i_r < num_clases and 0 <= i_p < num_clases:
                matriz[i_r, i_p] += 1

        cax = ax3.matshow(matriz, cmap='Blues')
        fig.colorbar(cax, ax=ax3)
        ax3.set_title("3. Matriz de Confusión", pad=20)
        ax3.set_xticks(range(num_clases))
        ax3.set_yticks(range(num_clases))
        ax3.set_xticklabels([f"C{int(c)}" for c in clases_unicas])
        ax3.set_yticklabels([f"C{int(c)}" for c in clases_unicas])
        ax3.set_xlabel("Predicción")
        ax3.set_ylabel("Real")

        for i in range(num_clases):
            for j in range(num_clases):
                ax3.text(j, i, str(matriz[i, j]), va='center', ha='center', color='black', weight='bold')

    plt.tight_layout()
    plt.show()


# 4. AUXILIARES Y MENÚ
def pedir_entero_positivo(mensaje):
    while True:
        try:
            val = int(input(mensaje))
            if val > 0:
                return val
            print("❌ Debe ser un entero positivo mayor a 0.")
        except ValueError:
            print("❌ Entrada inválida. Ingrese un entero válido.")


def imprimir_hiperparametros(red):
    print("\n" + "="*40)
    print("      HIPERPARÁMETROS ACTUALES DE LA RED")
    print("="*40)
    print(f" u (Entradas):          {red.u}")
    print(f" v (Salidas/Clases):    {red.v}")
    print(f" L (Capas ocultas):     {red.L}")
    print(f" b (Neuronas por capa): {red.b}")
    print(f" e (Épocas acumuladas): {red.e}")
    print("="*40 + "\n")


def main():
    print("="*50)
    print(" PERCEPTRÓN MULTICAPA ")
    print("="*50)

    red = None

    while True:
        print("1. Crear un nuevo perceptrón multicapa")
        print("2. Cargar perceptrón desde un archivo")
        opcion_inicio = input("Seleccione una opción (1/2): ").strip()

        if opcion_inicio == '1':
            u = pedir_entero_positivo("Número de neuronas de entrada (u): ")
            v = pedir_entero_positivo("Número de neuronas de salida/clases (v): ")
            L = pedir_entero_positivo("Número de capas ocultas (L): ")
            b = pedir_entero_positivo("Número de neuronas por capa oculta (b): ")
            red = PerceptronMulticapa(u, v, L, b)
            print("\n✅ Red Neuronal creada exitosamente.")
            break
        elif opcion_inicio == '2':
            ruta = input("Ingrese la ruta del archivo CSV de la red: ").strip()
            if os.path.exists(ruta):
                try:
                    red = cargar_red(ruta)
                    print("\n✅ Red cargada exitosamente.")
                    break
                except Exception as ex:
                    print(f"❌ Error al leer la estructura de la red: {ex}\n")
            else:
                print("❌ El archivo no existe.\n")
        else:
            print("❌ Opción inválida.\n")

    imprimir_hiperparametros(red)

    while True:
        print("MENÚ DE OPCIONES:")
        print("1. Entrenar la red")
        print("2. Probar la red")
        print("3. Guardar la red")
        print("4. Salir")

        opcion = input("Seleccione una opción (1-4): ").strip()

        if opcion == '1':
            ruta_train = input("Ruta del archivo de entrenamiento: ").strip()
            if not os.path.exists(ruta_train):
                print("❌ El archivo no existe.\n")
                continue

            try:
                X, y = cargar_dataset(ruta_train)
            except Exception as ex:
                print(f"❌ Error al cargar el archivo: {ex}\n")
                continue

            if X.shape[1] != red.u:
                print(f"❌ ERROR CONFIGURACIÓN: El archivo tiene {X.shape[1]} entradas, pero la red requiere u={red.u}.\n")
                continue

            epocas = pedir_entero_positivo("Número de épocas a entrenar: ")
            
            print("\nEntrenando red...")
            historial = red.train(X, y, epocas)
            print("¡Entrenamiento finalizado!")

            imprimir_hiperparametros(red)
            y_pred = red.predict(X)
            mostrar_graficos(X, y, y_pred, es_entrenamiento=True, historial_error=historial)

        elif opcion == '2':
            ruta_test = input("Ruta del archivo de prueba: ").strip()
            if not os.path.exists(ruta_test):
                print("❌ El archivo no existe.\n")
                continue

            try:
                X, y = cargar_dataset(ruta_test)
            except Exception as ex:
                print(f"❌ Error al cargar el archivo: {ex}\n")
                continue

            if X.shape[1] != red.u:
                print(f"❌ ERROR CONFIGURACIÓN: El archivo tiene {X.shape[1]} entradas, pero la red requiere u={red.u}.\n")
                continue

            y_pred = red.predict(X)

            imprimir_hiperparametros(red)
            mostrar_graficos(X, y, y_pred, es_entrenamiento=False, num_clases=red.v)

        elif opcion == '3':
            ruta_save = input("Ruta/Nombre donde guardar la red: ").strip()
            try:
                guardar_red(red, ruta_save)
                print(f"✅ Red guardada exitosamente en '{ruta_save}'.\n")
            except Exception as ex:
                print(f"❌ Error al guardar la red: {ex}\n")

        elif opcion == '4':
            print("¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida.\n")

if __name__ == "__main__":
    main()