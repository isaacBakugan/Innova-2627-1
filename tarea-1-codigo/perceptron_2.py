#Gabriel Jimenez 
#32001385

import matplotlib.pyplot as plt


def cargar_csv(ruta):
    datos = []

    with open(ruta, "r", encoding="utf-8-sig") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if linea == "":
                continue

            partes = linea.split(",")
            try:
                fila = [float(valor) for valor in partes]
            except ValueError:
                continue

            datos.append(fila)

    return datos


def suma(entradas, pesos, sesgo):
    total = sesgo
    for i in range(len(entradas)):
        total = total + entradas[i] * pesos[i]
    return total


def escalon(valor):
    if valor >= 0:
        return 1
    return 0


def signo(valor):
    if valor >= 0:
        return 1
    return -1


def calcular_predicciones(datos, pesos, sesgo, funcion):
    predichos = []

    for fila in datos:
        entradas = fila[:-1]
        resultado = suma(entradas, pesos, sesgo)
        predichos.append(funcion(resultado))

    return predichos


def graficar(datos, predichos):
    x = []
    y = []
    esperados = []

    for fila in datos:
        x.append(fila[0])
        if len(fila) > 2:
            y.append(fila[1])
        else:
            y.append(0)
        esperados.append(fila[-1])

    colores = []
    for i in range(len(esperados)):
        if esperados[i] == predichos[i]:
            colores.append("green")
        else:
            colores.append("red")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(x, y, c=esperados)
    plt.title("Valor esperado")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(1, 3, 2)
    plt.scatter(x, y, c=predichos)
    plt.title("Valor predicho")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(1, 3, 3)
    plt.scatter(x, y, c=colores)
    plt.title("Coincidencia")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.tight_layout()
    plt.show()


def pedir_activacion():
    print("Funciones de activacion:")
    print("1. Escalon")
    print("2. Signo")
    opcion = input("Seleccione una opcion: ")

    if opcion == "2":
        return signo
    return escalon


def main():
    print("Perceptron")
    ruta = input("Ingrese la ruta del archivo CSV: ")

    try:
        datos = cargar_csv(ruta)
    except FileNotFoundError:
        print("No se encontro el archivo.")
        return

    if len(datos) == 0:
        print("No se cargaron datos.")
        return

    cantidad_entradas = len(datos[0]) - 1

    repetir = "s"
    while repetir.lower() == "s":
        pesos = []

        sesgo = float(input("Peso del sesgo: "))
        for i in range(cantidad_entradas):
            peso = float(input("Peso w" + str(i + 1) + ": "))
            pesos.append(peso)

        funcion = pedir_activacion()
        predichos = calcular_predicciones(datos, pesos, sesgo, funcion)

        print("Predicciones:", predichos)
        graficar(datos, predichos)

        repetir = input("Desea probar con otros pesos? (s/n): ")


if __name__ == "__main__":
    main()

