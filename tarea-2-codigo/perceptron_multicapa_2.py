# Gabriel Jimenez
# 32001385

"""Perceptrón multicapa para la Tarea 2.

Requiere:
    pip install numpy matplotlib
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def sigmoide(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Datos:
    entradas: np.ndarray
    etiquetas: np.ndarray
    clases: np.ndarray


class PerceptronMulticapa:
    def __init__(
        self,
        entradas: int,
        salidas: int,
        capas_ocultas: int,
        neuronas_por_capa: int,
        epocas: int = 0,
        pesos: list[np.ndarray] | None = None,
    ) -> None:
        if min(entradas, salidas, neuronas_por_capa) <= 0 or capas_ocultas < 0:
            raise ValueError("Las cantidades de neuronas y capas deben ser positivas.")

        self.u = int(entradas)
        self.v = int(salidas)
        self.L = int(capas_ocultas)
        self.b = int(neuronas_por_capa)
        self.e = int(epocas)
        tamanos = [self.u] + [self.b] * self.L + [self.v]

        if pesos is None:
            generador = np.random.default_rng()
            self.pesos = [
                generador.normal(0, 1 / np.sqrt(tamanos[i] + 1), (tamanos[i + 1], tamanos[i] + 1))
                for i in range(len(tamanos) - 1)
            ]
        else:
            self.pesos = pesos

    def propagacion(self, entradas: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activaciones = [entradas]
        entradas_con_sesgo: list[np.ndarray] = []
        actual = entradas

        for peso in self.pesos:
            con_sesgo = np.concatenate((actual, np.ones((actual.shape[0], 1))), axis=1)
            entradas_con_sesgo.append(con_sesgo)
            actual = sigmoide(con_sesgo @ peso.T)
            activaciones.append(actual)

        return activaciones, entradas_con_sesgo

    def predecir(self, entradas: np.ndarray) -> np.ndarray:
        return self.propagacion(entradas)[0][-1]

    def entrenar(self, entradas: np.ndarray, objetivos: np.ndarray, epocas: int, tasa: float = 0.5) -> list[float]:
        errores: list[float] = []
        for _ in range(epocas):
            activaciones, entradas_con_sesgo = self.propagacion(entradas)
            deltas: list[np.ndarray] = [np.empty(0)] * len(self.pesos)
            salida = activaciones[-1]
            deltas[-1] = (objetivos - salida) * salida * (1 - salida)

            for i in range(len(self.pesos) - 2, -1, -1):
                siguiente = self.pesos[i + 1][:, :-1]
                activacion = activaciones[i + 1]
                deltas[i] = (deltas[i + 1] @ siguiente) * activacion * (1 - activacion)

            for i, delta in enumerate(deltas):
                self.pesos[i] += tasa * delta.T @ entradas_con_sesgo[i] / len(entradas)

            errores.append(float(np.mean((objetivos - salida) ** 2)))

        self.e += epocas
        return errores

    def guardar(self, ruta: str) -> None:
        encabezado = ["tipo", "capa", "posicion"]
        max_pesos = max(peso.shape[1] for peso in self.pesos)
        encabezado.extend(f"w{i}" for i in range(max_pesos))

        with open(ruta, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            for clave, valor in (("u", self.u), ("v", self.v), ("L", self.L), ("b", self.b), ("e", self.e)):
                escritor.writerow([clave, valor] + [""] * max_pesos)
            escritor.writerow(encabezado)

            for capa, pesos_capa in enumerate(self.pesos, start=1):
                tipo = "o" if capa == len(self.pesos) else "h"
                for posicion, pesos_neurona in enumerate(pesos_capa):
                    escritor.writerow([tipo, capa, posicion, *pesos_neurona])

    @classmethod
    def cargar(cls, ruta: str) -> "PerceptronMulticapa":
        with open(ruta, newline="", encoding="utf-8") as archivo:
            filas = list(csv.reader(archivo))

        if len(filas) < 6:
            raise ValueError("El archivo de red no contiene los hiperparámetros requeridos.")
        parametros = {fila[0]: int(fila[1]) for fila in filas[:5]}
        redes: dict[int, list[tuple[int, list[float]]]] = {}
        for fila in filas[6:]:
            if len(fila) < 4:
                continue
            capa = int(fila[1])
            posicion = int(fila[2])
            pesos = [float(valor) for valor in fila[3:] if valor.strip()]
            redes.setdefault(capa, []).append((posicion, pesos))

        pesos: list[np.ndarray] = []
        for capa in sorted(redes):
            neuronas = sorted(redes[capa])
            pesos.append(np.array([fila[1] for fila in neuronas], dtype=float))

        esperadas = parametros["L"] + 1
        if len(pesos) != esperadas:
            raise ValueError(f"Se esperaban {esperadas} capas de pesos y se encontraron {len(pesos)}.")
        return cls(parametros["u"], parametros["v"], parametros["L"], parametros["b"], parametros["e"], pesos)


def leer_datos(ruta: str, entradas_esperadas: int, salidas_esperadas: int) -> Datos:
    with open(ruta, newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        encabezado = next(lector, None)
        filas = list(lector)

    if not encabezado or not filas:
        raise ValueError("El CSV debe tener encabezado y al menos un vector.")
    matriz = np.array([[float(valor) for valor in fila] for fila in filas], dtype=float)
    entradas, etiquetas = matriz[:, :-1], matriz[:, -1].astype(int)
    clases = np.unique(etiquetas)
    if entradas.shape[1] != entradas_esperadas or len(clases) != salidas_esperadas:
        raise ValueError(
            f"El archivo tiene {entradas.shape[1]} entradas y {len(clases)} clases; "
            f"configure la red como u={entradas.shape[1]} y v={len(clases)}."
        )
    objetivos = np.zeros((len(etiquetas), salidas_esperadas))
    mapa = {clase: indice for indice, clase in enumerate(clases)}
    for fila, clase in enumerate(etiquetas):
        objetivos[fila, mapa[clase]] = 1.0
    return Datos(entradas, objetivos, clases)


def matriz_confusion(reales: np.ndarray, predichas: np.ndarray, num_clases: int) -> np.ndarray:
    matriz = np.zeros((num_clases, num_clases), dtype=int)
    for real, predicha in zip(reales, predichas):
        matriz[real, predicha] += 1
    return matriz


def graficar(datos: Datos, predicciones: np.ndarray, titulo: str, errores_epoca: list[float] | None = None) -> None:
    """Muestra los tres gráficos. Si se pasan errores_epoca (entrenamiento) se grafica el
    error por época; si no (prueba), se muestra la matriz de confusión."""
    clases_predichas = np.argmax(predicciones, axis=1)
    clases_reales = np.argmax(datos.etiquetas, axis=1)
    correctas = clases_predichas == clases_reales
    figura = plt.figure(figsize=(16, 5))

    if datos.entradas.shape[1] >= 3:
        eje = figura.add_subplot(131, projection="3d")
        eje.scatter(datos.entradas[:, 0], datos.entradas[:, 1], datos.entradas[:, 2], c=clases_predichas, cmap="tab10")
        eje.set_zlabel("x3")
    else:
        eje = figura.add_subplot(131)
        eje.scatter(datos.entradas[:, 0], datos.entradas[:, 1], c=clases_predichas, cmap="tab10")
    eje.set_title("Datos y clase predicha")
    eje.set_xlabel("x1")
    eje.set_ylabel("x2")

    eje = figura.add_subplot(132)
    eje.bar(np.arange(len(correctas)), correctas.astype(int), color=np.where(correctas, "seagreen", "crimson"))
    eje.set_title(f"Predicción correcta (1) / incorrecta (0) — {np.mean(correctas):.2%}")
    eje.set_xlabel("Vector")
    eje.set_ylim(-0.05, 1.05)

    eje = figura.add_subplot(133)
    if errores_epoca is not None:
        eje.plot(np.arange(1, len(errores_epoca) + 1), errores_epoca, linewidth=1)
        eje.set_title("Error por época")
        eje.set_xlabel("Época")
        eje.set_ylabel("Error cuadrático medio")
    else:
        matriz = matriz_confusion(clases_reales, clases_predichas, len(datos.clases))
        imagen = eje.matshow(matriz, cmap="Blues")
        figura.colorbar(imagen, ax=eje)
        for (fila, columna), valor in np.ndenumerate(matriz):
            eje.text(columna, fila, str(valor), ha="center", va="center",
                     color="white" if valor > matriz.max() / 2 else "black")
        etiquetas = [str(clase) for clase in datos.clases]
        eje.set_xticks(range(len(etiquetas)), etiquetas)
        eje.set_yticks(range(len(etiquetas)), etiquetas)
        eje.xaxis.set_ticks_position("bottom")
        eje.set_title("Matriz de confusión")
        eje.set_xlabel("Clase predicha")
        eje.set_ylabel("Clase real")
    figura.suptitle(titulo)
    figura.tight_layout()
    plt.show()


def imprimir_hiperparametros(red: PerceptronMulticapa) -> None:
    print("\nHiperparámetros:")
    print(f"  u (entradas)          = {red.u}")
    print(f"  v (salidas)           = {red.v}")
    print(f"  L (capas ocultas)     = {red.L}")
    print(f"  b (neuronas por capa) = {red.b}")
    print(f"  e (épocas)            = {red.e}")


def crear_o_cargar() -> PerceptronMulticapa:
    while True:
        print("1. Crear un nuevo perceptrón multicapa")
        print("2. Cargar un perceptrón desde un archivo")
        opcion = input("Seleccione una opción: ").strip()
        try:
            if opcion == "1":
                return PerceptronMulticapa(
                    int(input("Neuronas de entrada: ")),
                    int(input("Neuronas de salida: ")),
                    int(input("Capas ocultas: ")),
                    int(input("Neuronas por capa oculta: ")),
                )
            if opcion == "2":
                return PerceptronMulticapa.cargar(input("Archivo de la red: ").strip())
            print("Opción inválida.\n")
        except (ValueError, OSError, KeyError, IndexError) as error:
            print(f"Error: {error}\n")


def menu() -> None:
    red = crear_o_cargar()

    while True:
        imprimir_hiperparametros(red)
        print("\n1. Entrenar\n2. Probar\n3. Guardar\n4. Salir")
        opcion = input("Seleccione una opción: ").strip()
        try:
            if opcion == "1":
                datos = leer_datos(input("Archivo CSV de entrenamiento: ").strip(), red.u, red.v)
                epocas = int(input("Número de épocas: "))
                if epocas <= 0:
                    raise ValueError("El número de épocas debe ser positivo.")
                errores = red.entrenar(datos.entradas, datos.etiquetas, epocas)
                predicciones = red.predecir(datos.entradas)
                print(f"Error inicial: {errores[0]:.6f} | Error final: {errores[-1]:.6f}")
                graficar(datos, predicciones, "Entrenamiento", errores)
            elif opcion == "2":
                datos = leer_datos(input("Archivo CSV de prueba: ").strip(), red.u, red.v)
                predicciones = red.predecir(datos.entradas)
                exactitud = np.mean(np.argmax(predicciones, axis=1) == np.argmax(datos.etiquetas, axis=1))
                print(f"Exactitud: {exactitud:.2%}")
                graficar(datos, predicciones, "Prueba")
            elif opcion == "3":
                ruta = input("Archivo de salida de la red: ").strip()
                red.guardar(ruta)
                print(f"Red guardada en {os.path.abspath(ruta)}")
            elif opcion == "4":
                break
            else:
                print("Opción inválida.")
        except (ValueError, OSError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    menu()

