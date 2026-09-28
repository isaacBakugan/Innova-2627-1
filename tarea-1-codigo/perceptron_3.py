# Nombre del integrante: Jonathan Sarli
# Cédula del integrante: 30496924

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# LECTURA DEL CSV
def cargar_datos_csv(ruta_archivo):
    entradas = []
    esperados = []
    
    with open(ruta_archivo, mode='r', encoding='utf-8-sig') as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:  # Saltar líneas vacías
                continue
            partes = linea.split(',')
            
            # Intentar convertir a números (si es encabezado tipo 'x1,x2,y', lo ignora)
            try:
                valores = [float(val) for val in partes]
                entradas.append(valores[:-1])
                esperados.append(valores[-1])
            except ValueError:
                continue  # Salta la fila si contiene texto (ej. encabezado)
            
    return entradas, esperados


# MATEMÁTICA DEL PERCEPTRÓN
def calcular_suma_ponderada(x, pesos, sesgo):
    suma = sesgo
    for i in range(len(x)):
        suma += x[i] * pesos[i]
    return suma

def funcion_escalon(z):
    return 1.0 if z >= 0 else 0.0

def funcion_signo(z):
    return 1.0 if z >= 0 else -1.0

def ejecutar_perceptron(entradas, pesos, sesgo, funcion_activacion):
    predicciones = []
    for x in entradas:
        z = calcular_suma_ponderada(x, pesos, sesgo)
        pred = funcion_activacion(z)
        predicciones.append(pred)
    return predicciones


# GENERACIÓN DE GRÁFICOS
# GENERACIÓN DE GRÁFICOS
def generar_graficos(entradas, esperados, predichos):
    x_coords = [fila[0] for fila in entradas]
    y_coords = [fila[1] if len(fila) > 1 else 0 for fila in entradas]

    # Asignación explícita de colores sólidos (Rojo = 1.0, Azul = 0.0 o -1.0)
    colores_esperados = ['firebrick' if e == 1.0 else 'navy' for e in esperados]
    colores_predichos = ['firebrick' if p == 1.0 else 'navy' for p in predichos]
    colores_coincidencia = ['green' if e == p else 'red' for e, p in zip(esperados, predichos)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Elementos para las leyendas
    patch_rojo = mpatches.Patch(color='firebrick', label='Clase 1 (1.0)')
    patch_azul = mpatches.Patch(color='navy', label='Clase 0 / -1')
    patch_verde = mpatches.Patch(color='green', label='Acierto')
    patch_error = mpatches.Patch(color='red', label='Error')

    # 1. Valores Esperados
    axes[0].scatter(x_coords, y_coords, c=colores_esperados, edgecolors='k', s=100)
    axes[0].set_title("1. Valores Esperados")
    axes[0].set_xlabel("Dimensión 1 (x1)")
    axes[0].set_ylabel("Dimensión 2 (x2)")
    axes[0].legend(handles=[patch_rojo, patch_azul], bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, fontsize='small')

    # 2. Valores Predichos
    axes[1].scatter(x_coords, y_coords, c=colores_predichos, edgecolors='k', s=100)
    axes[1].set_title("2. Valores Predichos")
    axes[1].set_xlabel("Dimensión 1 (x1)")
    axes[1].set_ylabel("Dimensión 2 (x2)")
    axes[1].legend(handles=[patch_rojo, patch_azul], bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, fontsize='small')

    # 3. Coincidencias
    axes[2].scatter(x_coords, y_coords, c=colores_coincidencia, edgecolors='k', s=100)
    axes[2].set_title("3. Coincidencias")
    axes[2].set_xlabel("Dimensión 1 (x1)")
    axes[2].set_ylabel("Dimensión 2 (x2)")
    axes[2].legend(handles=[patch_verde, patch_error], bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, fontsize='small')

    plt.tight_layout()
    plt.show()

# FUNCIONES AUXILIARES DE VALIDACIÓN

def pedir_float(mensaje):
    """Garantiza que el usuario ingrese un número entero o decimal válido."""
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("  ❌ Valor inválido. Por favor, ingrese un número (ejemplo: 1.5, -2, 0).")

def pedir_respuesta_sn(mensaje):
    """Garantiza que el usuario responda únicamente 's' o 'n'."""
    while True:
        resp = input(mensaje).strip().lower()
        if resp in ['s', 'n']:
            return resp
        print("  ❌ Valor inválido. Por favor, ingrese únicamente 's' (sí) o 'n' (no).")

def pedir_ruta_csv():
    """Pide la ruta del archivo CSV y valida que realmente se pueda abrir."""
    while True:
        ruta = input("\nIngrese el nombre o ruta del archivo CSV (ejemplo: datos.csv): ").strip()
        try:
            entradas, esperados = cargar_datos_csv(ruta)
            num_entradas = len(entradas[0])
            print(f"\n¡Archivo cargado con éxito! Se detectaron {len(entradas)} filas y {num_entradas} entradas por fila.")
            return entradas, esperados, num_entradas
        except FileNotFoundError:
            print(f"  ❌ Error: No se encontró el archivo '{ruta}'. Verifique la ruta e intente de nuevo.")
        except Exception as e:
            print(f"  ❌ Error al leer el archivo CSV: {e}. Verifique el formato.")


# INTERFAZ DE USUARIO
def menu_principal():
    print("=== PERCEPTRÓN SIMPLE DE UNA CAPA ===")
    
    # 1. Cargar archivo CSV validado
    entradas, esperados, num_entradas = pedir_ruta_csv()

    # 2. Selección validada de la Función de Activación
    while True:
        print("\nSeleccione la función de activación:")
        print("1. Escalón Unitario (Retorna 0 o 1)")
        print("2. Signo (Retorna -1 o 1)")
        opcion = input("Ingrese 1 o 2: ").strip()

        if opcion == "1":
            f_activacion = funcion_escalon
            nombre_func = "Escalón Unitario"
            break
        elif opcion == "2":
            f_activacion = funcion_signo
            nombre_func = "Signo"
            break
        else:
            print("  ❌ Opción inválida. Ingrese solo el número 1 o 2.")

    # 3. Bucle para probar parámetros
    while True:
        print(f"\n--- Ingrese los parámetros manualmente (Función: {nombre_func}) ---")
        
        pesos = []
        for i in range(num_entradas):
            peso = pedir_float(f"Ingrese el peso w{i+1}: ")
            pesos.append(peso)
            
        sesgo = pedir_float("Ingrese el valor del sesgo/bias (w0): ")

        # Calcular predicciones
        predicciones = ejecutar_perceptron(entradas, pesos, sesgo, f_activacion)

        print("\n--- RESULTADOS ---")
        print("Esperados: ", esperados)
        print("Predichos: ", predicciones)

        # Mostrar gráficos
        print("\nMostrando gráficos...")
        generar_graficos(entradas, esperados, predicciones)

        # Preguntar si desea intentar con otros pesos
        repetir = pedir_respuesta_sn("\n¿Desea probar con otros pesos/sesgo en este mismo archivo? (s/n): ")
        if repetir != 's':
            # Preguntar si desea cambiar de archivo o salir definitivamente
            cambiar_archivo = pedir_respuesta_sn("¿Desea probar con OTRO archivo CSV o reiniciar la configuración? (s/n): ")
            if cambiar_archivo == 's':
                return menu_principal()  # Reinicia todo el menú desde el inicio
            else:
                print("\n¡Programa finalizado!")
                break

if __name__ == "__main__":
    menu_principal()