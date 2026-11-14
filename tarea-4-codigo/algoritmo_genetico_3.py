# Nombre del integrante: Jonathan Sarli
# Cédula del integrante: 30496924

# haga su tarea aqui

import importlib.util
import json
import os
import random
import statistics

BITS_PER_VAR = 24
BASE_MUTATION_RATE = 0.04


def load_function_module(filename):
    """Dynamically loads a function contract module from relative path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Function file not found: {filename}")

    spec = importlib.util.spec_from_file_location("target_func", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "BOUNDS") or not hasattr(module, "VARIABLE_COUNT") or not hasattr(module, "fitness"):
        raise AttributeError(f"Module {filename} does not follow the required function contract.")
    return module


def encode_variables(variables, bounds):
    """Encodes real-valued variables (phenotype) into a binary string (genotype)."""
    low, high = bounds
    max_int = (1 << BITS_PER_VAR) - 1
    bitstring = ""
    for val in variables:
        clamped = max(low, min(high, val))
        norm = (clamped - low) / (high - low)
        int_val = int(round(norm * max_int))
        bitstring += format(int_val, f"0{BITS_PER_VAR}b")
    return bitstring


def decode_genes(bitstring, var_count, bounds):
    """Decodes a binary string (genotype) into real-valued variables (phenotype)."""
    low, high = bounds
    max_int = (1 << BITS_PER_VAR) - 1
    variables = []
    for i in range(var_count):
        chunk = bitstring[i * BITS_PER_VAR : (i + 1) * BITS_PER_VAR]
        int_val = int(chunk, 2)
        real_val = low + (int_val / max_int) * (high - low)
        variables.append(real_val)
    return variables


def evaluate_member(genes, func_module):
    """Decodes genes and computes fitness using the target function module."""
    variables = decode_genes(genes, func_module.VARIABLE_COUNT, func_module.BOUNDS)
    fit = float(func_module.fitness(*variables))
    return {
        "variables": variables,
        "fitness": fit,
        "genes": genes,
    }


def create_random_population(pop_size, func_module):
    """Generates an initial random population."""
    total_bits = BITS_PER_VAR * func_module.VARIABLE_COUNT
    members = []
    for _ in range(pop_size):
        genes = "".join(random.choice("01") for _ in range(total_bits))
        members.append(evaluate_member(genes, func_module))
    members.sort(key=lambda m: m["fitness"], reverse=True)
    return members


def tournament_selection(members, k=3):
    """Selects one parent using tournament selection."""
    candidates = random.sample(members, min(k, len(members)))
    return max(candidates, key=lambda m: m["fitness"])


def crossover(genes_a, genes_b):
    """Single-point crossover between two binary chromosomes."""
    if len(genes_a) <= 1:
        return genes_a, genes_b
    point = random.randint(1, len(genes_a) - 1)
    child_1 = genes_a[:point] + genes_b[point:]
    child_2 = genes_b[:point] + genes_a[point:]
    return child_1, child_2


def mutate(genes, mutation_variability):
    """Bit-flip mutation scaled by mutation_variability."""
    prob = min(0.5, max(0.001, BASE_MUTATION_RATE * mutation_variability))
    mutated = [
        ("1" if bit == "0" else "0") if random.random() < prob else bit
        for bit in genes
    ]
    return "".join(mutated)


def run_evolution(members, params, func_module):
    """Runs the genetic algorithm loop and prints generation stats."""
    pop_size = params["population_size"]
    threshold = params["difference_threshold"]
    max_gens = params["max_generations"]
    mut_var = params["mutation_variability"]
    start_gen = params["generations_elapsed"]

    members.sort(key=lambda m: m["fitness"], reverse=True)

    for step in range(1, max_gens + 1):
        current_gen = start_gen + step
        # Elitism: preserve the best individual
        next_gen = [members[0]]

        while len(next_gen) < pop_size:
            parent_1 = tournament_selection(members)
            parent_2 = tournament_selection(members)
            c1_genes, c2_genes = crossover(parent_1["genes"], parent_2["genes"])
            c1_genes = mutate(c1_genes, mut_var)
            next_gen.append(evaluate_member(c1_genes, func_module))

            if len(next_gen) < pop_size:
                c2_genes = mutate(c2_genes, mut_var)
                next_gen.append(evaluate_member(c2_genes, func_module))

        members = next_gen
        members.sort(key=lambda m: m["fitness"], reverse=True)
        params["generations_elapsed"] = current_gen

        fitness_values = [m["fitness"] for m in members]
        max_fit = fitness_values[0]
        med_fit = statistics.median(fitness_values)
        min_fit = fitness_values[-1]

        print(
            f"[LOG] Gen {current_gen} | Max: {max_fit:.6f} | "
            f"Median: {med_fit:.6f} | Min: {min_fit:.6f}"
        )

        if (max_fit - min_fit) <= threshold:
            print(
                f"[LOG] Convergence threshold ({threshold}) reached at generation {current_gen}."
            )
            break

    return members


def display_summary(func_name, params, members):
    """Displays all parameters and sorted population members."""
    print("\n" + "=" * 65)
    print("RESUMEN DE LA POBLACIÓN")
    print("=" * 65)
    print(f"Archivo de función         : {func_name}")
    print(f"Tamaño de población        : {params['population_size']}")
    print(f"Umbral de diferencia       : {params['difference_threshold']}")
    print(f"Generaciones máximas/lote  : {params['max_generations']}")
    print(f"Generaciones transcurridas : {params['generations_elapsed']}")
    print(f"Variabilidad de mutación   : {params['mutation_variability']}")
    print("-" * 65)
    print(f"{'#':<4} {'Variables (Fenotipo)':<30} {'Fitness':<15}")
    print("-" * 65)
    for idx, m in enumerate(members, 1):
        vars_str = ", ".join(f"{v:.6f}" for v in m["variables"])
        print(f"{idx:<4} [{vars_str:<28}] {m['fitness']:<15.6f}")
    print("=" * 65 + "\n")


def save_population(filepath, func_name, params, members):
    """Saves population state to JSON adhering to the assignment contract."""
    data = {
        "function": func_name,
        "parameters": params,
        "members": members,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[LOG] Population saved successfully to {os.path.basename(filepath)}")


def load_population(filepath):
    """Loads population state from JSON and reconstructs genes if missing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {os.path.basename(filepath)}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not all(k in data for k in ("function", "parameters", "members")):
        raise KeyError("Invalid JSON schema: missing required top-level keys.")

    func_name = data["function"]
    params = data["parameters"]
    members = data["members"]
    func_module = load_function_module(func_name)

    for m in members:
        if "genes" not in m or not m["genes"]:
            m["genes"] = encode_variables(m["variables"], func_module.BOUNDS)
        m["fitness"] = float(func_module.fitness(*m["variables"]))

    members.sort(key=lambda m: m["fitness"], reverse=True)
    return func_name, params, members, func_module


def prompt_int(msg, default, min_val=1):
    """Prompts for an integer >= min_val, repeating on invalid input."""
    while True:
        raw = input(f"{msg} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if val < min_val:
                print(f"Opción no válida: el número debe ser mayor o igual a {min_val}.")
                continue
            return val
        except ValueError:
            print("Opción no válida: debe ingresar un número entero.")


def prompt_float(msg, default, min_val=0.0, strict_greater=False):
    """Prompts for a float >= min_val (or > min_val), repeating on invalid input."""
    while True:
        raw = input(f"{msg} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if strict_greater and val <= min_val:
                print(f"Opción no válida: el valor debe ser estrictamente mayor a {min_val}.")
                continue
            if not strict_greater and val < min_val:
                print(f"Opción no válida: no se permiten números negativos (mínimo {min_val}).")
                continue
            return val
        except ValueError:
            print("Opción no válida: debe ingresar un número válido.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    func_name = None
    params = None
    members = None
    func_module = None

    while True:
        print("\n--- ALGORITMO GENÉTICO (ESTUDIANTE 3) ---")
        print("1. Crear y entrenar una nueva población")
        print("2. Cargar población desde un archivo JSON")
        print("3. Continuar entrenando población actual")
        print("4. Guardar población actual en un archivo JSON")
        print("5. Salir")
        choice = input("Seleccione una opción: ").strip()

        if choice == "1":
            fn = input("Nombre del archivo de función [funcion_a_3.py]: ").strip()
            func_name = fn if fn else "funcion_a_3.py"
            if not func_name.endswith(".py"):
                func_name += ".py"
            try:
                func_module = load_function_module(func_name)
            except Exception as e:
                print(f"Opción no válida: {e}")
                continue

            pop_size = prompt_int("Tamaño de la población", 10, min_val=2)
            threshold = prompt_float("Umbral de diferencia", 0.05, min_val=0.0)
            max_gens = prompt_int("Número máximo de generaciones", 500, min_val=1)
            mut_var = prompt_float("Variabilidad de la mutación", 1.0, min_val=0.0, strict_greater=True)

            params = {
                "population_size": pop_size,
                "difference_threshold": threshold,
                "max_generations": max_gens,
                "mutation_variability": mut_var,
                "generations_elapsed": 0,
            }
            members = create_random_population(pop_size, func_module)
            members = run_evolution(members, params, func_module)
            display_summary(func_name, params, members)

        elif choice == "2":
            fname = input("Nombre del archivo JSON a cargar: ").strip()
            if not fname:
                print("Opción no válida: debe ingresar el nombre del archivo.")
                continue
            if not fname.endswith(".json"):
                fname += ".json"
            filepath = os.path.join(base_dir, fname)
            try:
                func_name, params, members, func_module = load_population(filepath)
                display_summary(func_name, params, members)
            except Exception as e:
                print(f"Error al cargar el archivo: {e}")

        elif choice == "3":
            if members is None:
                print("Opción no válida: primero debe crear o cargar una población.")
                continue
            extra_gens = prompt_int(
                "Generaciones adicionales a entrenar", params["max_generations"], min_val=1
            )
            params["max_generations"] = extra_gens
            members = run_evolution(members, params, func_module)
            display_summary(func_name, params, members)

        elif choice == "4":
            if members is None:
                print("Opción no válida: no hay ninguna población en memoria para guardar.")
                continue
            fname = input("Nombre del archivo JSON para guardar: ").strip()
            if not fname:
                print("Opción no válida: debe ingresar un nombre válido para el archivo.")
                continue
            if not fname.endswith(".json"):
                fname += ".json"
            filepath = os.path.join(base_dir, fname)
            save_population(filepath, func_name, params, members)

        elif choice == "5":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, seleccione un número del 1 al 5.")


if __name__ == "__main__":
    main()