from datasets import load_dataset
import random

def main():
    """
    Función para cargar el dataset y explorarlo
    """

    print("Cargando dataset...")

    try:
        ds = load_dataset("somosnlp-hackathon-2022/spanish-to-quechua")
        print(f"Dataset cargado con éxito: {ds}")

        print("Los 5 ejemplos aleatorios del dataset:")
        ds_train = ds['train']
        
        ramdon_indices = random.sample(range(len(ds_train)), 20)

        print(ds)

        for i in ramdon_indices:
            example = ds_train[i]
            sorce_text = example['es']
            target_text = example['qu']

            print(f"Ejemplo {i}:")
            print(f"Texto Fuente (Español): {sorce_text}")
            print(f"Texto Objetivo (Quechua): {target_text}")
            print("\n")

    except Exception as e:
        print(f"Error al cargar el dataset: {e}")

if __name__ == "__main__":
    main()
        
        
