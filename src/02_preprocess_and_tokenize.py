import os
from transformers import AutoTokenizer
from datasets import load_dataset

# Vamos a probar con un modelo pequeño
MODEL_CHECKPOINTS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
} 

# Codigo de idioma para cada modelo
LANG_CODES = {
    "nllb": {"src": "spa_Latn", "tgt": "quy_Latn"},
    "mbart": {"src": "es_XX", "tgt": "en_XX"} # mbart no tiene codigo de idioma para quechua, por lo que debe de aprender el idioma desde cero a traves del finetuning
}

DATASET_NAME = "somosnlp-hackathon-2022/spanish-to-quechua"
MAX_LENGTH = 128 # Longitud maxima de los tokens

def preprocess_function(examples, tokenizer, model_type):
    """
    Función que se aplica a cada lote de ejemplos del dataset para tokenizarlos
    """

    src_lang_code = LANG_CODES[model_type]["src"]
    tgt_lang_code = LANG_CODES[model_type]["tgt"]

    # Preparar las entradas y salidas para la tokenización
    inputs = [ex for ex in examples["es"]]
    targets = [ex for ex in examples["qu"]]

    # Tokenizamos segun el modelo seleccionado
    # if model_type == "nllb":
    #     tokenizer.src_lang = src_lang_code
    #     model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)

    #     tokenizer.tgt_lang = tgt_lang_code
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    
    # elif model_type == "mbart":
    #     model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        
    #     tokenizer.tgt_lang = tgt_lang_code
    #     labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)

    tokenizer.src_lang = src_lang_code
    
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)

    tokenizer.tgt_lang = tgt_lang_code

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def main():
    """
    Función principal que ejecuta el pipeline de tokenización de cada modelo
    """
    print("Cargando el dataset...")
    dataset = load_dataset(DATASET_NAME)

    for model_key, model_checkpoint in MODEL_CHECKPOINTS.items():
        print(f"\n\tProcesando para el modelo: {model_key.upper()} : {model_checkpoint}")

        # Cargamos el tokenizador
        # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        if model_key == "mbart":
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        print("Tokenizador Cargado")

        # Se aplica la función de preprocesamiento a todo el dataset
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, model_key),
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        # Guardamos el dataset tokenizado
        output_dir = os.path.join("..", "data", f"tokenized_{model_key}")
        tokenized_dataset.save_to_disk(output_dir)
        print(f"Dataset tokenizado para el modelo {model_key} guardado en {output_dir}")

if __name__ == "__main__":
    main()
