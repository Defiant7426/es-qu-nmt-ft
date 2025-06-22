import os
import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SRC_DIR)

MODEL_KEY = "mbart"

MODEL_CHECKPOINTS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
}[MODEL_KEY]

TOKENIZED_DATA_PATH = os.path.join("..", "data", f"tokenized_{MODEL_KEY}")
MODEL_OUTPUT_DIR = os.path.join("..", "models", f"{MODEL_KEY}-finetuned-es-qu")

print(f"Cargando dataset tokenizado desde: {TOKENIZED_DATA_PATH}")
tokenized_datasets = load_from_disk(TOKENIZED_DATA_PATH)

if MODEL_KEY == "mbart":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS, use_fast=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS)

print("Datos y tokenizador cargados con éxito.")

bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    """
    Calcula BLEU y chrF++ durante la evaluacion
    """
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decodifica las predicciones
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Reemplaza -100 con el token de padding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Limpieza de las cadenas de texto generadas
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # Calculamos las metricas
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu": bleu_result["score"], "chrf": chrf_result["score"]}

# Se carga el modelo pre-entrenado
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINTS)

# El data collador
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Argumentos de entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    eval_strategy="epoch",  # Evaluar al final de cada época
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Bajar si tienes problemas de memoria
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3, # Guardar solo los 3 mejores checkpoints
    num_train_epochs=2, 
    predict_with_generate=True, # Para evaluar métricas de traducción
    fp16=True, 
    push_to_hub=False, 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Entrenamiento
print("Iniciando entrenamiento...")
trainer.train()

print(f"Modelo guardado en: {MODEL_OUTPUT_DIR}")


