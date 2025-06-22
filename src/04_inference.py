import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_KEY = "mbart"
LAST_CHECKPOINT = "checkpoint-25500" 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "..", "models", f"{MODEL_KEY}-finetuned-es-qu", LAST_CHECKPOINT)

print(f"Cargando el modelo y tokenizador desde: {MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Cargamos el modelo y el tokenizador
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def translate(text):
    """
    Traduce el texto de español a quechua
    """
    # Tokenizamos el texto de entrada
    if MODEL_KEY == "nllb":
        tokenizer.src_lang = "spa_Latn"
        inputs = tokenizer(text, return_tensors="pt")
    
    elif MODEL_KEY == "mbart":
        inputs = tokenizer(f"es_XX {text}", return_tensors="pt")
     
    inputs = {k: v.to(device) for k, v in inputs.items()}

    translated_tokens = model.generate(
        **inputs,
        max_length=128,  
        num_beams=5,     
        early_stopping=True
    )

    # Convertimos los tokens traducidos a texto
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    if MODEL_KEY == "mbart" and translation.startswith("en_XX"):
        translation = translation.replace("en_XX", "").strip()

    return translation

if __name__ == "__main__":
    
    sentences_to_test = [
        "El perro corre en el parque.",
        "La agricultura es muy importante para la economía de nuestra región.",
        "¿Cómo te llamas y de dónde vienes?",
        "Mi hermana mayor está cocinando papas para el almuerzo.",
        "El gobierno anunció nuevas leyes para proteger el medio ambiente.",
        "No estoy seguro si debería comprar el libro rojo o el azul.",
        "María no irá a la fiesta mañana porque tiene que estudiar.",
    ]
    
    print(f"\n--- Iniciando traducciones con el modelo: {MODEL_KEY.upper()} ---")
    
    for sentence in sentences_to_test:
        translation = translate(sentence)
        print(f"\nEspañol: {sentence}")
        print(f"Quechua: {translation}")
