import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

id2label = {0: 'Доставка', 1: 'Магазин', 2: 'Товар'}

# Путь к модели
MODEL_PATH = "./final_model" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Определяем устройство (если у проверяющего нет GPU, упадет на CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
model.to(device)
model.eval()

def get_result(text: pd.Series) -> pd.Series:
    """
    Функция принимает колонку с отзывами (pd.Series), 
    предсказывает классы и возвращает их в виде pd.Series с именем 'class_predicted'.
    """
    predictions = []
    texts = text.tolist()
    batch_size = 16
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
        batch_preds = [id2label[pred] for pred in preds]
        predictions.extend(batch_preds)
        
    return pd.Series(predictions, name='class_predicted')

if __name__ == "__main__":
    test_data = pd.Series([
        "Курьер опоздал на 3 часа, очень плохо!",
        "Продавец в магазине был очень грубым и хамил.",
        "Товар оказался бракованным, экран в трещинах."
    ])
    
    result = get_result(test_data)
    print("Тестовый инференс отработал успешно:")
    print(result)