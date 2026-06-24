import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

id2label = {0: 'Доставка', 1: 'Магазин', 2: 'Товар'}

# Путь к модели
MODEL_PATH = "./final_model" 

# можно потянуть с HF
# MODEL_PATH = "jorapro/my-rubert-tiny2-review-classifier" 


tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps' # я на маке сидел с M2
else:
    device = 'cpu'
device = torch.device(device)


model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def get_result(text: pd.Series) -> pd.Series:
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
        "Продавец не обращал внимание пол часа, а после еще и нахамил",
        "Телефон пришел бракованным, экран в трещинах."
    ])
    
    result = get_result(test_data)
    print("Тестовый инференс")
    for i in range(len(result)):
        print('Отзыв:', test_data[i], "|.   Предсказанный класс:", result[i])