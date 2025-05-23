from src.train_bert_model import BertModel

bert_variants = [
    "neuralmind/bert-base-portuguese-cased",
]

for model_name in bert_variants:
    for category in ['food','moda','livros','celular','pets','baby','games','laptops','toys','auto']:
        print(f"Training model: {model_name} by product category: {category}")
        bert_model = BertModel(
            model_name=model_name,
            rq="RQ2",
            category=category
        )
        bert_model.train()
        print(f"Model {model_name}_{category} trained and saved.")