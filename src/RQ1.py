from src.train_bert_model import BertModel

bert_variants = [
    "google-bert/bert-base-multilingual-cased",
    "neuralmind/bert-base-portuguese-cased",
    "distilbert/distilbert-base-multilingual-cased",
    "cservan/multilingual-albert-base-cased-32k",
    "FacebookAI/xlm-roberta-base"
]

for model_name in bert_variants:
    print(f"Training model: {model_name}")
    bert_model = BertModel(
        model_name=model_name,
        rq="RQ1",
        )
    bert_model.train()
    print(f"Model {model_name} trained and saved.")