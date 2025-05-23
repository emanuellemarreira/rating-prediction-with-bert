import pandas as pd
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate
import optuna
import time
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.special import softmax


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
DIR_TRAIN = os.path.join(DATASET_DIR, 'train.csv')
DIR_TEST = os.path.join(DATASET_DIR, 'test.csv')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'code') 


class BertModel:
    def __init__(self, 
                 model_name, 
                 rq,
                 category = 'all',
                 dir_train = DIR_TRAIN, 
                 dir_test = DIR_TEST, 
                 save_path = SAVE_PATH
                 ):
        self.model_name = model_name
        self.rq = rq
        self.category = category
        self.dir_train = dir_train
        self.dir_test = dir_test
        self.save_path = save_path
        self.output_dir = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.tokenized_train = None
        self.tokenized_test = None
        self.train_dataset = None
        self.test_dataset = None
        self.labels = None
        self.label2id = None
        self.id2label = None
        self.model = None
    
    def preprocess_function(self, examples):
        textos = [str(t) if t is not None else "" for t in examples['text']]
        return self.tokenizer(textos, truncation=True, padding='max_length', max_length=512)
    
    def load_data(self):
        df_train = pd.read_csv(self.dir_train)
        df_test = pd.read_csv(self.dir_test)
        if self.category != 'all':
            df_train.drop(df_train[df_train["categoria"] != self.category].index, inplace=True)
            df_test.drop(df_test[df_test["categoria"] != self.category].index, inplace=True)
        self.train_dataset = Dataset.from_pandas(df_train)
        self.test_dataset = Dataset.from_pandas(df_test)
        self.tokenized_train = self.train_dataset.map(self.preprocess_function, batched=True)
        self.tokenized_test = self.test_dataset.map(self.preprocess_function, batched=True)
        self.labels = sorted(df_train['label'].unique().tolist())
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def compute_metrics(self, eval_pred):
        auc_score = evaluate.load("roc_auc")
        f1_macro = evaluate.load("f1")
        mae = evaluate.load("mae")

        predictions, labels = eval_pred
        probabilities = softmax(predictions, axis=1)
        binary_labels = np.where(np.isin(labels, [3, 4]), 1, 0)
        auc = auc_score.compute(prediction_scores=probabilities[:, [3,4]].max(axis=1), references=binary_labels)['roc_auc']

        f1 = f1_macro.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="macro")['f1']
        
        mse_score = np.mean((np.argmax(predictions, axis=1) - labels)**2)
        rmse_score = np.sqrt(mse_score)

        mae_score = mae.compute(predictions=np.argmax(predictions, axis=1), references=labels)['mae']
        return {"auc": round(auc, 5), "f1": round(f1, 5), "rmse": round(rmse_score, 5), "mae": round(mae_score, 5)}
    
    def objective(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        num_epochs = trial.suggest_categorical("num_train_epochs", [1, 5, 10])
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

        if self.category != 'all':
            os.makedirs(f"{self.save_path}/results/{self.rq}/{self.model_name}_{self.category}_results", exist_ok=True)
            self.output_dir = f"{self.save_path}/results/{self.rq}/{self.model_name}_{self.category}_results"
        else:
            os.makedirs(f"{self.save_path}/results/{self.rq}/{self.model_name}", exist_ok=True)
            self.output_dir = f"{self.save_path}/results/{self.rq}/{self.model_name}"
            
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            save_strategy="no",
            logging_strategy="epoch",
            load_best_model_at_end=False,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        start = time.time()
        trainer.train()
        results = trainer.evaluate()
        horas, resto = divmod(time.time() - start, 3600)
        minutos, segundos = divmod(resto, 60)
        print(f"Tempo: {int(horas)} horas, {int(minutos)} minutos,{segundos:.2f} segundos")
        trainer.save_model(f"{self.output_dir}/trial_{trial.number}")
        return results['eval_rmse']
    
    def train(self):
        self.load_data()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.labels), 
            label2id=self.label2id, 
            id2label=self.id2label
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=5)
        best_trial = study.best_trial
        best_params = best_trial.params
        best_model = AutoModelForSequenceClassification.from_pretrained(
            f"{self.output_dir}/trial_{best_trial.number}",
        )

        trainer = Trainer(
            model=best_model,
            args=TrainingArguments(
                output_dir=f"{self.output_dir}/best_model",
                overwrite_output_dir=True,
                per_device_train_batch_size=best_params['batch_size'],
                per_device_eval_batch_size=best_params['batch_size'],
                num_train_epochs=best_params['num_train_epochs'],
                learning_rate=best_params['learning_rate'],
                weight_decay=best_params['weight_decay'],
                save_strategy="no",
                logging_strategy="epoch",
                load_best_model_at_end=False,
                push_to_hub=False,
            ),
            eval_dataset=self.tokenized_test,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        results = trainer.evaluate()

        predictions = trainer.predict(self.tokenized_test)
        predicted_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        report = classification_report(true_labels, predicted_labels, digits=4)

        cm = confusion_matrix(true_labels, predicted_labels)

        with open(f"{self.output_dir}/best_trial.txt", "a") as f:
            f.write(f"Best trial: {str(best_trial.number)}\n")
            f.write(f"Best params: {str(best_params)}\n")
            f.write(f"Results: {str(results)}\n")
            f.write(f"Classification report: {str(report)}\n")
            f.write(f"Confusion matrix: {str(cm)}\n")

