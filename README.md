# rating-prediction-with-bert

Este projeto aborda o desafio de rating prediction, que consiste em prever automaticamente a nota (rating) que um usuário atribuiria a um produto. Para isso, utiliza-se exclusivamente o texto das avaliações escritas como entrada para variantes do modelo BERT, capazes de capturar nuances linguísticas e contextuais para realizar a previsão com base no conteúdo textual fornecido.

## 📦 Dataset

O dataset conta com +50k de comentários de usuários da Amazon, todos escritos em português brasileiro, com divisão 80% para treino e 20% para teste.

## 🔍 Objetivos de Pesquisa (Research Questions)

Este projeto busca responder duas perguntas de pesquisa:

**RQ1:** Alguma versão do BERT se destaca na tarefa de rating prediction entre diferentes categorias de produtos?

**RQ2:** A melhor configuracão do modelo mantém seu desempenho ao considerar separadamente essas categorias?

O código está organizado por perguntas de pesquisa.

## 💻 Como Usar

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/rating-prediction-with-bert.git
cd rating-prediction-with-bert
```

### 2. Criar o ambiente Conda

```bash
conda env create -f environment.yml
conda activate rating-pred-bert
```
### 3. Rodar os experimentos (por pergunta de pesquisa)
```bash
python run-rq1.py
python run-rq2.py
```

