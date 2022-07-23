FROM python:3.10

WORKDIR /app


COPY ./requirements.txt /app
RUN pip install --upgrade -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download sv_core_news_sm
RUN python -c 'from transformers import BertTokenizerFast, BertModel; BertModel.from_pretrained("bert-base-cased"); BertTokenizerFast.from_pretrained("bert-base-cased")' 

COPY . /app

CMD ["python", "main.py", "dataset=dwug_es", "model=bert"]