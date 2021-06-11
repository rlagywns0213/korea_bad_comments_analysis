from flask import Flask, render_template, request
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input, Model
import gensim
import numpy as np
import BahdanauAttention #모델.py 불러오기
from konlpy.tag import Mecab
import pickle
import tensorflow as tf
import re

lstm_model = BahdanauAttention.BahdanauAttention(64)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False #한글 깨짐 현상
wv_model = gensim.models.Word2Vec.load('model/aihub_review_6.model')
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic") #mecab 윈도우에서 설정
tokenizer = pickle.load(open('model/tokenizer.pickle','rb'))

############ 모델 부분
max_len = 100
EMBEDDING_DIM = 100
sequence_input = Input(shape=(max_len,), dtype='int32')

VOCAB_SIZE = len(tokenizer.index_word) + 1
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

# tokenizer에 있는 단어 사전을 순회하면서 word2vec의 100차원 vector를 가져옵니다
for word, idx in tokenizer.word_index.items():
    embedding_vector = wv_model[word] if word in wv_model else None
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector

embedded_sequences = Embedding(VOCAB_SIZE,
                               EMBEDDING_DIM,
                               input_length=max_len,
                               weights=[embedding_matrix],  # weight는 바로 위의 embedding_matrix 대입
                               trainable=False  # embedding layer에 대한 train은 꼭 false로 지정
                               )(sequence_input)

# embedded_sequences = Embedding(vocab_size, 128, input_length=max_len, mask_zero = True)(sequence_input)
lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(embedded_sequences)
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(
    LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)

state_h = Concatenate()([forward_h, backward_h])  # 은닉 상태
state_c = Concatenate()([forward_c, backward_c])  # 셀 상태
attention = lstm_model # 가중치 크기 정의
context_vector, attention_weights = attention(lstm, state_h)

dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation="sigmoid")(dropout)
model = Model(inputs=sequence_input, outputs=output)
model.load_weights('model/best_model.h5')
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']


def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = mecab.morphs(new_sentence) # 토큰화
  new_sentence = [word for word in new_sentence] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len,padding='post') # 패딩
  score = float(model.predict(pad_new)) # 예측
  return round(score, 2)
  # if(score > 0.5):
  #   print("{:.2f}% 확률로 욕설에 가깝습니다.".format(score * 100))
  # else:
  #   print("{:.2f}% 확률로 욕설이 아닙니다.".format((1 - score) * 100))

@app.route('/', methods=['GET','POST'])
def test():
    return render_template('user.html')

@app.route('/post', methods=['GET','POST'])
def post():
    original_test = request.form['test']
    score = sentiment_predict(original_test)

    return render_template('post.html',  score=score)

@app.route('/ajax_model', methods=['GET','POST'])
def ajax_model():
    original_test = request.json['send_data']
    score = sentiment_predict(original_test)
    return str(score*100)

if __name__ == '__main__':
    app.run()