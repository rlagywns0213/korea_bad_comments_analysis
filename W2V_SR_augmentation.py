import random
import re
import gensim #word2vec 라이브러리

# 한글만 남기고 삭제
def get_only_hangul(line):
    parseText = re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('', line)
    return parseText

def W2V_SR_augmentation(W2V_model="aihub_review_6.model", sentence, p=0.5):

    """
    W2V_model : 학습시킨 Word2Vec 모델( default = 악플 데이터 6백만건을 학습시킨 모델 사용 )
    sentence : 학습시킬 문장
    p : SR 을 진행할 확률 (단어당 확률 )
    """
    wv_model = gensim.models.Word2Vec.load(W2V_model)  # 모델 로드
    r = random.uniform(0, 1)
    if r > p:
        return sentence  # p확률로 그냥 반환
    else:
        new_sentence = ''
        for i in get_only_hangul(sentence).split(' '):
            try:
                new_word = wv_model.wv.similar_by_word(i)[0][0]
                new_sentence += new_word + ' '
            #                     print(wv_model.similar_by_word(i)[0][0])
            except KeyError:
                #                 print(i, "는 단어장에 존재하지 않습니다.")
                new_sentence += i + ' '

    return new_sentence