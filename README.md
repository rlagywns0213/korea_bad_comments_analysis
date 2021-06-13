# 특정 도메인에서의 데이터 증강 기법을 적용한 악성 댓글 분류 모델의 성능 비교 (Capstone design 2021-1)
* 작성자 : 김효준
* 시연 영상 : www

## Process

The `directory` should look like:

    korea_bad_comments_analysis
    ├── Codes
    ├── Reports
    ├── model
    │   ├── best_model.h5
    │   ├── tokenizer.pickle
    │   └── aihub_review_6.model # memory issue 로 업로드 불가
    │   ├── debug.log
    ├── templates
    │   ├── post.html
    │   └── user.html # user 창
    ├── BahdnauAttention.py # for LSTM attention model
    ├── README.md
    ├── W2V_SR_augmentation.py # 제안한 데이터 증강 기법
    └── comment_confirm.py  # FLASK 구현

## 0. 요약

- 효과적인 텍스트 증강 기법은 다음 그림에서와 같이 EDA : Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks 에서 밝혀진 바 있음
![image](https://user-images.githubusercontent.com/28617444/121798181-d2787080-cc5f-11eb-9adc-ed8f80a4e8c1.png)

  1. SR (Synonym Replacement): 특정 단어를 비슷한 의미의 유의어로 교체
  2. RI (Random Insertion): 임의의 단어를 삽입
  3. RS (Random Swap): 텍스트 내의 두 단어를 임의로 선정하여 서로 위치를 바꿔줌
  4. RD (Random Deletion): 임의의 단어를 삭제


- 그러나, 악성 댓글 Classification 과제에서는 욕설 단어가 가지는 정보를 왜곡하는 RI, RD 는 효과적이지 않은 증강 기법임을 보임
- 추가적으로 [Korea WordNet](http://wordnet.kaist.ac.kr/)에서는 욕설 단어가 존재하지 않으며, 성능 저하를 확인

- 따라서, 악성 댓글 도메인에서의 단어 임베딩을 통해 **도메인 내에서 비슷한 의미의 유의어로 교체해주는 효과적인 데이터 증강 기법 W2V_SR 을 제안하고자 함**

- 추후 악성 댓글뿐만 아니라 특정 도메인 내에서 사용될 수 있는 방법론으로 확장성이 높음

## 1. 과제 개요
가. 과제 선정 배경 및 필요성
- 최근, SNS가 활발해지면서 영화, 쇼핑, 뉴스 등 다양한 산업 분야에서 많은 익명의 리뷰들이 등장
- 익명의 특성으로 인해 단순 욕설, 차별적인 말, 타인 비하 등 불쾌감을 주는 리뷰로 인해 피해받고 있는 실정
- 현재, 네이버 AI 악플 감지기 서비스가 있지만, 반어적인 어투 또는 비꼬는 말 등의 여러 악성 리뷰를 판단하지 못하는 경우가 있기에 더 높은 정확도가 요구되는 모델 필요
- 더구나, 크롤링 기법을 통해 악성 데이터를 수집하는데 있어 진입장벽이 낮지만, 이러한 데이터를 일일이 악성 라벨링 해주는데 들어가는 시간적 비용이 막대한 실정
- 기존 논문에 의하면, 모델의 성능 비교에 그치는 반면, 본 과제는 더 복잡한 신경망 구조의 모델을 활용하여 준지도 학습을 도입하고자 함

나. 과제 주요내용

- 데이터셋 선정
      1) Korean HateSpeech Dataset
      2) AI 허브 인공지능 윤리 연구를 위한 비정형 텍스트 데이터셋

## 2. 분석 방향

1. 텍스트 전처리
 - 불용어 제거
 - 형태소 분석 (KoNLPy의 mecab 라이브러리 사용)
 - 2가지 워드 임베딩

    ⓵ TF-IDF를 통한 워드임베딩 ⓶ Word2Vec 모델을 통한 워드임베딩

2.  데이터 증강기법
 - 기존 텍스트 증강기법인 EDA : Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks 의 SR을 응용
 - 한글 SR 기법인 Korean WordNet (KWN) 과 특정 도메인에서 학습시킨 Word2vec 모델을 활용한 W2V_SR 기법의 성능 비교
 - 악플 데이터셋의 불균형한 label 분포
    - 0 : 5091,     1 : 3253
 - 이를 해결해주기 위해 Train dataset에서 악플인 데이터만 증강하였음

3. 모델링
 - BiLSTM, CNN + BiLSTM, BiLSTM + Attention 3가지 예측 모델 구축

## 3. 주요 분석 결과

*** 다음 도출된 성능은 3번 test accuracy를 도출한 후, 평균낸 값*

### 1. 모델별 예측 성능 비교
| **Model** | **Test Accuracy**
| ----------- | ------------ |
| LSTM    | 0.845         |
| Bi-LSTM    | 0.853         |
| 1D-CNN | 0.837   |
| Bi-LSTM+Attention | 0.866   |

### 2. 데이터 증강 기법 성능 비교

#### Korean WordNet (KWN) 중 SR 기법
| **Model** | **Test Accuracy**
| ----------- | ------------ |
| LSTM    | 0.744         |
| Bi-LSTM    | 0.732         |
| 1D-CNN | 0.743   |
| Bi-LSTM+Attention | 0.772   |

  - SR 기법뿐만 아니라, 논문에서 제시된 EDA(SR, RI, RS, RD) 를 모두 사용하여 데이터를 증강한 경우 모든 모델 성능이 오버피팅 현상이 일어남.
  ![image](https://user-images.githubusercontent.com/28617444/121798787-4cf6bf80-cc63-11eb-84f0-43a5df41345b.png)


  이는 악성 댓글 예측 task 에서 중요한 악성 단어를 왜곡하기에 오버피팅이 일어난 것으로 보임

#### W2V_SR_augmentation

| **Model** | **Test Accuracy**
| ----------- | ------------ |
| LSTM    | 0.850         |
| Bi-LSTM    | 0.846         |
| 1D-CNN | 0.834   |
| Bi-LSTM+Attention | 0.869   |

- Word2Vec 모델을 통해 악성 댓글 도메인 내에서 워드 임베딩을 수행한 결과, 특정 단어를 의미 있는 단어로 대체하는 SR 기법에 효과적인 성능 개선을 보임

- 기존 SR 증강 기법은 모델 성능의 저하를 보였으나, 제안한 W2V_SR 기법은 모델 성능 저하 없이 약간의 성능 개선도 보임


## 4. 시연 영상

- W2V_SR 기법을 통해 데이터를 증강한 후, 가장 높은 성능을 보인 모델을 저장하여 웹으로 구현
  - Flask, html, Javascript 사용

#### 사용자 화면
![image](https://user-images.githubusercontent.com/28617444/121799016-a14e6f00-cc64-11eb-913f-44e3a6b97be9.png)



## 참고자료

1. https://wikidocs.net/48920
2. [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/pdf/1901.11196.pdf)
3. [Korean WordNet](http://wordnet.kaist.ac.kr/)

## Author

HyoJun Kim / [blog](http://rlagywns0213.github.io/)
