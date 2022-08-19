import pandas as pd
from gensim.models.word2vec import Word2Vec
import glob


music_genre_list = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(0,10):

    gn = music_genre_list[i]
    df = pd.read_csv('./Melon/07_clean_gn_concat/{}_fin.csv'.format(gn))
    # df.info()

    # ---- 'Clean lyric' null값 처리 ----
    df = df[df['Clean_lyric'].notna()]
    # df_kr.info()

    clean_kr_lyric = df['Clean_lyric']
    # print('clean_kr_lyric:\n', clean_kr_lyric)

    # ---- 토큰화 ----
    cleaned_kr_lyric = []
    for lyric in clean_kr_lyric:
        token = lyric.split()
        cleaned_kr_lyric.append(token)
    # print('cleaned_kr_lyric:\n', cleaned_kr_lyric)

    # ---- 모델 생성 ----
    # -- 등장 횟수 3회 이하인 단어 제거(min_count) / 40차원(vector_size) / 앞뒤로 읽을 단어 수 20개(window) --
    embedding_model = Word2Vec(cleaned_kr_lyric, min_count=3, vector_size=40, sg=0, batch_words=1000, window=20, workers=8, epochs=100)

    # -- 모델 저장 및 확인 --
    embedding_model.save('./Melon/Models/{}_w2v.model'.format(gn))
    print(list(embedding_model.wv.index_to_key))
    print(len(embedding_model.wv.index_to_key))