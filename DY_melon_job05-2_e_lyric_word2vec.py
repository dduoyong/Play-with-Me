import pandas as pd
from gensim.models import Word2Vec

music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(6, 7):

    gn = music_genre_lst[i]

    lyric_word = pd.read_csv('./Melon/06_clear_lyric_data/{}_clean_eng_lyric.csv'.format(gn))
    lyric_word.info()
    lyric_word = lyric_word[lyric_word['English_clean_lyric'].notna()]
    lyric_word.info()

    cleaned_token_lyrics = list(lyric_word['English_clean_lyric'])
    print(cleaned_token_lyrics[0])

    cleaned_tokens = []
    for sentence in cleaned_token_lyrics:
        token = sentence.split()
        cleaned_tokens.append(token)

    print(cleaned_tokens[0])

    #원래는 단어의 갯수만큼 차원의 공간이 만들어지는데 그럴 경우 너무 많으니 100차원으로 줄인다/차원의 저주
    embedding_model = Word2Vec(cleaned_tokens, vector_size=80,
                               window = 10, min_count= 1,
                               workers = 8, epochs=100, sg=0)

    #window=4 단어들의 앞뒤관계 즉 문맥을 학습하여 (그냥 Dense Layer로 학습하면 문맥의 의미 반영X)

    #min_count 최소 20번 출현하는 단어들에 대해서만 vectorizing, 출현빈도가 작은 것들은 학습이 안됨

    #worker=4 CPU 몇 개 쓸 것인가

    #sg는 알고리즘, 어떤 알고리즘을 쓸 것인가 default는 1 vector공간상의 batch하는 알고리즘 무엇을 쓸 것인가

    #model저장
    embedding_model.save('./Melon/05_models/{}_word2vec_lyrics.model'.format(gn))
    print(list(embedding_model.wv.index_to_key))
    print(len(embedding_model.wv.index_to_key))
