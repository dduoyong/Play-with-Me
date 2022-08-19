import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec
from konlpy.tag import Okt
import re

#cosin_sim 유사도 값 10개 찾기, 대신 0은 뺀다 자기자신이므로 simScore = simScore[1:11]
def getRecommandation(cosin_sim):
    simScore = list(enumerate(cosin_sim[-1]))
    simScore = sorted(simScore, key = lambda x:x[1], reverse=True)
    simScore = simScore[:10]
    artistIdx = [i[0] for i in simScore]

    recSongList = df_lyrics.iloc[artistIdx, [0, 1, 2]]

    return recSongList

music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(6, 7):

    gn = music_genre_lst[i]

    df_lyrics = pd.read_csv('./Melon/06_clear_lyric_data/{}_clean_eng_lyric.csv'.format(gn))
    df_lyrics = df_lyrics[df_lyrics['English_clean_lyric'].notna()]
    # print(df_lyrics.head())
    # print(df_lyrics.info())

    Tfidf_matrix = mmread('./Melon/05_models/{}_Tfidf_lyrics.mtx'.format(gn)).tocsr()
    with open('./Melon/05_models/tfidf.pickle', 'rb') as f:
        Tfidf = pickle.load(f)

        embedding_model = Word2Vec.load('./Melon/05_models/{}_word2vec_lyrics.model'.format(gn))
        keyword = 'cafe'
        sim_word = embedding_model.wv.most_similar(keyword, topn=10)

        words = [keyword]
        for word, _ in sim_word:
            words.append(word)
        sentence = []
        count = 10
        for word in words:
            sentence = sentence + [word] * count
            count -= 1


        # 문장 이용해서 추천해주기!
        # okt = Okt()
        # sentence = '화려한 반전과 소름돋는 액션'
        # review = re.sub('[^가-힣 ]', ' ', sentence)  # 가부터 힣까지 그리고 띄어쓰기 빼고 다 띄어쓰기로 대체한다 => 한글과 띄어쓰기만 남기기
        # token = okt.pos(review, stem=True)
        #
        # df_token = pd.DataFrame(token, columns=['word', 'class'])
        # df_token = df_token[(df_token['class'] == 'Noun') |
        #                     (df_token['class'] == 'Verb') |
        #                     (df_token['class'] == 'Adjective')]
        # words = []
        # for word in df_token.word:
        #     if len(word) > 1:
        #         words.append(word)
        # cleaned_sentence = ' '.join(words)
        # print(cleaned_sentence)

        sentence_vec = Tfidf.transform(sentence)
        cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
        recommendation = getRecommandation(cosine_sim)
        print(recommendation)


        df_emo = df_lyrics[['artist', 'title', 'emo', 'track_id']]
        df_emo.dropna(inplace=True)
        print(df_emo.head())
        print(df_emo.info())

        # if df_emo['emo'] == 'Sad' :
        #     print(df_emo[df_emo['emo'] == 'Sad'][:15])


        print(df_emo.loc[df_emo['emo'] == 'Sad'][:15])