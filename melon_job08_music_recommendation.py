import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec
from konlpy.tag import Okt
import re

#cosin_sim 유사도 값 10개 찾기, 대신 0은 뺀다 자기자신이므로 simScore = simScore[1:11]
def getRecommandation(cosin_sim):
    #
    # idx = df_lyrics['artist']+df_lyrics['title']
    # sim_scores = list(enumerate(cosin_sim[idx]))
    # sim_scores = sorted(sim_scores, key = lambda x:[1], reverse=True)
    # sim_scores = sim_scores[1:11]
    # song_indices = [idx[0] for idx in sim_scores]
    # recSongList =
    # return

    simScore = list(enumerate(cosin_sim[-1]))
    simScore = sorted(simScore, key = lambda x:x[1], reverse=True)
    simScore = simScore[:10]
    artistIdx = [i[0] for i in simScore]

    recSongList = df_lyrics.iloc[artistIdx, [0, 1]]

    return recSongList

df_lyrics = pd.read_csv('./melon/04_melon_clear_lyric/Pop_clean_eng_lyric.csv')
Tfidf_matrix = mmread('./melon/melon_models/Tfidf_pop_lyrics.mtx').tocsr()
with open('./melon/melon_models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

    embedding_model = Word2Vec.load('./melon/melon_models/word2vec_Pop_lyrics.model')
    keyword = 'exercise'
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