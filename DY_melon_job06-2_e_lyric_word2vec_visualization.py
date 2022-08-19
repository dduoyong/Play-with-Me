import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
# from matplotlib import font_manager, rc
import matplotlib as mpl

# font_path = './malgun.ttf'
# font_name = font_manager.FontProperties(
#     fname= font_path).get_name()
#
# mpl.rcParams['axes.unicode_minus'] = False
# rc('font', family = font_name)

#------장르 선택하여 Word2Vec------
music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(6, 7):

    gn = music_genre_lst[i]

    #장르별 생성된 Word2Vec 모델 불러오기
    embedding_model = Word2Vec.load('./Melon/05_models/{}_word2vec_lyrics.model'.format(gn))
    key_word = 'love'
    sim_word = embedding_model.wv.most_similar(key_word, topn=10)
    #가장 유사한 단어 10개를 불러오거라(most_similar)
    #백터 공간상에서 거리를 재서
    #sim_word 형태소의 유사도 값이 들어있음
    print(sim_word)


    #2차원의 plot으로 그려보기
    vectors = []
    labels = []

    for label, _ in sim_word:
        labels.append(label)
        vectors.append(embedding_model.wv[label])
    print(vectors[0])
    print(len(vectors[0]))

    df_vectors = pd.DataFrame(vectors)
    print(df_vectors.head())

    #유사단어 10개 뽑아서 차원 축소하는 거까지

    #차원 축소
    #tsne model이 2차원으로 차원 축소해 줌
    #pca 차원 축소 알고리즘
    #n_iter 에폭 값에 해당
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_value = tsne_model.fit_transform(df_vectors)
    df_xy = pd.DataFrame({'words':labels,
                          'x':new_value[:, 0],
                          'y':new_value[:, 1]})

    print(df_xy)
    print(df_xy.shape)

    df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)
    #x=0, y=0, keyword가 중심

    plt.figure(figsize=(8, 8))
    plt.scatter(0, 0, s=1500, marker = '*')

    #11개인데 키워드는 10개만 그리기
    for i in range(len(df_xy) - 1):
        a = df_xy.loc[[i, 10]]
        plt.plot(a.x, a.y, '-D', linewidth=1)
        #좌표 안에 해당 키워드 글자 넣어주기
        plt.annotate(df_xy.words[i], xytext=(1, 1),
                     xy=(df_xy.x[i], df_xy.y[i]),
                     textcoords='offset points',
                     ha= 'right', va='bottom')
        #ha수평정렬, va수직정렬
        #좌표에 offset을 주면 살짝 띄움

    plt. show()