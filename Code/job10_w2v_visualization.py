import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl


# # ---- 폰트 지정 ----
# font_path = './malgun.ttf'
# font_name = font_manager.FontProperties(fname= font_path).get_name()
# mpl.rcParams['axes.unicode_minus'] = False
# rc('font', family = font_name)


music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(0, 1):

    gn = music_genre_lst[i]

    # ---- 장르별 생성된 Word2Vec 모델 불러오기 ----
    embedding_model = Word2Vec.load('../Melon/Models/{}_w2v.model'.format(gn))
    # ---- sim word 확인 ----
    key_word = 'love'
    sim_word = embedding_model.wv.most_similar(key_word, topn=10)
    print(sim_word)

    # ---- 2차원의 plot으로 그려보기 ----
    vectors = []
    labels = []

    for label, _ in sim_word:
        labels.append(label)
        vectors.append(embedding_model.wv[label])
    print(vectors[0])
    print(len(vectors[0]))

    df_vectors = pd.DataFrame(vectors)
    print(df_vectors.head())

    # ---- 유사 단어 10개 뽑아서 차원 축소 ----
    # -- tsne model 이 2차원으로 차원 축소(pca 차원 축소 알고리즘) --
    # -- n_iter 에폭 값에 해당 --
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_value = tsne_model.fit_transform(df_vectors)
    df_xy = pd.DataFrame({'words':labels, 'x':new_value[:, 0], 'y':new_value[:, 1]})
    print(df_xy)
    print(df_xy.shape)

    # -- x=0, y=0, keyword가 중심 --
    df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)

    plt.figure(figsize=(8, 8))
    plt.scatter(0, 0, s=1500, marker = '*')

    # ---- 유사 키워드 10개 그리기 ----
    for i in range(len(df_xy) - 1):
        a = df_xy.loc[[i, 10]]
        plt.plot(a.x, a.y, '-D', linewidth=1)
        # -- 좌표 안에 해당 키워드 글자 넣어주기 --
        # -- ha수평정렬, va수직정렬 / 좌표에 offset을 주면 살짝 띄움 --
        plt.annotate(df_xy.words[i], xytext=(1, 1), xy=(df_xy.x[i], df_xy.y[i]),
                     textcoords='offset points', ha= 'right', va='bottom')
    plt. show()