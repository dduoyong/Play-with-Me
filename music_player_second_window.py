import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import pandas as pd
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QStringListModel
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
import webbrowser
import urllib.request


emo_window = uic.loadUiType('./music_player_2.ui')[0]

class secondwindow(QDialog, QWidget, emo_window):
    def __init__(self, main_window):
        super().__init__()
        self.setupUi(self)

        self.main_window = main_window

        # ---- 이미지 설정 ----
        # 프로필 이미지
        self.lbl_profile.setPixmap(QtGui.QPixmap('./PyQt_imgs/profile.png'))
        # 감정 라벨 이미지
        self.lbl_energetic.setPixmap(QtGui.QPixmap('./PyQt_imgs/energetic.png'))
        self.lbl_happy.setPixmap(QtGui.QPixmap('./PyQt_imgs/happy.png'))
        self.lbl_comfortable.setPixmap(QtGui.QPixmap('./PyQt_imgs/comfortable.png'))
        self.lbl_chilling.setPixmap(QtGui.QPixmap('./PyQt_imgs/chilling.png'))
        self.lbl_depressed.setPixmap(QtGui.QPixmap('./PyQt_imgs/depressed.png'))
        self.lbl_sad.setPixmap(QtGui.QPixmap('./PyQt_imgs/sad.png'))
        # 앨범아트 이미지
        self.lbl_album_art.setPixmap(QtGui.QPixmap('./PyQt_imgs/album_art.png'))
        # 플레이어 아이콘
        self.btn_play.setIcon(QtGui.QIcon('./PyQt_imgs/play.png'))
        self.btn_next.setIcon(QtGui.QIcon('./PyQt_imgs/next.png'))
        self.btn_previous.setIcon(QtGui.QIcon('./PyQt_imgs/previous.png'))

        # ---- 버튼 설정 ----
        # 홈버튼
        self.btn_home.clicked.connect(self.home)
        # print(main_window.selected)

        # 감정 버튼
        self.btn_energetic.clicked.connect(self.energetic_reco)
        self.btn_happy.clicked.connect(self.happy_reco)
        self.btn_comfortable.clicked.connect(self.comfortable_reco)
        self.btn_chilling.clicked.connect(self.chilling_reco)
        self.btn_depressed.clicked.connect(self.depressed_reco)
        self.btn_sad.clicked.connect(self.sad_reco)

        # 플레이리스트 버튼
        self.buttons = [self.btn_title_1, self.btn_title_2, self.btn_title_3, self.btn_title_4,
                        self.btn_title_5, self.btn_title_6, self.btn_title_7]

        self.btn_title_1.clicked.connect(self.show_album_art)
        self.btn_title_2.clicked.connect(self.show_album_art)
        self.btn_title_3.clicked.connect(self.show_album_art)
        self.btn_title_4.clicked.connect(self.show_album_art)
        self.btn_title_5.clicked.connect(self.show_album_art)
        self.btn_title_6.clicked.connect(self.show_album_art)
        self.btn_title_7.clicked.connect(self.show_album_art)

        gn = main_window.selected[0]
        # print(gn)

        # ---- Dance 장르 모델 및 csv 파일 불러오기 for recommendation ----
        self.Tfidf_matrix = mmread('./Melon/Models/Tfidf_{}_lyric.mtx'.format(gn)).tocsr()
        with open('./Melon/Models/{}_tfidf.pickle'.format(gn), 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon/Models/{}_w2v.model'.format(gn))
        self.df = pd.read_csv('./Melon/08_album_info/{}_track_data.csv'.format(gn))

        # ---- search bar 자동완성 ----
        self.titles = list(self.df['title'])
        self.titles.sort()
        model = QStringListModel()
        model.setStringList(self.titles)
        completer = QCompleter()
        completer.setModel(model)
        self.le_word.setCompleter(completer)


    # ---- 감정 recommendation ----
    def energetic_reco(self):
        reco_list = self.recom_by_keyword(self.main_window.selected[1])
        print(reco_list)
        reco_list = reco_list[reco_list['emo'] == 'Energetic']
        reco_list.info()
        num = len(reco_list)
        if num >= 7 : num = 7
        for i in range(num):
            self.buttons[i].setText(reco_list.iloc[i, 0] + '- ' + reco_list.iloc[i, 1])

    def happy_reco(self):
        reco_list = self.recom_by_keyword(self.main_window.selected[1])
        print(reco_list)
        reco_list = reco_list[reco_list['emo'] == 'Happy']
        reco_list.info()
        num = len(reco_list)
        if num >= 7 : num = 7
        for i in range(num):
            self.buttons[i].setText(reco_list.iloc[i, 0] + '- ' + reco_list.iloc[i, 1])

    def comfortable_reco(self):
        reco_list = self.recom_by_keyword(self.main_window.selected[1])
        print(reco_list)
        reco_list = reco_list[reco_list['emo'] == 'Comfortable']
        reco_list.info()
        num = len(reco_list)
        if num >= 7 : num = 7
        for i in range(num):
            self.buttons[i].setText(reco_list.iloc[i, 0] + '- ' + reco_list.iloc[i, 1])

    def chilling_reco(self):
        reco_list = self.recom_by_keyword(self.main_window.selected[1])
        print(reco_list)
        reco_list = reco_list[reco_list['emo'] == 'Chilling']
        reco_list.info()
        num = len(reco_list)
        if num >= 7 : num = 7
        for i in range(num):
            self.buttons[i].setText(reco_list.iloc[i, 0] + '- ' + reco_list.iloc[i, 1])

    def depressed_reco(self):
        reco_list = self.recom_by_keyword(self.main_window.selected[1])
        print(reco_list)
        reco_list = reco_list[reco_list['emo'] == 'Depressed']
        reco_list.info()
        num = len(reco_list)
        if num >= 7 : num = 7
        for i in range(num):
            self.buttons[i].setText(reco_list.iloc[i, 0] + '- ' + reco_list.iloc[i, 1])

    def sad_reco(self):
        reco_list = self.recom_by_keyword(self.main_window.selected[1])
        print(reco_list)
        reco_list = reco_list[reco_list['emo'] == 'Sad']
        reco_list.info()
        num = len(reco_list)
        if num >= 7 : num = 7
        for i in range(num):
            self.buttons[i].setText(reco_list.iloc[i, 0] + '- ' + reco_list.iloc[i, 1])


    # ---- album art 이미지 띄우기 ----




    def home(self):
        self.close()

    def recom_by_keyword(self, key_word):
        if key_word:
            key_word = key_word.split()[0]
            try:
                sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
            except:
                pass
            sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)

            words = [key_word]
            for word, _ in sim_word:
                words.append(word)
            sentence = []
            count = 30
            for word in words:
                sentence = sentence + [word] * count
                count -= 1

            sentence_vec = self.Tfidf.transform(sentence)
            cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
            recommendation = self.getRecommendation(cosine_sim)
            recommendation['emo'] = self.df['emo']

            print(recommendation)
            return recommendation
        else:
            return 0

    def recom_by_title(self, title):
        title_idx = self.df[self.df['title'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[title_idx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        print(recommendation[1:8])
        return recommendation

    def getRecommendation(self, cosin_sim):
        simScore = list(enumerate(cosin_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:30]
        artistIdx = [i[0] for i in simScore]
        print(artistIdx)
        recSongList = self.df.iloc[artistIdx, [0, 1]]
        print(recSongList)
        return recSongList









