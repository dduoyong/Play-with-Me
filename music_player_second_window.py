import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
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


        # 재생버튼
        self.btn_play.clicked.connect(self.play_music)

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
            # self.buttons[i].clicked.connect(lambda: webbrowser.open(reco_list.iloc[i, 4]))

    def happy_reco(self):
        self.reco_list = self.recom_by_keyword(self.main_window.selected[1])
        print(self.reco_list)
        self.reco_list = self.reco_list[self.reco_list['emo'] == 'Happy']
        self.reco_list.info()
        num = len(self.reco_list)
        if num >= 7 : num = 7
        for i in range(num):
            self.buttons[i].setText(self.reco_list.iloc[i, 0] + '-' + self.reco_list.iloc[i, 1])


    def show_album_art(self):
        btn = self.sender()
        title = btn.text().split('-')[-1]

        # Web에서 Image 정보 로드
        urlString = self.reco_list[self.reco_list['title'] == title]['album_art']
        urlString = list(urlString)[0]
        # print(urlString)

        imageFromWeb = urllib.request.urlopen(urlString).read()
        # print(type(imageFromWeb))

        # 웹에서 Load한 Image를 이용하여 QPixmap에 사진데이터를 Load하고, Label을 이용하여 화면에 표시
        pixmap = QPixmap()
        pixmap.loadFromData(imageFromWeb)
        # print(pixmap)
        self.lbl_album_art.setPixmap(pixmap)

    def play_music(self):
        btn = self.sender()
        title = btn.text().split('-')[-1]
        track_url = self.reco_list[self.reco_list['title'] == title]['track_url']
        print('title:', title)
        print(track_url)
        # track_url = list(track_url)
        print('track_url:', track_url)
        url = QUrl('track_url')
        QDesktopServices.openUrl(url)

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
            recommendation['track_url'] = self.df['track_url']
            recommendation['album_art'] = self.df['album_art']
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









