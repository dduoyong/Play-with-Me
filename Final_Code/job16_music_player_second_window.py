import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
from PyQt5.QtWebEngineWidgets import QWebEngineView
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
import requests


emo_window = uic.loadUiType('./music_player_2.ui')[0]

class secondwindow(QDialog, QWidget, emo_window):
    def __init__(self, main_window):
        super().__init__()
        self.setupUi(self)
        self.main_window = main_window
        self.return_value = 1
        # ---- 이미지 설정 ----
        # 프로필 이미지
        self.lbl_profile.setPixmap(QtGui.QPixmap('./PyQt_imgs/profile.png'))
        # 홈 버튼 이미지
        self.btn_home.setIcon(QtGui.QIcon('./PyQt_imgs/home.png'))
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
        # 장르 버튼
        self.btn_adultpop.clicked.connect(self.adultpop)
        self.btn_ballad.clicked.connect(self.ballad)
        self.btn_dance.clicked.connect(self.dance)
        self.btn_fnb.clicked.connect(self.fnb)
        self.btn_indie.clicked.connect(self.indie)
        self.btn_idol.clicked.connect(self.idol)
        self.btn_pop.clicked.connect(self.pop)
        self.btn_rnb.clicked.connect(self.rnb)
        self.btn_hiphop.clicked.connect(self.hiphop)
        self.btn_rockmetal.clicked.connect(self.rockmetal)
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

        # ---- 장르 모델 및 csv 파일 불러오기 ----
        gn = main_window.selected[0]
        # print(gn)
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_{}_lyric.mtx'.format(gn)).tocsr()
        with open('./Melon_Data/Models/{}_tfidf.pickle'.format(gn), 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/{}_w2v.model'.format(gn))
        self.df = pd.read_csv('./Melon_Data/08_album_info/{}_track_data.csv'.format(gn))


    # ---- 장르별 추천 ----
    def adultpop(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Adult Pop_lyric.mtx').tocsr()
        with open('./Melon_Data/Models/Adult Pop_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Adult Pop_w2v.model')
        self.df = pd.read_csv('./Melon_Data/08_album_info/Adult Pop_track_data.csv')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def ballad(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Ballad_lyric.mtx').tocsr()
        with open('../Melon_Data/Models/Ballad_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Ballad_w2v.model')
        self.df = pd.read_csv('../Melon_Data/08_album_info/Ballad_track_data.csv')
        self.btn_ballad.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def dance(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Dance_lyric.mtx').tocsr()
        with open('../Melon_Data/Models/Dance_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Dance_w2v.model')
        self.df = pd.read_csv('../Melon_Data/08_album_info/Dance_track_data.csv')
        self.btn_dance.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def fnb(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Fork Blues_lyric.mtx').tocsr()
        with open('./Melon_Data/Models/Fork Blues_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Fork Blues_w2v.model')
        self.df = pd.read_csv('./Melon_Data/08_album_info/Fork Blues_track_data.csv')
        self.btn_fnb.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def indie(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Indie_lyric.mtx').tocsr()
        with open('../Melon_Data/Models/Indie_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Indie_w2v.model')
        self.df = pd.read_csv('../Melon_Data/08_album_info/Indie_track_data.csv')
        self.btn_indie.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def idol(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Idol_lyric.mtx').tocsr()
        with open('../Melon_Data/Models/Idol_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Idol_w2v.model')
        self.df = pd.read_csv('../Melon_Data/08_album_info/Idol_track_data.csv')
        self.btn_idol.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def hiphop(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Hip Hop_lyric.mtx').tocsr()
        with open('./Melon_Data/Models/Hip Hop_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Hip Hop_w2v.model')
        self.df = pd.read_csv('./Melon_Data/08_album_info/Hip Hop_track_data.csv')
        self.btn_hiphop.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def pop(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Pop_lyric.mtx').tocsr()
        with open('../Melon_Data/Models/Pop_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Pop_w2v.model')
        self.df = pd.read_csv('../Melon_Data/08_album_info/Pop_track_data.csv')
        self.btn_pop.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def rnb(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_RnB Soul_lyric.mtx').tocsr()
        with open('./Melon_Data/Models/RnB Soul_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/RnB Soul_w2v.model')
        self.df = pd.read_csv('./Melon_Data/08_album_info/RnB Soul_track_data.csv')
        self.btn_rnb.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_rockmetal.setStyleSheet('color:rgb(137, 137, 137)')

    def rockmetal(self):
        self.Tfidf_matrix = mmread('./Melon_Data/Models/Tfidf_Rock Metal_lyric.mtx').tocsr()
        with open('./Melon_Data/Models/Rock Metal_tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./Melon_Data/Models/Rock Metal_w2v.model')
        self.df = pd.read_csv('./Melon_Data/08_album_info/Rock Metal_track_data.csv')
        self.btn_rockmetal.setStyleSheet('color:rgb(78, 159, 151)')

        # —- 장르 버튼 색상 변경 ——
        self.btn_adultpop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_ballad.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_dance.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_fnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_indie.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_idol.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_hiphop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_pop.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rnb.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_rockmetal.setStyleSheet('color:rgb(78, 159, 151)')

    # ---- 감정별 추천 ----
    def energetic_reco(self):
        self.reco_list = self.recom_by_keyword(self.main_window.selected[1])
        # print(self.reco_list['emo'])
        try:
            # 키워드가 vocabulary에 없으면 창 닫고 첫번째 창으로 이동
            if self.reco_list == 0:
                self.return_value = 0
                self.close()
        except:
            self.reco_list = self.reco_list[self.reco_list['emo'] == 'Energetic']
            if len(self.reco_list):
                num = len(self.reco_list)
                for i in range(7):
                    self.buttons[i].setText('')
                if num >= 7: num = 7
                if num > 0:
                    for i in range(num):
                        self.buttons[i].setText(self.reco_list.iloc[i, 0] + '-' + self.reco_list.iloc[i, 1])

            else:
                self.return_value = 0
                self.close()


        # —- 감정 버튼 색상 변경 ——
        self.btn_energetic.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_happy.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_comfortable.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_chilling.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_depressed.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_sad.setStyleSheet('color:rgb(137, 137, 137)')


    def happy_reco(self):
        self.reco_list = self.recom_by_keyword(self.main_window.selected[1])
        try:
            if self.reco_list == 0:
                self.return_value = 0
                self.close()
        except:
            self.reco_list = self.reco_list[self.reco_list['emo'] == 'Happy']
            if len(self.reco_list):
                num = len(self.reco_list)
                for i in range(7):
                    self.buttons[i].setText('')
                if num >= 7: num = 7
                if num > 0:
                    for i in range(num):
                        self.buttons[i].setText(self.reco_list.iloc[i, 0] + '-' + self.reco_list.iloc[i, 1])

            else:
                self.return_value = 0
                self.close()

        # —- 감정 버튼 색상 변경 ——
        self.btn_energetic.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_happy.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_comfortable.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_chilling.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_depressed.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_sad.setStyleSheet('color:rgb(137, 137, 137)')

    def comfortable_reco(self):
        self.reco_list = self.recom_by_keyword(self.main_window.selected[1])
        # print(self.reco_list['emo'])
        try:
            if self.reco_list == 0:
                self.return_value = 0
                self.close()
        except:
            self.reco_list = self.reco_list[self.reco_list['emo'] == 'Comfortable']
            if len(self.reco_list):
                num = len(self.reco_list)
                for i in range(7):
                    self.buttons[i].setText('')
                if num >= 7: num = 7
                if num > 0:
                    for i in range(num):
                        self.buttons[i].setText(self.reco_list.iloc[i, 0] + '-' + self.reco_list.iloc[i, 1])

            else:
                self.return_value = 0
                self.close()

        # —- 감정 버튼 색상 변경 ——
        self.btn_energetic.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_happy.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_comfortable.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_chilling.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_depressed.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_sad.setStyleSheet('color:rgb(137, 137, 137)')


    def chilling_reco(self):
        self.reco_list = self.recom_by_keyword(self.main_window.selected[1])
        # print(self.reco_list['emo'])
        try:

            if self.reco_list == 0:
                self.return_value = 0
                self.close()
        except:
            self.reco_list = self.reco_list[self.reco_list['emo'] == 'Chilling']
            if len(self.reco_list):
                num = len(self.reco_list)
                for i in range(7):
                    self.buttons[i].setText('')
                if num >= 7: num = 7
                if num > 0:
                    for i in range(num):
                        self.buttons[i].setText(self.reco_list.iloc[i, 0] + '-' + self.reco_list.iloc[i, 1])

            else:
                self.return_value = 0
                self.close()

        # —- 감정 버튼 색상 변경 ——
        self.btn_energetic.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_happy.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_comfortable.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_chilling.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_depressed.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_sad.setStyleSheet('color:rgb(137, 137, 137)')

    def depressed_reco(self):
        self.reco_list = self.recom_by_keyword(self.main_window.selected[1])
        # print(self.reco_list['emo'])
        try:
            if self.reco_list == 0:
                self.return_value = 0
                self.close()
        except:
            self.reco_list = self.reco_list[self.reco_list['emo'] == 'Depressed']
            if len(self.reco_list):
                num = len(self.reco_list)
                for i in range(7):
                    self.buttons[i].setText('')
                if num >= 7: num = 7
                if num > 0:
                    for i in range(num):
                        self.buttons[i].setText(self.reco_list.iloc[i, 0] + '-' + self.reco_list.iloc[i, 1])

            else:
                self.return_value = 0
                self.close()

        # —- 감정 버튼 색상 변경 ——
        self.btn_energetic.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_happy.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_comfortable.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_chilling.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_depressed.setStyleSheet('color:rgb(78, 159, 151)')
        self.btn_sad.setStyleSheet('color:rgb(137, 137, 137)')

    def sad_reco(self):
        self.reco_list = self.recom_by_keyword(self.main_window.selected[1])
        # print(self.reco_list['emo'])
        try:
            if self.reco_list == 0:
                self.return_value = 0
                self.close()
        except:
            self.reco_list = self.reco_list[self.reco_list['emo'] == 'Sad']
            if len(self.reco_list):
                num = len(self.reco_list)
                for i in range(7):
                    self.buttons[i].setText('')
                if num >= 7: num = 7
                if num > 0:
                    for i in range(num):
                        self.buttons[i].setText(self.reco_list.iloc[i, 0] + '-' + self.reco_list.iloc[i, 1])

            else:
                self.return_value = 0
                self.close()

        # —- 감정 버튼 색상 변경 ——
        self.btn_energetic.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_happy.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_comfortable.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_chilling.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_depressed.setStyleSheet('color:rgb(137, 137, 137)')
        self.btn_sad.setStyleSheet('color:rgb(78, 159, 151)')

    def show_album_art(self):
        btn = self.sender()
        self.title = btn.text().split('-')[-1]

        # Web에서 Image 정보 로드
        urlString = self.reco_list[self.reco_list['title'] == self.title]['album_art']
        if list(urlString):
            urlString = list(urlString)[0]
            print(urlString)
            # print(urlString)

            imageFromWeb = urllib.request.urlopen(urlString).read()
            # print(type(imageFromWeb))

            # 웹에서 Load한 Image를 이용하여 QPixmap에 사진데이터를 Load하고, Label을 이용하여 화면에 표시
            pixmap = QPixmap()
            pixmap.loadFromData(imageFromWeb)
            # print(pixmap)
            self.lbl_album_art.setPixmap(pixmap)

        else:
            self.lbl_album_art.setPixmap(QtGui.QPixmap('./PyQt_imgs/album_art.png'))

    def play_music(self):

        track_url = self.reco_list[self.reco_list['title'] == self.title]['track_url']
        # print('title:', title)
        # print(track_url)
        track_url = list(track_url)[0]
        track_url = track_url.lstrip()
        webbrowser.open(track_url)

    def home(self):
        self.close()

    def recom_by_keyword(self, key_word):
        if key_word:
            key_word = key_word.split()[0]
            try:
                sim_word = self.embedding_model.wv.most_similar(key_word, topn=10)
            except:
                return 0
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

