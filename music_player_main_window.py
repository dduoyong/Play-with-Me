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
import music_player_second_window
# from konlpy.tag import Okt
# import re


#----pyqt .ui 파일 불러오기----
gn_window = uic.loadUiType('./music_player_1.ui')[0]


class Exam(QWidget, gn_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ---- 이미지 설정 ----
        # 장르 이미지
        self.lbl_adultpop.setPixmap(QtGui.QPixmap('./PyQt_imgs/adultpop.png'))
        self.lbl_ballad.setPixmap(QtGui.QPixmap('./PyQt_imgs/ballad.png'))
        self.lbl_dance.setPixmap(QtGui.QPixmap('./PyQt_imgs/dance.png'))
        self.lbl_fnb.setPixmap(QtGui.QPixmap('./PyQt_imgs/forkblues.png'))
        self.lbl_indie.setPixmap(QtGui.QPixmap('./PyQt_imgs/indie.png'))
        self.lbl_idol.setPixmap(QtGui.QPixmap('./PyQt_imgs/idol.png'))
        self.lbl_pop.setPixmap(QtGui.QPixmap('./PyQt_imgs/pop.png'))
        self.lbl_rnb.setPixmap(QtGui.QPixmap('./PyQt_imgs/randb.png'))
        self.lbl_hiphop.setPixmap(QtGui.QPixmap('./PyQt_imgs/hiphop.png'))
        self.lbl_rockmetal.setPixmap(QtGui.QPixmap('./PyQt_imgs/rockmetal.png'))
        # 프로필 이미지
        self.lbl_profile.setPixmap(QtGui.QPixmap('./PyQt_imgs/profile.png'))

        # ---- 버튼 설정 ----
        # 장르 버튼
        self.btn_adultpop.clicked.connect(self.second_window)
        self.btn_ballad.clicked.connect(self.second_window)
        self.btn_dance.clicked.connect(self.second_window)
        self.btn_fnb.clicked.connect(self.second_window)
        self.btn_indie.clicked.connect(self.second_window)
        self.btn_idol.clicked.connect(self.second_window)
        self.btn_pop.clicked.connect(self.second_window)
        self.btn_rnb.clicked.connect(self.second_window)
        self.btn_hiphop.clicked.connect(self.second_window)
        self.btn_rockmetal.clicked.connect(self.second_window)


    def second_window(self):
        btn = self.sender()
        self.selected = (btn.text(), self.le_keyword.text())
        self.hide()
        self.second = music_player_second_window.secondwindow(self)
        self.second.exec()
        self.show()








class APP(QMainWindow):
    def __init__(self):

        # 버튼에 링크 추가하기
        self.btn_title_1.clicked.connect(lambda: webbrowser.open('df[track_url][i]'))

















if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())