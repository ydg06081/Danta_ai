# -*- coding: utf-8 -*-
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QAxContainer import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLineEdit
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

'''
    신한아이 인디 국내주식 실시간 현재가 데이터 조회 예제
'''

class IndiWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IndiExample")

        # 일반 TR OCX
        self.IndiTR = QAxWidget("GIEXPERTCONTROL.GiExpertControlCtrl.1")
        self.IndiTR.ReceiveData.connect(self.ReceiveData)
        self.IndiTR.ReceiveSysMsg.connect(self.ReceiveSysMsg) # 일반 TR에 대한 응답을 받는 함수를 연결해 줍니다.

        # 실시간 TR OCX
        self.IndiReal = QAxWidget("GIEXPERTCONTROL.GiExpertControlCtrl.1")
        self.IndiReal.ReceiveRTData.connect(self.ReceiveRTData) # 실시간 TR에 대한 응답을 받는 함수를 연결해줍니다.

        # TR ID를 저장해놓고 처리할 딕셔너리 생성
        self.rqidD = {}

        # PyQt5를 통해 화면을 그려주는 코드입니다.
        self.MainSymbol = ""
        self.edSymbol = QLineEdit(self)
        self.edSymbol.setGeometry(20, 20, 60, 20)
        self.edSymbol.setText("")

        # PyQt5를 통해 버튼만들고 함수와 연결시킵니다.
        btnResearch = QPushButton("Search", self)
        btnResearch.setGeometry(85, 20, 50, 20)
        btnResearch.clicked.connect(self.btn_Search)  # 버튼을 누르면 'btn_Search' 함수가 실행됩니다.

    # 버튼을 누르면 일반TR로 현재가 조회를 먼저 요청을 합니다.
    def btn_Search(self):
        # 조회할 종목의 이름을 받아옵니다.
        Symbol = self.edSymbol.text()
        self.MainSymbol = Symbol

        # 기존종목 실시간 TR연결 해제, 여러 종목을 실시간 TR에 등록해도 상관없습니다.
        if self.MainSymbol != "":
            self.IndiReal.dynamicCall("UnRequestRTReg(QString, QString)", "SB", self.MainSymbol)
            self.IndiReal.dynamicCall("UnRequestRTReg(QString, QString)", "SC", self.MainSymbol)
            self.IndiReal.dynamicCall("UnRequestRTReg(QString, QString)", "SH", self.MainSymbol)

        # 종목 현재가 조회, 일반TR로 요청합니다. 바로 실시간 TR을 등록해도 상관없습니다. 
        ret = self.IndiTR.dynamicCall("SetQueryName(QString)", "SC")
        ret = self.IndiTR.dynamicCall("SetSingleData(int, QString)", 0, self.MainSymbol) # 인풋 : 단축코드
        rqid = self.IndiTR.dynamicCall("RequestData()")

        self.rqidD[rqid] = "SC"

    # 실시간 TR로 등록을 해줍니다.
    def ReceiveData(self, rqid):
        TRName = self.rqidD[rqid]
        if TRName == "SC":
            print(TRName)
            DATA = {}
            DATA['ISIN_CODE'] = self.IndiTR.dynamicCall("GetSingleData(int)", 0) # 표준코드
            DATA['CODE'] = self.IndiTR.dynamicCall("GetSingleData(int)", 1)       # 단축코드
            DATA['Time'] = self.IndiTR.dynamicCall("GetSingleData(int)", 2)       # 채결시간
            DATA['Close'] = self.IndiTR.dynamicCall("GetSingleData(int)", 3)      # 현재가
            DATA['Vol'] = self.IndiTR.dynamicCall("GetSingleData(int)", 7)        # 누적거래량
            DATA['TRADING_VALUE'] = self.IndiTR.dynamicCall("GetSingleData(int)", 8)  # 누적거래대금
            DATA['ContQty'] = self.IndiTR.dynamicCall("GetSingleData(int)", 9)   # 단위채결량
            DATA['Open'] = self.IndiTR.dynamicCall("GetSingleData(int)", 10)      # 시가
            DATA['High'] = self.IndiTR.dynamicCall("GetSingleData(int)", 11)      # 고가
            DATA['Low'] = self.IndiTR.dynamicCall("GetSingleData(int)", 12)       # 저가

            # 실시간 등록
            ret = self.IndiReal.dynamicCall("RequestRTReg(QString, QString)", "SC", self.MainSymbol)
            print(DATA)

        self.rqidD.__delitem__(rqid)

    # 실시간 등록을 진행하고 실시간으로 데이터를 받아오도록 한다.
    def ReceiveRTData(self, RealType):

        if RealType == "SC":
            DATA = {}
            DATA['ISIN_CODE'] = self.IndiReal.dynamicCall("GetSingleData(int)", 0)  # 표준코드
            DATA['CODE'] = self.IndiReal.dynamicCall("GetSingleData(int)", 1)  # 단축코드
            DATA['Time'] = self.IndiReal.dynamicCall("GetSingleData(int)", 2)  # 채결시간
            DATA['Close'] = self.IndiReal.dynamicCall("GetSingleData(int)", 3)  # 현재가
            DATA['Vol'] = self.IndiReal.dynamicCall("GetSingleData(int)", 7)  # 누적거래량
            DATA['TRADING_VALUE'] = self.IndiReal.dynamicCall("GetSingleData(int)", 8)  # 누적거래대금
            DATA['ContQty'] = self.IndiReal.dynamicCall("GetSingleData(int)", 9)  # 단위채결량
            DATA['Open'] = self.IndiReal.dynamicCall("GetSingleData(int)", 10)  # 시가
            DATA['High'] = self.IndiReal.dynamicCall("GetSingleData(int)", 11)  # 고가
            DATA['Low'] = self.IndiReal.dynamicCall("GetSingleData(int)", 12)  # 저가

            print(DATA)

    def ReceiveSysMsg(self, MsgID):
        print("System Message Received = ", MsgID)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    IndiWindow = IndiWindow()
    IndiWindow.show()
    app.exec_()

