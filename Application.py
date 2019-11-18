import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
import distance
import numpy as np
import pytesseract
from pytesseract import Output
import cv2
from PIL import Image
import pandas as pd
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing import sequence


class Ui_PrivateDocumentDetection(object):
    def setupUi(self, PrivateDocumentDetection):
        PrivateDocumentDetection.setObjectName("PrivateDocumentDetection")
        PrivateDocumentDetection.resize(800, 597)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(203, 222, 227))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(222, 222, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(203, 222, 227))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(222, 222, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(203, 222, 227))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(222, 222, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(222, 222, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        PrivateDocumentDetection.setPalette(palette)
        self.centralwidget = QtWidgets.QWidget(PrivateDocumentDetection)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.pathImg = QtWidgets.QLineEdit(self.centralwidget)
        self.pathImg.setObjectName("pathImg")
        self.gridLayout.addWidget(self.pathImg, 0, 0, 1, 2)
        self.download = QtWidgets.QPushButton(self.centralwidget)
        self.download.setObjectName("download")
        self.gridLayout.addWidget(self.download, 0, 2, 1, 1)
        self.screen = QtWidgets.QGraphicsView(self.centralwidget)
        self.screen.setObjectName("screen")
        self.scene = QtWidgets.QGraphicsScene(self.centralwidget)
        self.screen.setScene(self.scene)
        self.gridLayout.addWidget(self.screen, 1, 0, 1, 2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.chooseSetting = QtWidgets.QComboBox(self.centralwidget)
        self.chooseSetting.setObjectName("chooseSetting")
        self.chooseSetting.addItem("")
        self.chooseSetting.addItem("")
        self.chooseSetting.addItem("")
        self.chooseSetting.addItem("")
        self.gridLayout.addWidget(self.chooseSetting, 2, 1, 1, 1)
        self.result = QtWidgets.QLineEdit(self.centralwidget)
        self.result.setObjectName("result")
        self.gridLayout.addWidget(self.result, 3, 0, 1, 2)
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 3, 2, 1, 1)
        PrivateDocumentDetection.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(PrivateDocumentDetection)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        PrivateDocumentDetection.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(PrivateDocumentDetection)
        self.statusbar.setObjectName("statusbar")
        PrivateDocumentDetection.setStatusBar(self.statusbar)

        self.retranslateUi(PrivateDocumentDetection)
        QtCore.QMetaObject.connectSlotsByName(PrivateDocumentDetection)

    def retranslateUi(self, PrivateDocumentDetection):
        _translate = QtCore.QCoreApplication.translate
        PrivateDocumentDetection.setWindowTitle(_translate("PrivateDocumentDetection", "MainWindow"))
        self.download.setText(_translate("PrivateDocumentDetection", "Download"))
        self.label.setText(_translate("PrivateDocumentDetection", "Please, choose classifier"))
        self.chooseSetting.setItemText(0, _translate("PrivateDocumentDetection", "Keyword spotting"))
        self.chooseSetting.setItemText(1, _translate("PrivateDocumentDetection", "CNN"))
        self.chooseSetting.setItemText(2, _translate("PrivateDocumentDetection", "FCNN"))
        self.chooseSetting.setItemText(3, _translate("PrivateDocumentDetection", "LSTM"))
        self.start.setText(_translate("PrivateDocumentDetection", "Start"))


class Application(QtWidgets.QMainWindow, Ui_PrivateDocumentDetection):
    def display_pic(self):

        path = self.pathImg.text()
        print(path)
        input_img = cv2.imread(path)
        cv2.imshow('njn', input_img)

        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(222, 222, 240), QtCore.Qt.SolidPattern))
        self.scene.addPixmap(QtGui.QPixmap(path))
        self.scene.clearSelection()
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        image = QtGui.QImage(self.scene.sceneRect().size().toSize(), QtGui.QImage.Format_ARGB32)
        image.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(image)
        self.scene.render(painter)
        # image.save("file_name2.png")
        painter.end()

    def chooseType(self):

        value = self.chooseSetting.currentIndex()
        return value

    def det(word):
        sensFile = open('D:\\test\stoplist.txt', "r")
        flag = 0
        for line in sensFile:
            tmp = line.split()[0]
            simm = (len(tmp) - distance.levenshtein(tmp, word)) / len(tmp)
            if simm > 0.8:
                flag = 1
                break
            # outputFile.write(tmp+' '+str(simm)+'\n')
        # outputFile.close()
        sensFile.close()
        return flag

    def sensitive(list):
        if (1 in list):
            return 1
        else:
            return 0

    def detect_tess(path_to_image):
        print('img ' + path_to_image)
        input_img = cv2.imread(path_to_image)
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--oem 2 --psm 3')
        # print(pytesseract.image_to_string(img))
        # print(d['text'])

        word_level = np.zeros(len(d['text']))
        for i in range(len(d['text'])):
            print(d['text'][i])
            word_level[i] = Application.det(d['text'][i])
            print(word_level[i])
        prediction = Application.sensitive(word_level)
        Application.show_sensitive_text(input_img, d)
        return prediction

    def show_sensitive_text(input_img, data):
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            if (Application.det(data['text'][i]) == 1):
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(input_img, data['text'][i], (x + w + 3, y + h), 4, 0.5, (0, 0, 126))
        cv2.imshow('img', input_img)
        cv2.imwrite('tmp.jpg', input_img)
        # cv2.waitKey(0)

    def show_detected_text(input_img, data):
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            # print((d['text'][i]!='')&(d['text'][i]!='    '))
            # if (det(d['text'][i])==1):
            cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(input_img, data['text'][i], (x + w + 3, y + h), 4, 0.5, (0, 126, 0))
            # cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            # cv2.putText(input_img,d['text'][i],(x+w+3,y+h),4,0.5,(0,0,126))
        cv2.imshow('img', input_img)
        # cv2.imwrite('psp.jpg', input_img)
        cv2.waitKey(0)

    def setResult(prediction):
        if (prediction == 1):
            return 'Image contains confidential data'
        if (prediction == 0):
            return 'Image does not contain confidential data'

    def classify(self):
        option = self.chooseType()
        path = self.pathImg.text()
        print(option)
        if (option == 0):  # spotting
            print('hello')
            prediction = Application.detect_tess(path)
            self.scene.clearSelection()
            self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(222, 222, 240), QtCore.Qt.SolidPattern))
            self.scene.addPixmap(QtGui.QPixmap('tmp.jpg'))

            self.scene.setSceneRect(self.scene.itemsBoundingRect())
            image = QtGui.QImage(self.scene.sceneRect().size().toSize(), QtGui.QImage.Format_ARGB32)
            image.fill(QtCore.Qt.transparent)
            self.scene.clearSelection()

            painter = QtGui.QPainter(image)
            self.scene.render(painter)
            # image.save("file_name2.png")
            painter.end()
            # self.result.setText(Application.setResult(prediction))
        if (option == 1):
            print('hello')
            prediction = self.predictCNN()
            # self.result.setText(Application.setResult(prediction))
        if (option == 2):
            print('hello')
            prediction = self.predictFCNN()
            # self.result.setText(Application.setResult(prediction))
        if (option == 3):
            print('hello')
            prediction = self.predictLSTM()
            # self.result.setText(Application.setResult(prediction))

        self.result.setText(Application.setResult(prediction))
        self.pathImg.clear()

    def predictFCNN(self):
        X_data = list()
        model = Application.load_fcnn()
        image_pix = Image.open(self.pathImg.text())
        f = pd.read_csv('D:\\test\text.csv', sep=',', header=None)
        f = f.transpose()
        f.dropna(inplace=True)
        vocabPD = pd.DataFrame(f[0].value_counts()[0:5000])
        vocabPD.reset_index(inplace=True)
        vocabPD.columns = ['0', 'counts']
        image_data = pytesseract.image_to_data(image_pix, output_type=Output.DICT)

        image_cat = np.zeros(vocabPD.shape[0])
        image_cat = vocabPD['0'].isin(image_data['text']).astype(int)
        X_data.append(np.array(image_cat))
        pred = model.predict(np.array(X_data))
        pred = pred[0][0]
        pred = round(pred)
        return pred

    def predictCNN(self):
        X_nums = list()
        model = Application.load_cnn()
        image_pix = Image.open(self.pathImg.text())
        f = pd.read_csv('D:\\test\text.csv', sep=',', header=None)
        f = f.transpose()
        f.dropna(inplace=True)
        vocabPD = pd.DataFrame(f[0].value_counts()[0:5000])
        vocabPD.reset_index(inplace=True)
        vocabPD.columns = ['0', 'counts']
        image_data = pytesseract.image_to_data(image_pix, output_type=Output.DICT)
        image_nums = list()
        image_nums = vocabPD.index[vocabPD['0'].isin(image_data['text'])].tolist()
        X_nums.append(np.array(image_nums))
        X_nums = sequence.pad_sequences(X_nums, maxlen=400)
        pred = model.predict(np.array(X_nums))
        pred = pred[0][0]
        pred = round(pred)
        return pred

    def predictLSTM(self):
        X_nums = list()
        model = Application.load_lstm()
        image_pix = Image.open(self.pathImg.text())
        f = pd.read_csv('D:\\test\.csv', sep=',', header=None)
        f = f.transpose()
        f.dropna(inplace=True)
        vocabPD = pd.DataFrame(f[0].value_counts()[0:5000])
        vocabPD.reset_index(inplace=True)
        vocabPD.columns = ['0', 'counts']
        image_data = pytesseract.image_to_data(image_pix, output_type=Output.DICT)
        image_nums = list()
        image_nums = vocabPD.index[vocabPD['0'].isin(image_data['text'])].tolist()
        X_nums.append(np.array(image_nums))
        X_nums = sequence.pad_sequences(X_nums, maxlen=400)
        pred = model.predict(np.array(X_nums))
        pred = pred[0][0]
        pred = round(pred)
        return pred

    def load_fcnn():
        file_json = open("dense1.json", "r")
        load_json = file_json.read()
        file_json.close()
        load = model_from_json(load_json)
        load.load_weights("dense1.h5")

        load.compile(loss=losses.binary_crossentropy, optimizer=optimizers.RMSprop(lr=0.001),
                     metrics=[metrics.binary_accuracy])
        return load

    def load_cnn():
        file_json = open("emb1.json", "r")
        load_json = file_json.read()
        file_json.close()
        load = model_from_json(load_json)
        load.load_weights("emb1.h5")

        load.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
                     metrics=['acc'])
        return load

    def load_lstm():
        file_json = open("lstm.json", "r")
        load_json = file_json.read()
        file_json.close()
        load = model_from_json(load_json)
        load.load_weights("lstm.h5")

        load.compile(loss='binary_crossentropy', optimizer='rmsprop',
                     metrics=['accuracy'])
        return load

    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.download.clicked.connect(self.display_pic)  # Выполнить функцию display_pic
        # при нажатии кнопки
        self.chooseSetting.activated.connect(self.chooseType)
        self.start.clicked.connect(self.classify)


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = Application()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()