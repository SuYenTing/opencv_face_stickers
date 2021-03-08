# 以OpenCV實作人臉偵測、特徵捕捉與變裝

## 說明

這個小專題是我在Tibame AI/Big Data資料分析師養成班的OpenCV課程後的自主練習。

主要是以筆電攝像頭拍攝自己的人臉，程式會偵測人臉位置，並在人臉上加入一些貼圖做變裝。

這個功能目前在坊間的修圖軟體皆可以看到，我自己是利用OpenCV搭配Dlib函數庫來實作。

成果範例：

![](./images/demo.gif)


## 檔案說明

* main.py：主程式

* manual.ipynb：Jupyter Notebook檔案，說明本專案實作過程

* requirements.txt：需要安裝的package清單

* images：圖庫

* shape_predictor_68_face_landmarks.dat：OpenCV人臉特徵捕捉的預訓練模型，[參考頁面](https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py)

