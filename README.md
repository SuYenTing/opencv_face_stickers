# 以OpenCV實作人臉偵測、特徵捕捉與變裝

## 說明

這個小專題是我在Tibame AI/Big Data資料分析師養成班的OpenCV課程後的自主練習。

主要是以筆電攝像頭拍攝自己的人臉，程式會偵測人臉位置，並在人臉上加入一些貼圖做變裝。

這個功能目前在坊間的修圖軟體皆可以看到，我自己是利用OpenCV搭配Dlib函數庫來實作。

成果範例：

![](./images/demo.gif)


## 檔案說明

* main.py：主程式

* document.ipynb：Jupyter Notebook檔案，說明程式實作過程。若想直接觀看說明，可至我的[Blog文章](https://suyenting.github.io/post/opencv_face_stickers/)觀看。

* requirements.txt：需要安裝的package清單

* images：圖庫

## 備註

執行程式前請先下載OpenCV人臉特徵捕捉的預訓練模型：shape_predictor_68_face_landmarks.dat

下載位置請參考[OpenCV Github頁面說明](https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py)
或者[stackoverflow問題解決辦法](https://stackoverflow.com/questions/64643440/how-do-i-fix-runtimeerror-unable-to-open-shape-predictor-68-face-landmarks-dat)