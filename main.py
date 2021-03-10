# 人臉辨識加素材圖片
import cv2
import dlib


# 建立素材圖片物件類別
class MaterialImg:

    # 建構式
    def __init__(self, materialImg, putMark, widthMarks, heightMarks, materialImgSize, faceImg, landmarks):
        self.materialImg = materialImg  # 素材圖片
        self.putMark = putMark  # 圖片放置對應的人臉辨識點
        self.widthMarks = widthMarks  # 計算臉部相對位置寬度(用於圖片縮放比例)
        self.heightMarks = heightMarks  # 計算臉部相對位置高度(用於圖片縮放比例)
        self.materialImgSize = materialImgSize  # 素材圖片放大倍率
        self.faceImg = faceImg  # 人臉辨識圖片
        self.landmarks = landmarks  # 臉部特徵標記位置

    # 素材圖片寬度=素材放大倍率*辨識臉部的寬度
    def materialImgWidth(self):
        faceWidth = abs(self.landmarks.part(self.widthMarks[0]).x - self.landmarks.part(self.widthMarks[1]).x)
        return self.materialImgSize * faceWidth

    # 素材圖片高度=素材放大倍率*辨識臉部的高度
    def materialImgHeight(self):
        faceHeight = abs(self.landmarks.part(self.heightMarks[0]).y - self.landmarks.part(self.heightMarks[1]).y)
        return self.materialImgSize * faceHeight

    # 素材圖片放置的x範圍
    def putXrange(self):
        putX = self.landmarks.part(self.putMark).x
        return [putX, putX + self.materialImgWidth()]

    # 素材圖片放置的y範圍
    def putYrange(self):
        putY = self.landmarks.part(self.putMark).y
        return [putY, putY + self.materialImgHeight()]

    # 調整素材圖片大小
    def resizeFeatureImg(self):
        print('width:', str(self.materialImgWidth()))
        print('height:', str(self.materialImgHeight()))
        width = self.materialImgWidth()
        height = self.materialImgHeight()
        if width == 0 or height == 0:
            return None
        else:
            return cv2.resize(self.materialImg, (width, height))

    # 將素材圖片的透明部分用人臉辨識圖片來取代
    def synImage(self):

        # 將素材圖片透明部分對應的人臉辨識圖片reverse mask出來
        faceImgMask = self.faceImg[self.putYrange()[0]:self.putYrange()[1], self.putXrange()[0]:self.putXrange()[1]]

        # 將素材圖片透明部分遮住
        mask = self.resizeFeatureImg()

        # 若resizeFeatureImg()返回None 則不進行合成直接返回原圖
        if mask is None:
            return faceImgMask

        mask = mask[:, :, 3] > 125
        mask = mask.astype('uint8')
        materialImgMask = self.resizeFeatureImg()[:, :, 0:3]
        materialImgMask = cv2.bitwise_and(materialImgMask, materialImgMask, mask=mask)

        # 判斷人臉辨識圖片和素材圖片的shape是否相同 若不相同則不進行合成直接返回原圖(防止人臉切到時形狀不一致)
        if faceImgMask.shape[0] == mask.shape[0] and faceImgMask.shape[1] == mask.shape[1]:

            faceImgMask = cv2.bitwise_and(faceImgMask, faceImgMask, mask=1 - mask)

            # 將mask的素材圖片與mask的人臉辨識圖片結合起來
            return cv2.add(materialImgMask, faceImgMask)

        else:
            return faceImgMask


# 讀取人臉偵測模型
detector = dlib.get_frontal_face_detector()

# 讀取人臉特徵捕捉模型
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 讀取墨鏡素材圖片
sunglassesImg = cv2.imread("./images/sunglasses.png", cv2.IMREAD_UNCHANGED)
sunglassesImg[:, :, 3][sunglassesImg[:, :, 3] <= 125] = 0

# 讀取香菸素材圖片
cigaretteImg = cv2.imread("./images/cigarette.png", cv2.IMREAD_UNCHANGED)
cigaretteImg[:, :, 3][cigaretteImg[:, :, 3] <= 125] = 0

# 打開電腦攝像鏡頭
cap = cv2.VideoCapture(0)

while True:

    # 讀入目前攝像鏡頭畫面
    ret, frame = cap.read()

    # 轉為灰階圖片
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 依灰階照片內容進行人臉偵測模型
    faces = detector(gray_frame)

    # 針對每張人臉做特徵捕捉 並放置墨鏡及香菸照片
    for face in faces:
        # 臉部特徵標記位置
        landmarks = predictor(image=gray_frame, box=face)

        # 建立墨鏡素材照片物件
        iSunglassesImg = MaterialImg(materialImg=sunglassesImg,  # 素材圖片
                                     putMark=17,  # 圖片放置對應的人臉辨識點
                                     widthMarks=[16, 0],  # 計算臉部相對位置寬度(用於圖片縮放比例)
                                     heightMarks=[17, 1],  # 計算臉部相對位置高度(用於圖片縮放比例)
                                     materialImgSize=1,  # 素材圖片放大倍率
                                     faceImg=frame,  # 人臉辨識圖片
                                     landmarks=landmarks)  # 臉部特徵標記位置

        # 建立香菸素材照片物件
        iCigaretteImg = MaterialImg(materialImg=cigaretteImg,  # 素材圖片
                                    putMark=65,  # 圖片放置對應的人臉辨識點
                                    widthMarks=[65, 10],  # 計算臉部相對位置寬度(用於圖片縮放比例)
                                    heightMarks=[65, 10],  # 計算臉部相對位置高度(用於圖片縮放比例)
                                    materialImgSize=2,  # 素材圖片放大倍率
                                    faceImg=frame,  # 人臉辨識圖片
                                    landmarks=landmarks)  # 臉部特徵標記位置

        # 加入墨鏡素材
        frame[iSunglassesImg.putYrange()[0]:iSunglassesImg.putYrange()[1],
        iSunglassesImg.putXrange()[0]:iSunglassesImg.putXrange()[1]] = iSunglassesImg.synImage()

        # 加入香菸素材
        frame[iCigaretteImg.putYrange()[0]:iCigaretteImg.putYrange()[1],
        iCigaretteImg.putXrange()[0]:iCigaretteImg.putXrange()[1]] = iCigaretteImg.synImage()

    # 輸出圖片
    cv2.imshow("frame", frame)

    # 按ESC離開
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()