
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np

"""
pre-processing for Training_Data

-前処理: 物体検出が行いやすいように、画像の前処理を行う
-物体検出: 物体の検出を行い、画像から切り出す
  -輪郭検出: 画像上の領域(輪郭)を認識することで、物体を検出する
  -物体認識: OpenCVの学習済みモデルを利用し、対象の物体を認識し、検出を行う
-機械学習の準備: 切り出した画像を用い、予測や学習を行うための準備を行う

"""


def main():
    with open("./path_and_label.txt", "r") as f:
        lines = f.readlines()
        list_line = []
        for line in lines:
            list_line.append(line.split())
    image = to_grayscale(list_line[900][0])
    image1 = binary_threshold(list_line[900][0])
    image = cv2.resize(image, (960, 540))
    image1 = cv2.resize(image1, (960, 540))

    cv2.imshow("show gray image", image)
    cv2.imshow("show thresh_binary", image1)
    #cv2.imshow("show the image option the ", detect_contour(image, 10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_grayscale(path):
    img = cv2.imread(path)
    print(img)
    print("ラベルデータ", "オブジェクト:" + str(type(img)), "データ型: " + str(img.dtype), "N次元配列:" + str(img.shape), sep='\n')
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed


def to_matplotlib_format(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def binary_threshold(path):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    under_thresh = 105
    upper_thresh = 145
    maxValue = 255
    # 背景を落とす
    th, drop_back = cv2.threshold(grayed, under_thresh, maxValue, cv2.THRESH_BINARY)
    # 境界の明確化
    th, clarify_born = cv2.threshold(grayed, upper_thresh, maxValue, cv2.THRESH_BINARY_INV)
    merged = np.minimum(drop_back, clarify_born)
    return merged


def blur(img):
    filtered = cv2.GaussianBlur(img, (11, 11), 0)
    return filtered


def detect_contour(path, min_size):
    contoured = cv2.imread(path)
    forcrop = cv2.imread(path)

    # make binary image
    birds = binary_threshold(path)
    birds = cv2.bitwise_not(birds)

    # detect contour
    im2, contours, hierarchy = cv2.findContours(birds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []

    # draw contour
    for c in contours:
        if cv2.contourArea(c) < min_size:
            continue

        # rectangle area
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = padding_position(x, y, w, h, 5)

        # crop the image
        cropped = forcrop[y:(y+h), x:(x+w)]
        cropped = resize_image(cropped, (210, 2010))
        crops.append(cropped)

        # draw contour
        cv2.drawContours(contoured, c, -1, (0, 0, 255), 3)
        cv2.rectangle(contoured, (x, y), (x+w, y+h), (0, 255, 0), 3) #rectangele

        return contoured, crops


def padding_position(x, y, w, h, p):
    return x - p, y-p, w+p * 2, h+p*2


def resize_image(img, size):
    # size is enough to img
    img_size = img.shape[:2]
    if img_size[0] > size[1] or img_size[1] > size[0]:
        raise Exception("img is larger than size")

    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    # filling
    mask = np.full(size, 255, dtype=np.uint8)
    mask[row:(row + img.shape[0]), col:(col + img.shape[1])] = 0
    filled = cv2.inpaint(resized, mask, 3, cv2.INPAINT_TELEA)

    return filled


def various_contours(path):
    color = cv2.imread(path)
    grayed = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayed, 218, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(binary)
    _, contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 90:
            continue

        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.CHAIN_APPROX_SIMPLE
        cv2.drawContours(color, c, -1, (0, 0, 255), 3)
        cv2.drawContours(color, [approx], -1, (0, 255, 0), 3)

        pass

 #varous_contours(IMG_FOR_CONTOUR)


if __name__ == "__main__":
    main()
