import cv2
import cudasift as cs
import numpy as np

img1 = cv2.imread("img1.png")
img2 = cv2.imread("img2.png")

print("img1:",img1.shape)
print("img2:",img2.shape)

#cpu sift
if 1:
    sift = cv2.xfeatures2d.SIFT_create(int(2000))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    keypoints1,descriptor1 = sift.detectAndCompute(gray1, None)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    keypoints2,descriptor2 = sift.detectAndCompute(gray1, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor1, descriptor1, k=2)
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)
    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    #  print("goodmatch:",goodMatch[:20])

    img_out = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, goodMatch, None, flags=2)
    cv2.imwrite("cpu_match.jpg",img_out)
    feature = descriptor1[0,:].reshape(16,8)

# cuda sift
print("+++++++++++++++++++cuda++++++++++++++++++++++++")
def to_cvKeyPoints(keypoints_1):
    cv_kps = []
    keypoints_1 = keypoints_1[['xpos', 'ypos', 'scale', 'sharpness', 'edgeness', 'orientation', 'score', 'ambiguity']]
 
    for i in range(len(keypoints_1)):
        x = keypoints_1['xpos'][i]
        y = keypoints_1['ypos'][i]
        size = keypoints_1['scale'][i]
        kp = cv2.KeyPoint(x,y,size)
        cv_kps.append(kp) 
    return cv_kps

if 1:
    array = img1.copy()
    gray1 = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    nfeatures = int(max(gray1.shape) / 1.25)
    siftdata1 = cs.PySiftData(nfeatures)
    ret = cs.ExtractKeypoints(gray1, siftdata1, thresh=2,upScale=True)
    #  cs.PyPrintSiftData(siftdata1)# 打印信息
    keypoints_1, descriptors_1 = siftdata1.to_data_frame()
    descriptors_1 = (descriptors_1 * 255).astype("float32")
    keypoints_1 = to_cvKeyPoints(keypoints_1)
    array = img2.copy()
    #  print("array:",array.shape)
    gray2 = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    nfeatures = int(max(gray2.shape) / 1.25)
    siftdata2 = cs.PySiftData(nfeatures)
    ret = cs.ExtractKeypoints(gray1, siftdata2, thresh=2,upScale=True)
    keypoints_2, descriptors_2 = siftdata2.to_data_frame()
    descriptors_2 = (descriptors_2 * 255).astype("float32")
    keypoints_2 = to_cvKeyPoints(keypoints_2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)
    goodMatch = np.expand_dims(goodMatch, 1)
    #  print("goodmatch:",goodMatch[:20])
    
    img_out_gpu = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, goodMatch, None, flags=2)
    cv2.imwrite("gpu_match.jpg",img_out_gpu)










