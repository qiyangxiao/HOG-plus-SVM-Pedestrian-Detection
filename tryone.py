from utils import slide, cv2NMS, boxOut
import cv2

imgpath = 'test.jpg'
modelpath = '.\\model\\svc-rbf-232f03.pkl'
outpath = 'test_boxed.jpg'

window_list, _, prob_list = slide(modelpath, imgpath, min_width=64)
result = cv2NMS(window_list, prob_list, score_threshold=0.99, nms_threshold=0.3)
boxed_img = boxOut(imgpath, outpath, result)
cv2.imshow('image', boxed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()