import cv2 as cv

img = cv.imread('datasets_and_media/trees.jpg')
key = ord('r')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def nothing(pos):
    pass


cv.namedWindow('controls')
cv.createTrackbar("lower", 'controls', 0, 255, nothing)
cv.createTrackbar("upper", 'controls', 0, 255, nothing)
while key != ord("s") and key != ord("S"):
    lower = int(cv.getTrackbarPos("lower", 'controls'))
    upper = int(cv.getTrackbarPos("upper", 'controls'))
    canny_img = cv.Canny(img, lower, upper)
    cv.imshow("Tree Picture", canny_img)
    key = cv.waitKey(5)
cv.destroyAllWindows()
