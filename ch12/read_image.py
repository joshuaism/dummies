import cv2 as cv

img = cv.imread('datasets_and_media/wood.jpg')
key = ord("r")
while key != ord("s") and key != ord("S"):
    cv.imshow("Wood Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()
