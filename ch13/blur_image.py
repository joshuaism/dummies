import cv2 as cv

img = cv.imread('datasets_and_media/trees.jpg')
key = ord("r")

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ksize == kernel size???
img = cv.blur(img, (5, 5))

# gaussian blur (normal distribution)
# img = cv.GaussianBlur(img, (5, 5), 3)

# median blur (helps remove "salt and pepper" noise)
# img = cv.medianBlur(img, 5)

while key != ord("s") and key != ord("S"):
    cv.imshow("Tree Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()
