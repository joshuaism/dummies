import cv2 as cv

img = cv.imread('datasets_and_media/trees.jpg')
key = ord("r")

# BGR Pixel 40,40
print(img[40, 40, 0:3])
# convert to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# RGB Pixel 40,40
print(img[40, 40, 0:3])

img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# HSV Pixel 40,40
print(img[40, 40, 0:3])
# Grayscale
img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img[40, 40])

while key != ord("s") and key != ord("S"):
    cv.imshow("Tree Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()
