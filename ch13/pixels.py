import cv2 as cv

img = cv.imread('datasets_and_media/trees.jpg')
key = ord("r")
print(img.dtype)

# Rows, COlumns, Channels
print(img.shape)

# Blue, Green, Red intensity of pixel 40,40(down)
print(img[40, 40, 0])
print(img[40, 40, 1])
print(img[40, 40, 2])

# Make a black square
img[0:100, 0:100, 0:3] = 0

while key != ord("s") and key != ord("S"):
    cv.imshow("Tree Picture", img)
    key = cv.waitKey()
cv.destroyAllWindows()
