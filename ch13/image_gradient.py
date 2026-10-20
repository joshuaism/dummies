import cv2 as cv

img = cv.imread('datasets_and_media/trees.jpg')
key = 'r'
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.blur(img, (5, 5))

# Sobel reveals edges

# Sobel derivative - x direction
sobel_x = cv.Sobel(img, -1, 1, 0, ksize=5)

# Sobel derivative - y direction
sobel_y = cv.Sobel(img, -1, 0, 1, ksize=5)

# Sobel derivative - both directions
sobel_both = cv.Sobel(img, -1, 1, 1, ksize=5)

display = img

while key != 's' and key != 'S':
    cv.imshow("Tree Picture", display)
    key = cv.waitKey(10)
    if key == -1:
        continue
    key = chr(key)
    match key:
        case 'o':
            display = img
        case 'x':
            display = sobel_x
        case 'y':
            display = sobel_y
        case 'b':
            display = sobel_both
cv.destroyAllWindows()
