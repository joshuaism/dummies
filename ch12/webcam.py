import cv2 as cv

# 0 integrated camera
# 1 ???
# 2 obs virtual camera
webcam = cv.VideoCapture(0)
img = cv.imread('datasets_and_media/wood.jpg')
key = 'r'
while key != 's' and key != 'S':
    ret, still = webcam.read()
    if not ret:
        cv.imshow("Webcam Feed", img)
    else:
        cv.imshow("Webcam Feed", still)
    key = cv.waitKey(10)
    if key == -1:
        continue
    key = chr(key)
    match key:
        case '0':
            webcam = cv.VideoCapture(0)
        case '1':
            webcam = cv.VideoCapture(1)
        case '2':
            webcam = cv.VideoCapture(2)
cv.destroyAllWindows()
