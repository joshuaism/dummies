import cv2 as cv
import os

target_size = 1000000

for path, subdirs, files in os.walk("datasets_and_media/detector"):
    for name in files:
        file = os.path.join(path, name)
        size = os.path.getsize(file)
        if size > 100000:
            fx = target_size / size
            img = cv.imread(file)
            resized_image = cv.resize(
                img, None, fx=fx, fy=fx, interpolation=cv.INTER_LINEAR)
            cv.imwrite(file, resized_image)
