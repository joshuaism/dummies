# run in google colab
from ultralytics import YOLO
from google.colab import drive
import ultralytics
%pip install ultralytics
ultralytics.checks()

# Code Block 2 - Mount google drive and get data loaded in
drive.mount('/content/drive')

# Code Block 3 - Train a YOLO model
# Load a pretrained model
model = YOLO("yolov8n.pt")
results = model.train(
    data="/content/drive/My Drive/laranel_detector/data.yaml", epochs=50)

# Code Block 4 - Evaluate the model
results = model.predict(
    source="/content/drive/My Drive/laranel_detector/images/test/IMG20261019093339.jpg")
results[0].show()

# Code Block 5 - test the model
model.export()
