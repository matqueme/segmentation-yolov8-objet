from PIL import ImageDraw
from ultralytics import YOLO
from PIL import Image
import os

# Load YOLO segmentation model
model = YOLO("yolov8m-seg.pt")

image_input = "D:/Users/Mathis/Documents/ISEN/2023-2024/Segmentation image/images/input.png"
results = model.predict(image_input)

result = results[0]

masks = result.masks
len(masks)

mask1 = masks[0]

mask = mask1.data[0].numpy()
polygon = mask1.xy[0]

mask_img = Image.fromarray(mask, "I")

img = Image.open(image_input)
draw = ImageDraw.Draw(img)
draw.polygon(polygon, outline=(0, 255, 0), width=2)
img.show()
