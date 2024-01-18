from PIL import Image
from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("yolov8m-seg.pt")

# Input image paths
input_image_path = "D:/Users/Mathis/Documents/ISEN/2023-2024/Segmentation image/images/input.png"
background_image_path = "D:/Users/Mathis/Documents/ISEN/2023-2024/Segmentation image/images/background.jpg"
output_image_path = "D:/Users/Mathis/Documents/ISEN/2023-2024/Segmentation image/images/output.png"

# Load the input and background images
input_img = Image.open(input_image_path)
background_img = Image.open(background_image_path)

# Ensure both images have the same mode (convert input image to RGB)
input_img = input_img.convert("RGBA")

# Resize the input image to match the background image's size
input_img = input_img.resize(background_img.size)

# Perform segmentation
results = model.predict(input_image_path)
result = results[0]

# Get the mask and polygon for the dog
masks = result.masks
mask1 = masks[0]
mask = mask1.data[0].numpy()
polygon = mask1.xy[0]

# Create a mask image
mask_img = Image.fromarray(mask, "I")
# Convert to "L" mode (8-bit pixels, black and white)
mask_img = mask_img.convert("L")

# Resize the mask to match the size of the background image
mask_img = mask_img.resize(background_img.size)

# Create a copy of the input image to apply the mask
dog_region = input_img.copy()

# Apply the mask to the alpha channel of the dog region
dog_region.putalpha(mask_img)

# Paste the dog region onto the background image
output_img = Image.alpha_composite(background_img.convert("RGBA"), dog_region)

# Save the new image with the dog to the specified path
output_img.save(output_image_path)
