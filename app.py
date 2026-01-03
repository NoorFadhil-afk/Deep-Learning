import subprocess
import sys
import os
import numpy as np

# ===============================
# 1️⃣ Install required libraries
# ===============================
required_packages = [
    "ultralytics>=8.0",
    "torch>=2.0",
    "torchvision",
    "torchaudio",
    "gradio>=6.2",
    "Pillow",
    "numpy",
    "opencv-python",
    "matplotlib",
    "scipy",
    "pandas"
]

for package in required_packages:
    try:
        __import__(package.split('>=')[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ===============================
# 2️⃣ Import libraries
# ===============================
from ultralytics import YOLO
import gradio as gr
from PIL import Image

# ===============================
# 3️⃣ Load YOLO model
# ===============================
model = YOLO("best_3.pt")  # Make sure the model file is in the same folder

# ===============================
# 4️⃣ Detection function
# ===============================
def detect_helmet(img):
    """
    Detect helmets and draw bounding boxes.
    """
    results = model(img)
    img_with_boxes = results[0].plot()
    # Combine original and result image side by side
    combined = np.concatenate((np.array(img), img_with_boxes), axis=1)
    return Image.fromarray(combined)

# ===============================
# 5️⃣ Load example images (optional)
# ===============================
example_images = []
examples_dir = "examples"
if os.path.exists(examples_dir):
    for file in os.listdir(examples_dir):
        if file.lower().endswith((".jpg", ".png")):
            example_images.append([os.path.join(examples_dir, file)])

# ===============================
# 6️⃣ Create Gradio interface
# ===============================
iface = gr.Interface(
    fn=detect_helmet,
    inputs=gr.Image(type="pil", label="Upload workers image"),
    outputs=gr.Image(type="pil", label="Result: Detected Helmets"),
    title="🦺 Safety Helmet Detection App",
    description="""
Upload an image to automatically detect helmets using a trained YOLO model.  
- Original and detected images displayed side by side  
- Colored bounding boxes around detected helmets  
- Supports .jpg and .png images  
- Easy-to-use interface
""",
    examples=example_images
)

# ===============================
# 7️⃣ Launch the app
# ===============================
iface.launch(share=True, theme="soft")
