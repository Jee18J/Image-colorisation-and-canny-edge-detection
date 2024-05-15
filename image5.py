import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk

# Define directories and model paths
DIR = r"C:\Users\jeevi\OneDrive\Desktop\.vs\image"
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")

# Load the deep learning model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Configure model specific layers
class8 = net.getLayerId("class8_ab")#for predicting ab
conv8 = net.getLayerId("conv8_313_rh")#mapping output to colors
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Global variables to store colored and enhanced images
colorized = None
enhanced = None

# Function to handle image selection and processing
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        colorize_image(file_path)

def colorize_image(image_path):
    global customFont  # Make customFont global
    global colorized  # Make colorized global
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)#Converts the image from the BGR color space to the LAB color space using OpenCV's cvtColor() function.

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50#This can help in normalizing the data and bringing it into a more manageable range for further processing.

    print("Coloring the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Adjust image size to fit the display
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    colorized_image = Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
    colorized_image = colorized_image.resize((screen_width // 2, screen_height // 2), Image.Resampling.LANCZOS)
    colorized_photo = ImageTk.PhotoImage(colorized_image)

    display_label.config(image=colorized_photo)
    display_label.image = colorized_photo

    # Add Enhance button
    enhance_btn = tk.Button(root, text="Enhance", command=lambda: enhance_image(), font=customFont,
                            bg='#80c1ff', fg='white', padx=10, pady=5, relief=tk.FLAT)
    enhance_btn.pack(pady=10)

def enhance_image():
    global colorized, enhanced  # Access global colorized and enhanced variables
    
    # Apply image enhancement techniques (e.g., sharpening and contrast stretching)
    enhanced_image = cv2.GaussianBlur(colorized, (0, 0), 3)
    enhanced_image = cv2.addWeighted(colorized, 1.5, enhanced_image, -0.5, 0)
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.2, beta=0)

    # Edge Detection using Canny
    edges = cv2.Canny(enhanced_image, 100, 200)

    # Resize images to fit the screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    enhanced_resized = cv2.resize(enhanced_image, (screen_width // 2, screen_height // 2))
    edges_resized = cv2.resize(edges, (screen_width // 2, screen_height // 2))

    # Convert images for display
    enhanced_display = Image.fromarray(cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2RGB))
    edges_display = Image.fromarray(edges_resized)

    enhanced_photo = ImageTk.PhotoImage(enhanced_display)
    edges_photo = ImageTk.PhotoImage(edges_display)

    # Create a new window to display enhanced images
    enhanced_window = tk.Toplevel(root)
    enhanced_window.title("Enhanced Images")

    # Display enhanced image
    enhanced_label = tk.Label(enhanced_window, image=enhanced_photo)
    enhanced_label.image = enhanced_photo
    enhanced_label.pack(side="left", padx=10, pady=10)

    # Display edge-detected image
    edges_label = tk.Label(enhanced_window, image=edges_photo)
    edges_label.image = edges_photo
    edges_label.pack(side="right", padx=10, pady=10)


# Main function to construct the GUI
def main():
    global root
    root = tk.Tk()
    root.title("Image Colorization GUI")
    global customFont
    customFont = font.Font(family="Helvetica", size=12, weight="bold")

    heading_label = tk.Label(root, text="Image Colorisation", font=font.Font(family="Helvetica", size=16, weight="bold"), bg='#f0f0f0')
    heading_label.pack(pady=10)

    btn = tk.Button(root, text="Load Image", command=select_image, font=customFont, bg='#80c1ff', fg='white', padx=10, pady=5, relief=tk.FLAT)
    btn.pack(pady=20)

    global display_label
    display_label = tk.Label(root)
    display_label.pack(pady=20)

    # Set the GUI background color
    root.configure(bg='#f0f0f0')
    root.mainloop()

if __name__ == "__main__":
    main()






