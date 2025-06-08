import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Image Processor - Fast Fourier Transform')
        self.geometry('980x600')

        self.original_image = None
        self.processed_image = None

        self.create_widgets()

    def create_widgets(self):
        # Upload button (top-left)
        self.upload_btn = tk.Button(self, text='Upload Image', command=self.upload_image)
        self.upload_btn.place(x=10, y=10)

        # Image display labels
        self.original_label = tk.Label(self, text='Original Image')
        self.original_label.place(x=90, y=50)

        self.original_image_label = tk.Label(self, bd=2, relief='solid')
        self.original_image_label.place(x=10, y=80)

        self.processed_label = tk.Label(self, text='Processed Image')
        self.processed_label.place(x=340, y=50)

        self.processed_image_label = tk.Label(self, bd=2, relief='solid')
        self.processed_image_label.place(x=280, y=80)

        # Bottom control frame
        self.control_frame = tk.Frame(self)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.bw_btn = tk.Button(self.control_frame, text='Black & White', command=self.convert_bw)
        self.bw_btn.pack(side=tk.LEFT, padx=10)

        self.sharp_btn = tk.Button(self.control_frame, text='Sharpen', command=self.sharpen_image)
        self.sharp_btn.pack(side=tk.LEFT, padx=10)

        self.blur_btn = tk.Button(self.control_frame, text='Blur', command=self.blur_image)
        self.blur_btn.pack(side=tk.LEFT, padx=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror('Error', 'Failed to load image.')
            return

        self.original_image = img
        self.display_image(img, self.original_image_label)

        # Clear previous processed image
        self.processed_image_label.config(image='')
        self.processed_image_label.image = None

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (250, 300))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)

        label.configure(image=img_tk)
        label.image = img_tk  # Keep reference

    def convert_bw(self):
        if self.original_image is None:
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image(result, self.processed_image_label)

    def sharpen_image(self):
        if self.original_image is None:
            return
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(self.original_image, -1, kernel)
        self.display_image(sharpened, self.processed_image_label)

    def blur_image(self):
        if self.original_image is None:
            return
        blurred = cv2.GaussianBlur(self.original_image, (11, 11), 0)
        self.display_image(blurred, self.processed_image_label)

if __name__ == '__main__':
    app = ImageProcessorApp()
    app.mainloop()