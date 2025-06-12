import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
from src.frftProcess import apply2d_frft_separable, mseCalculation

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master=master, borderwidth=16)
        self.pack(fill=tk.BOTH, expand=True)
        self.master.title('Image Processor - Fractional Fourier Transform')

        self.original_image_array = None
        self.original_photo = None
        self.transformed_photo = None
        self.reconstructed_photo = None

        self.filter_var = tk.StringVar()
        self.filter_var.set(None)

        self.widgets()

    def widgets(self):
        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=0)
        self.master.grid_rowconfigure(3, weight=1)

        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

        image_frame = tk.Frame(self, borderwidth=2, relief="groove")
        image_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        image_frame.grid_columnconfigure(2, weight=1)
        image_frame.grid_rowconfigure(1, weight=1)

        tk.Label(image_frame, text='Original Image').grid(row=0, column=0, pady=(5,0))
        tk.Label(image_frame, text='Transformed (Real Part)').grid(row=0, column=1, pady=(5,0))
        tk.Label(image_frame, text='Reconstructed (Real Part)').grid(row=0, column=2, pady=(5,0))

        self.original_canvas = tk.Canvas(image_frame, bg='lightgray', width=200, height=200)
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.transformed_canvas = tk.Canvas(image_frame, bg='lightgray', width=200, height=200)
        self.transformed_canvas.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.reconstructed_canvas = tk.Canvas(image_frame, bg='lightgray', width=200, height=200)
        self.reconstructed_canvas.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

        button_frame = tk.Frame(self)
        button_frame.grid(row=0, column=0, columnspan=3, pady=10)

        self.inputImage_btn = tk.Button(button_frame, text='Load Image', command=self.inputImage)
        self.inputImage_btn.pack(side=tk.LEFT, padx=10)

        self.process_btn = tk.Button(button_frame, text='Process Image', command=self.process_frft_and_mse, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=10)

        radio_frame = tk.Frame(self)
        radio_frame.grid(row=0, column=0, columnspan=1, sticky='nw', padx=10)

        self.radio_btn = tk.Radiobutton(radio_frame, text='B & W', variable=self.filter_var, value='bw', command=self.convert_bw)
        self.radio_btn.pack(side=tk.LEFT, pady=20)

        self.radio_btn = tk.Radiobutton(radio_frame, text='Sharpen', variable=self.filter_var, value='sharpen', command=self.sharpen_image)
        self.radio_btn.pack(side=tk.LEFT, pady=20)

        self.radio_btn = tk.Radiobutton(radio_frame, text='Blur', variable=self.filter_var, value='blur', command=self.blur_image)
        self.radio_btn.pack(side=tk.LEFT, pady=20)

        tk.Frame(self, height=10).grid(row=2, column=0, columnspan=3)

        mse_frame = tk.Frame(self, borderwidth=2, relief="groove")
        mse_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        mse_frame.grid_columnconfigure(0, weight=1)
        mse_frame.grid_columnconfigure(1, weight=1)
        mse_frame.grid_rowconfigure(1, weight=1)

        tk.Label(mse_frame, text='MSE Table (Varying Phase 1 Order)').grid(row=0, column=0, pady=(5,0))
        tk.Label(mse_frame, text='MSE Table (Varying Phase 2 Order)').grid(row=0, column=1, pady=(5,0))

        self.mse_text1 = scrolledtext.ScrolledText(mse_frame, width=40, height=15, wrap=tk.WORD)
        self.mse_text1.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.mse_text2 = scrolledtext.ScrolledText(mse_frame, width=40, height=15, wrap=tk.WORD)
        self.mse_text2.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    def display_image_on_canvas(self, image_array, canvas):
        if image_array is None:
            canvas.delete("all")
            return

        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 200
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 200

        # Get original image dimensions
        img_width, img_height = image_array.shape[:2]

        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        img_resize = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(img_pil)

        canvas.delete("all")
        x_center = (canvas_width - new_width) / 2
        y_center = (canvas_height - new_height) / 2
        canvas.create_image(x_center, y_center, anchor=tk.NW, image=photo)
        canvas.image = photo

    def inputImage(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            image_cv = cv2.imread(file_path)
            self.original_image_array = image_cv.copy().astype(np.float64)
            self.display_image_on_canvas(image_cv, self.original_canvas)

            self.process_btn.config(state=tk.NORMAL)

            self.transformed_canvas.delete("all")
            self.reconstructed_canvas.delete("all")
            self.mse_text1.delete('1.0', tk.END)
            self.mse_text2.delete('1.0', tk.END)
            self.original_photo = image_cv

        except Exception as e:
            self.original_image_array = None
            self.process_btn.config(state=tk.DISABLED)

    def convert_bw(self):
        if self.original_photo is None:
            return
        gray = cv2.cvtColor(self.original_photo, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image_on_canvas(result, self.original_canvas)

    def sharpen_image(self):
        if self.original_photo is None:
            return
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(self.original_photo, -1, kernel)
        self.display_image_on_canvas(sharpened, self.original_canvas)

    def blur_image(self):
        if self.original_photo is None:
            return
        blurred = cv2.GaussianBlur(self.original_photo, (11, 11), 0)
        self.display_image_on_canvas(blurred, self.original_canvas)

    def process_frft_and_mse(self):
        if self.original_image_array is None:
            return

        self.process_btn.config(state=tk.DISABLED)

        try:
            original_normalized_for_mse = self.original_image_array / 255.0

            self.mse_text1.delete('1.0', tk.END)
            self.mse_text1.insert(tk.END, "Table 1: Computation of mean square error by changing the order of phase 1\n")
            self.mse_text1.insert(tk.END, " (Phase 2 is constant at -0.25)\n")
            self.mse_text1.insert(tk.END, "="*60 + "\n")
            self.mse_text1.insert(tk.END, f"{'Sr. No.':<10} {'Phase 1':<15} {'Phase 2':<15} {'MSE':<15}\n")
            self.mse_text1.insert(tk.END, "-"*60 + "\n")

            phase1_orders_table1 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
            constant_phase2_order = -0.25

            for i, a1_order in enumerate(phase1_orders_table1):
                transformed_img_forward = apply2d_frft_separable(self.original_image_array, a1_order, constant_phase2_order)
                reconstructed_img = apply2d_frft_separable(transformed_img_forward, -constant_phase2_order, -a1_order)

                mse_val = mseCalculation(original_normalized_for_mse, np.real(reconstructed_img))
                self.mse_text1.insert(tk.END, f"{i+1:<10} {a1_order:<15.2f} {constant_phase2_order:<15.2f} {mse_val:<15.4f}\n")
            self.mse_text1.insert(tk.END, "="*60 + "\n")

            self.mse_text2.delete('1.0', tk.END)
            self.mse_text2.insert(tk.END, "Table 2: Computation of mean square error by changing the order of phase 2\n")
            self.mse_text2.insert(tk.END, " (Phase 1 is constant at 0.25)\n")
            self.mse_text2.insert(tk.END, "="*60 + "\n")
            self.mse_text2.insert(tk.END, f"{'Sr. No.':<10} {'Phase 1':<15} {'Phase 2':<15} {'MSE':<15}\n")
            self.mse_text2.insert(tk.END, "-"*60 + "\n")

            constant_phase1_order = 0.25
            phase2_orders_table2 = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.50, -0.60, -0.70, -0.80, -0.90]

            for i, a2_order in enumerate(phase2_orders_table2):
                transformed_img_forward = apply2d_frft_separable(self.original_image_array, constant_phase1_order, a2_order)
                reconstructed_img = apply2d_frft_separable(transformed_img_forward, -a2_order, -constant_phase1_order)

                mse_val = mseCalculation(original_normalized_for_mse, np.real(reconstructed_img))
                self.mse_text2.insert(tk.END, f"{i+1:<10} {constant_phase1_order:<15.2f} {a2_order:<15.2f} {mse_val:<15.4f}\n")
            self.mse_text2.insert(tk.END, "="*60 + "\n")

            display_a1_order = 0.25
            display_a2_order = -0.25
            transformed_for_display = apply2d_frft_separable(self.original_image_array, display_a1_order, display_a2_order)
            reconstructed_for_display = apply2d_frft_separable(transformed_for_display, -display_a2_order, -display_a1_order)

            # Apply slight color shift to the reconstructed image for display
            reconstructed_for_display_colored = np.real(reconstructed_for_display).copy()
            if reconstructed_for_display_colored.ndim == 3:
                # Example: Adjust color balance. Values are in [0,1] after np.real,
                # then multiplied by 255 for display. So adjust in [0,1] here.
                reconstructed_for_display_colored[:, :, 0] = np.clip(reconstructed_for_display_colored[:, :, 0] * 1.1, 0, 1) # Red boost
                reconstructed_for_display_colored[:, :, 1] = np.clip(reconstructed_for_display_colored[:, :, 1] * 0.9, 0, 1) # Green reduction
                reconstructed_for_display_colored[:, :, 2] = np.clip(reconstructed_for_display_colored[:, :, 2] * 1.05, 0, 1) # Blue slight boost

            self.display_image_on_canvas(np.real(transformed_for_display), self.transformed_canvas)
            self.display_image_on_canvas(reconstructed_for_display_colored, self.reconstructed_canvas)

        except Exception as e:
            pass
        finally:
            self.process_btn.config(state=tk.NORMAL)

if __name__ == '__main__':
    root = tk.Tk()
    screenWidth = root.winfo_screenwidth()
    screenHeight = root.winfo_screenheight()
    appX = 900
    appY = 700
    x = (screenWidth / 2) - (appX / 2)
    y = (screenHeight / 2) - (appY / 2)

    app = Application(master=root)
    app.master.geometry(f"{appX}x{appY}+{int(x)}+{int(y)}")
    app.master.mainloop()
