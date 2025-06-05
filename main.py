from tkinter import *
from tkinter.ttk import *
import tkinter as tk
import numpy as np


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master=None, borderwidth=24)
        pass
       




# Main loop
root = tk.Tk()
screenWidth = root.winfo_screenwidth()
screenHeight = root.winfo_screenheight()
appX = 860
appY = 560
x = (screenWidth/2) - (appX/2)
y = (screenHeight/2) -(appY/2)

app = Application()
app.master.geometry(f"{appX}x{appY}+{int(x)}+{int(y)}")
app.master.title('Image Processor - Fast Fourier Transform')
app.master.mainloop()