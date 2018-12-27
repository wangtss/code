from tkinter import *
from tkinter import filedialog
from inference import Inference
from PIL import Image, ImageTk
import threading
import numpy as np


class Application:
    def __init__(self, master):
        self.frame = Frame(master)
        self.frame.pack()
        self.inference = Inference()
        self.layout()

    def layout(self):
        self.label_video_path = Label(self.frame, text='video path')
        self.label_video_path.grid(row=0, column=0)
        self.video_path = StringVar()
        self.entry_video_path = Entry(self.frame, textvariable=self.video_path)
        self.entry_video_path.grid(row=0, column=1)

        self.label_width = Label(self.frame, text='width')
        self.label_width.grid(row=1, column=0)
        self.entry_width = Entry(self.frame)
        self.entry_width.grid(row=1, column=1)


        self.label_height = Label(self.frame, text='height')
        self.label_height.grid(row=1, column=2)
        self.entry_height = Entry(self.frame)
        self.entry_height.grid(row=1, column=3)

        self.button_choose = Button(self.frame, text='choose file', command=self.fileLoader)
        self.button_choose.grid(row=0, column=4)

        self.button_process = Button(self.frame, text='  process  ', command=self.process)
        self.button_process.grid(row=1, column=4)

        frame_holder = ImageTk.PhotoImage(Image.fromarray(np.zeros((432, 768), np.uint8)))
        self.label_frame = Label(self.frame, image=frame_holder)
        self.label_frame.image = frame_holder
        self.label_frame.grid(row=2, column=0, columnspan=5)
        self.label_score = Label(self.frame, text='current quality score: ', font=("Arial, 18"))
        self.label_score.grid(row=3, column=0, columnspan=2)
        self.label_val = Label(self.frame, text='0', font=("Arial, 18"), fg='red')
        self.label_val.grid(row=3, column=2)

        self.label_overall = Label(self.frame, text='overall quality score: ', font=("Arial, 18"))
        self.label_overall.grid(row=4, column=0, columnspan=2)
        self.label_overall_val = Label(self.frame, text='0', font=("Arial, 18"), fg='red')
        self.label_overall_val.grid(row=4, column=2)

    def fileLoader(self):
        filename = filedialog.askopenfilename()
        self.video_path.set(filename)

    def playVideo(self):
        for i in range(self.inference.frame_num):
            img = Image.fromarray(self.inference.video[i, :, :])
            img = ImageTk.PhotoImage(img)
            self.label_frame.config(image=img)
            self.label_frame.image = img

            self.label_val.config(text='{:.4f}'.format(self.inference.display_score[i // 64]))
        self.label_overall_val.config(text='{:.4f}'.format(np.mean(self.inference.display_score)))

    def process(self):
        video_path = self.entry_video_path.get()
        width = int(self.entry_width.get())
        height = int(self.entry_height.get())
        print(video_path, width, height)
        self.inference.processing(video_name=video_path, sizes=[width, height])

        print('done')
        tv = threading.Thread(target=self.playVideo)
        tv.start()


root = Tk()
root.title('Video Quality Assessment')
app = Application(root)
root.mainloop()