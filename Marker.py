import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np
import random
import tkinter as tk
from idlelib.tooltip import Hovertip
from tkinter.messagebox import showinfo


def CollectSampleRects(select_contours, image_mask, box_size, threshold_perc):
    threshold = (box_size * box_size * threshold_perc) / 100
    sample_rects = []
    for cnt in select_contours:
        x, y, w, h = cv.boundingRect(cnt)
        for j in range(x, x + w, box_size):
            for k in range(y, y + h, box_size):
                roi = image_mask[k:k + box_size, j:j + box_size]
                cont = cv.countNonZero(roi)
                if (cont > threshold):
                    rect = [j, k, j + box_size, k + box_size]
                    sample_rects.append(rect)

    return sample_rects

def SaveSamplesFromImg(samples_folder, samples_array, image, cnt_all, fh, fv):
    for i, s in enumerate(samples_array):
        sample = image[s[1]:s[3], s[0]:s[2]]
        name_sample = samples_folder + str(i + 1 + cnt_all) + ".jpg"
        cv.imwrite(name_sample, sample)
        name_sample = ""
        if fh.get():
            name_sample = samples_folder + str(i + 1 + cnt_all + len(samples_array) * fh.get()) + ".jpg"
            sample_fh = cv.flip(sample, 1)
            cv.imwrite(name_sample, sample_fh)
            name_sample = ""
        if fv.get():
            name_sample = samples_folder + str(i + 1 + cnt_all + len(samples_array) * (fh.get() + 1)) + ".jpg"
            sample_fv = cv.flip(sample, 0)
            cv.imwrite(name_sample, sample_fv)
            name_sample = ""



class App(tk.Tk):
    info_text = ""
    save_dir = ""
    default_class_name = "class_1"
    sep = "\n" + "-" * 30 + "\n"

    orig_img = np.zeros((350, 500, 3), dtype=np.uint8)
    scale = 1
    points = []
    cnt_samples_train = 0
    cnt_samples_test = 0

    colors = ["orange red", "SpringGreen2", "dodger blue", "yellow2", "purple1", "cyan"]
    default_color = colors[0]

    hello_info = "Порядо работы с данным ПО:\n" \
                 "1. Выбрать и открыть изображение: Файл -> Открыть изображение\n" \
                 "2. Выбрать каталог сохранения образцов: нажать на кнопку 'Выбрать папку для сохранения'\n" \
                 "3. Ввести название класса в соответствующее поле и нажать 'Принять'\n" \
                 "4. Выбрать вариант увеличения кол-ва образцов:\n" \
                 "  ОГ - отразить по горизонтали\n" \
                 "  ОВ - отразить по вертикали\n" \
                 "  БО - без отражения\n" \
                 "5. С помощтю ЛКМ выделить интересующую область на изображении\n" \
                 "6. Нажать на кнопку 'Разметить область'. ПО автоматически 'разрежет' выделенную облать на фрагменты" \
                 " и сохранит в выбранном ранее каталоге в подпапках train/название_класса и test/название_класса"

    def resize(self, event):
        region = self.canvas.bbox(tk.ALL)
        self.canvas.configure(width=region[2], height=region[3])

    def load_image(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        filename = fd.askopenfilename(title="Выбор изображения", filetypes=[
                    ("JPEG", ".jpg .jpeg"),
                    ("PNG", ".png"),
                    ("TIFF", ".tiff"),
                ])
        if filename:
            img = cv.imread(filename)
            self.orig_img = img.copy()
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            WS = 1
            HS = 1
            if img.shape[1] > screen_width:
                WS = round(1600 / img.shape[1], 2)
            if img.shape[0] > screen_height:
                HS = round(1000 / img.shape[0], 2)

            if WS > HS:
                img = cv.resize(img, (0, 0), fx=HS, fy=HS)
                self.scale = HS
            else:
                img = cv.resize(img, (0, 0), fx=WS, fy=WS)
                self.scale = WS

            self.image = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=NW, image=self.image)

            self.state('zoomed')
            self.bnt_select_sdir["state"] = ACTIVE
            self.info.delete("1.0", END)

    def add_point(self, event):
        if self.save_dir:
            point = [event.x, event.y]
            self.points.append(point)
            self.info_text = str(point)
            self.info.insert(INSERT, self.info_text)
            self.canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill=self.default_color)

    def save_area(self):
        #canvas_points = self.points
        self.canvas.create_polygon(self.points, fill=self.default_color, outline="#004D40")
        for point in self.points:
            point[0] = int(point[0] / self.scale)
            point[1] = int(point[1] / self.scale)

        #print(self.points)
        pts = np.array(self.points)
        class_mask = np.zeros((self.orig_img.shape[1], self.orig_img.shape[1], 1), dtype=np.uint8)
        cv.fillPoly(class_mask, np.int32([self.points]), (255))
        contours, hierarchy = cv.findContours(class_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        class_rects = CollectSampleRects(contours, class_mask, 32, 75)
        random.shuffle(class_rects)
        train_samples = class_rects[:int((len(class_rects) + 1) * 0.75)]
        test_samples = class_rects[int((len(class_rects) + 1) * 0.75):]

        class_train_dir = os.path.join(self.save_dir, "train", self.default_class_name, "")
        class_test_dir = os.path.join(self.save_dir, "test", self.default_class_name, "")

        if not os.path.exists(class_train_dir):
            os.makedirs(class_train_dir)
            self.cnt_samples_train = 0
        else:
            self.cnt_samples_train = len(os.listdir(class_train_dir))
        SaveSamplesFromImg(class_train_dir, train_samples, self.orig_img, self.cnt_samples_train, self.flip_H, self.flip_V)
        #self.cnt_samples_train += cnt_train#len(train_samples)

        if not os.path.exists(class_test_dir):
            os.makedirs(class_test_dir)
            self.cnt_samples_test = 0
        else:
            self.cnt_samples_test = len(os.listdir(class_test_dir))
        SaveSamplesFromImg(class_test_dir, test_samples, self.orig_img, self.cnt_samples_test, self.flip_H, self.flip_V)
        #self.cnt_samples_test += cnt_test#len(train_samples)

        self.points.clear()
        self.info.insert(INSERT, self.sep)
        self.class_name["state"] = ACTIVE

    def select_save_dir(self):
        #self.info.delete("1.0", END)
        self.save_dir = fd.askdirectory()

        self.info.insert(INSERT, "Папка для образцов:\n" + self.save_dir + self.sep)
        self.bnt_select_sdir["state"] = DISABLED
        self.class_name["state"] = ACTIVE
        self.bnt_ok["state"] = ACTIVE
        self.bnt_save_area["state"] = ACTIVE
        self.flip_check_H["state"] = ACTIVE
        self.flip_check_V["state"] = ACTIVE
        self.flip_check_none["state"] = ACTIVE

    def select_class_name(self):
        class_name = self.class_name.get()
        if class_name != "Введите название класса":
            self.default_class_name = class_name
        self.class_name["state"] = DISABLED

    def change_color(self):
        self.default_color = self.selected_color.get()
        print(self.default_color)

    def select_flip(self):
        if self.flip_H.get() or self.flip_V.get():
            self.flip_none.set(0)
    def cancel_flip(self):
        if self.flip_none.get():
            self.flip_H.set(0)
            self.flip_V.set(0)

    def open_info(self):
        message = "Федеральное государственное бюджетное учреждение науки Институт программных " \
                  "систем им. А.К. Айламазяна Российской академии наук (ИПС им. А.К. Айламазяна РАН)\n\n" \
                  "Лаборатория методов обработки и анализа изображений (ЛМОАИ)\n\n" \
                  "2023 год"
        showinfo(title="Разработчик", message=message)

    def __init__(self):
        super().__init__()

        self.title("Marker")

        x = self.winfo_screenwidth() / 2 - self.winfo_reqwidth()
        y = self.winfo_screenheight() / 2 - self.winfo_reqheight()
        self.wm_geometry("+%d+%d" % (x, y))

        self.rowconfigure(0, weight=1)
        #self.columnconfigure(1, weight=2)

        #self.scroll_x = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        #self.scroll_y = tk.Scrollbar(self, orient=tk.VERTICAL)
        #self.canvas = tk.Canvas(self, width=int(self.winfo_screenwidth()/2), height=int(self.winfo_screenheight()/2),
                                #xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.canvas = tk.Canvas(self, width=int(self.winfo_screenwidth() / 2),
                                height=int(self.winfo_screenheight() / 2))

        #self.scroll_x.config(command=self.canvas.xview)
        #self.scroll_y.config(command=self.canvas.yview)

        self.frame = tk.Frame(self.canvas)
        self.viewer = tk.Label(text="info", background="white")

        self.menu = Menu()
        self.file_menu = Menu(tearoff=0)
        self.file_menu.add_command(label="Открыть изображение", command=self.load_image)
        self.info_menu = Menu(tearoff=0)
        self.info_menu.add_command(label="Разработчик", command=self.open_info)
        self.menu.add_cascade(label="Файл", menu=self.file_menu)
        self.menu.add_cascade(label="О программе", menu=self.info_menu)
        self.config(menu=self.menu)

        self.bnt_select_sdir = ttk.Button(text="Выбрать папку для сохранения", width=40, command=self.select_save_dir)
        self.bnt_select_sdir.grid(row=0, column=1, sticky="nw", padx=5)

        self.class_name = ttk.Entry(width=30, justify=CENTER)
        self.class_name.insert(INSERT, "Введите название класса")
        self.class_name.grid(row=0, column=1, sticky="nw", padx=5, pady=30)
        self.class_name["state"] = DISABLED

        self.bnt_ok = ttk.Button(text="Принять", width=8, command=self.select_class_name)
        self.bnt_ok.grid(row=0, column=1, sticky="nw", padx=(197, 0), pady=30)
        self.bnt_ok["state"] = DISABLED

        flips = ["ОГ", "ОВ", "БО"]
        tips = ["Отразить образцы по горизонтали", "Отразить образцы по вертикали", "Без отражения"]
        pad = 5
        self.flip_H = IntVar()
        self.flip_check_H = ttk.Checkbutton(text=flips[0], variable=self.flip_H, onvalue=1, offvalue=0,
                                            command=self.select_flip, state=DISABLED)
        self.flip_check_H.grid(row=0, column=1, sticky="nw", padx=5, pady=60)
        Hovertip(self.flip_check_H, tips[0])
        self.flip_V = IntVar()
        self.flip_check_V = ttk.Checkbutton(text=flips[1], variable=self.flip_V, onvalue=1, offvalue=0,
                                            command=self.select_flip, state=DISABLED)
        self.flip_check_V.grid(row=0, column=1, sticky="nw", padx=55, pady=60)
        Hovertip(self.flip_check_V, tips[1])
        self.flip_none = IntVar()
        self.flip_none.set(1)
        self.flip_check_none = ttk.Checkbutton(text=flips[2], variable=self.flip_none, onvalue=1, offvalue=0,
                                               command=self.cancel_flip, state=DISABLED)
        self.flip_check_none.grid(row=0, column=1, sticky="nw", padx=105, pady=60)
        Hovertip(self.flip_check_none, tips[2])


        self.bnt_save_area = ttk.Button(text="Разметить область", width=40, command=self.save_area)
        self.bnt_save_area.grid(row=0, column=1, sticky="nw", padx=5, pady=90)
        self.bnt_save_area["state"] = DISABLED

        self.selected_color = StringVar(value="orange red")
        for pad, color in enumerate(self.colors):
            self.color_btn = Radiobutton(selectcolor=color, fg="white", value=color, variable=self.selected_color,
                                         command=self.change_color)
            self.color_btn.grid(row=0, column=3, sticky="nw", pady=pad*25)

        self.info = Text(background="white", width=30, wrap="word")
        self.info.insert(INSERT, self.hello_info)
        self.info.grid(row=0, column=1, sticky="nsew", padx=5, pady=(120, 5))

        self.scroll_y = tk.Scrollbar(self, orient=VERTICAL, command = self.info.yview)
        self.scroll_y.grid(row=0, column=2, sticky=NS)
        self.info["yscrollcommand"] = self.scroll_y.set

        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nswe")
        self.canvas.config(cursor="tcross")
        #self.scroll_x.grid(row=1, column=0, sticky="we")
        #self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.bind("<Configure>", self.resize)
        self.update_idletasks()
        #self.minsize(self.winfo_width(), self.winfo_height())

        self.canvas.bind("<Button-1>", self.add_point)

if __name__ == "__main__":
    app = App()
    app.mainloop()