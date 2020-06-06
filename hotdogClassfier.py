from tkinter import Tk, Label, Button
import mxnet as mx
import gluoncv as gcv
import tkinter as tk
from tkinter import Canvas, filedialog
from PIL import ImageTk, Image
import os

# N C H W
# N for the batch dimension, C for channel, H for height, and W for width.
# print('shape:', image.shape)
# print('data type: ', image.dtype)
# print('min value:', image.min().asscalar())
# print('max value:', image.max().asscalar())


class HotdogGUI:
    def __init__(self, master):
        self.master = master
        master.title("Hotdog Not Hotog")

        button1 = tk.Button(master, text="Select an image",
                            command=lambda: self.clickFunction(master))
        button1.pack()

        # self.filename = tk.StringVar()
        # self.filename.set("No File Selected")
        # l = tk.Label(master, textvariable=self.filename)
        # l.pack()

        photo = 'stock\hotdog.jpg'
        root.photo = ImageTk.PhotoImage(Image.open(photo))
        self.vlabel = tk.Label(root, image=root.photo)
        self.vlabel.pack()

        self.result = tk.StringVar()
        self.result.set("-----------IS IT A HOTDOG?------------")
        ll = tk.Label(master, textvariable=self.result)
        ll.config(font=('arial', 30))
        ll.pack()

    def clickFunction(self, master):
        fileName = filedialog.askopenfilename(
            initialdir=os.getcwd(), title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if fileName:
            # self.filename.set(fileName)
            master.photo1 = ImageTk.PhotoImage(Image.open(fileName))
            self.vlabel.configure(image=master.photo1)
            result = self.hotdogChecker(fileName)
            self.result.set(result)

    def hotdogChecker(self, image):
        image = mx.image.imread(image)
        image = gcv.data.transforms.presets.imagenet.transform_eval(image)

        nw = gcv.model_zoo.resnet50_v1d(pretrained=True)
        prediction = nw(image)
        prediction = prediction[0]
        # print(prediction[950:])

        probability = mx.nd.softmax(prediction)

        rounded_prob = mx.nd.round(probability*100)/100
        # print(rounded_prob[950:])

        topProbObject = mx.nd.topk(probability, k=1)
        # print(topProbObject)

        # asscalar is used to convert an MXNet ND array with one element to a Python literal.
        class_i = topProbObject[0].astype('int').asscalar()
        class_label = nw.classes[class_i]
        class_prob = probability[class_i]
        print('#1 ', class_label, class_prob.asscalar()*100, class_i)
        if class_i == 934:
            # print('hotdog')
            return 'Hotdog'
        else:
            # print('not hotdog')
            return 'Not Hotdog'


root = Tk()
root.iconbitmap('shiba.ico')
gui = HotdogGUI(root)
root.mainloop()
