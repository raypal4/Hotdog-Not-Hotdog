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
                            command=lambda: self.getfilename_cb(master))
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

    def getfilename_cb(self, master):
        fname = filedialog.askopenfilename(
            initialdir=os.getcwd(), title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if fname:
            # self.filename.set(fname)
            master.photo1 = ImageTk.PhotoImage(Image.open(fname))
            self.vlabel.configure(image=master.photo1)
            result = self.hotdogChecker(fname)
            self.result.set(result)

    def hotdogChecker(self, image):
        image = mx.image.imread(image)
        image = gcv.data.transforms.presets.imagenet.transform_eval(image)

        network = gcv.model_zoo.resnet50_v1d(pretrained=True)
        prediction = network(image)
        prediction = prediction[0]
        # print(prediction[950:])

        probability = mx.nd.softmax(prediction)

        rounded_prob = mx.nd.round(probability*100)/100
        # print(rounded_prob[950:])

        k = 5
        topk_indicies = mx.nd.topk(probability, k=k)
        # print(topk_indicies)

        # asscalar is used to convert an MXNet ND array with one element to a Python literal.
        class_i = topk_indicies[0].astype('int').asscalar()
        class_label = network.classes[class_i]
        class_prob = probability[class_i]
        # print('#1 ', class_label, class_prob.asscalar()*100)
        if class_i == 934:
            # print('hotdog')
            return 'Hotdog'
        else:
            # print('not hotdog')
            return 'Not Hotdog'


root = Tk()
root.iconbitmap('shiba.ico')
my_gui = HotdogGUI(root)
root.mainloop()
