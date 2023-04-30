from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

window = Tk()
window.geometry("600x600")
window.title("Image Classification")
window.config(background = "antiquewhite")
refresh_img = PhotoImage(file = "./DL_Projects/Cifar10 Classification/Images/img_refresh.png")
exit_img = PhotoImage(file = "./DL_Projects/Cifar10 Classification/Images/img_exit.png")
select_img = PhotoImage(file = "./DL_Projects/Cifar10 Classification/Images/img_select.png")
classify_img = PhotoImage(file = "./DL_Projects/Cifar10 Classification/Images/img_classify.png")


model = models.load_model("./DL_Projects/Cifar10 Classification/cifar_image_classification.model")
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]




def Close():
    window.destroy()

def DrawImage(img):
    global canvas, image_container
    canvas = Canvas(window, width = 400, height = 350, borderwidth = 0) 
    canvas.pack()
    image_container = canvas.create_image(0,0,anchor=NW,image=img)

def Update():
    global my_image
    filename = filedialog.askopenfilename(initialdir = "C:/Users/kwarg/Downloads",
                                          title      = "Select a image", 
                                          filetypes  = (("jpg files", "*.jpg"),("png files", "*.png")))
    img = (Image.open(filename))
    resized_img= img.resize((400,350), Image.ANTIALIAS)
    my_image = ImageTk.PhotoImage(resized_img)
    Prediction_label.destroy()
    canvas.itemconfig(image_container, image = my_image)
    Classify(filename)
    
def Predict(path):
    global Prediction_label
    pic = cv2.imread(path)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic, (32,32))
    pic = tf.keras.utils.normalize(pic, axis=1)
    prediction = model.predict(np.array([pic]), verbose = 0)
    index = np.argmax(prediction)
    Prediction_label = Label(window, text = f'Prediction is {class_names[index]}', font = ('Arial',15), bg = "antiquewhite")
    Prediction_label.pack(padx = 50)
    Prediction_label.place(x = 215, y = 355)
    restart_button = Button(window, image = refresh_img, command = Update, borderwidth = 0)
    restart_button.pack(padx = 50)
    restart_button.place(x = 370, y = 410)
 
def Classify(path):
    classification_button = Button(window, image = classify_img , command=lambda: Predict(path), borderwidth = 0)
    classification_button.pack(padx = 50)
    classification_button.place(x = 240, y = 410)

def BrowseFiles():
    global my_image
    filename = filedialog.askopenfilename(initialdir = "C:/Users/kwarg/Downloads",
                                          title      = "Select a image", 
                                          filetypes  = (("png files", "*.png"),("jpg files", "*.jpg")))
    img = (Image.open(filename))
    resized_img= img.resize((400,350), Image.ANTIALIAS)
    my_image = ImageTk.PhotoImage(resized_img)
    DrawImage(my_image)
    Classify(filename)



filename_button = Button(window, image = select_img, command = BrowseFiles, borderwidth = 0)
filename_button.pack(padx = 50)
filename_button.place(x = 220, y = 490)


exit_button = Button(window, image = exit_img, command = Close, borderwidth = 0)
exit_button.pack(padx = 50)
exit_button.place(x = 270, y = 550)


window.mainloop()