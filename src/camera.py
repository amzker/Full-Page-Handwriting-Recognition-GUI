from autocorrect import Speller
spell = Speller()
from image_slicer import slice
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog
import time
import threading
import json
from typing import Tuple, List
import os 
import editdistance
from pathlib import Path
import numpy as np
from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

class FilePaths:
        """Filenames and paths to data."""
        fn_char_list = '../model/charList.txt'
        fn_summary = '../model/summary.json'
        fn_corpus = '../data/corpus.txt'
        

def get_img_height() -> int:
    return 32

def get_img_size(line_mode: bool = True) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 256, get_img_height()
def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)
        
def infer(model: Model, fn_img: Path) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)
    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    #os.system("echo "+recognized+probability+">>logs.txt")
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    return recognized[0]

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())
    
model = Model(char_list_from_file(),2, must_restore=True, dump=False)

def createwidgets():
    root.feedlabel = Label(root, bg="steelblue", fg="white", text="WEBCAM", font=('Comic Sans MS',20))
    root.feedlabel.grid(row=1, column=1, padx=10, pady=10, columnspan=2)
    root.cameraLabel = Label(root, bg="steelblue", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan=2)
    root.textLabel = Entry(root, borderwidth=2,width=100 ,relief="groove", textvariable=translated)
    root.textLabel.grid(row=6, column=1, padx=10, pady=10, columnspan=1)
    root.uptextLabel = Entry(root, borderwidth=2,width=100 ,relief="groove", textvariable=uptranslated)
    root.uptextLabel.grid(row=7, column=1, padx=10, pady=10, columnspan=1)
    root.trbtn = Button(root, text="Get Translation", command=translate, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=13)
    root.trbtn.grid(row=5, column=4)

    root.saveLocationEntry = Entry(root, width=55, textvariable=destPath)
    root.saveLocationEntry.grid(row=3, column=1, padx=10, pady=10)

    root.browseButton = Button(root, width=10, text="BROWSE", command=destBrowse)
    root.browseButton.grid(row=3, column=2, padx=10, pady=10)

    root.captureBTN = Button(root, text="CAPTURE", command=Capture, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=20)
    root.captureBTN.grid(row=4, column=1, padx=10, pady=10)
    
    root.CamNum = Entry(root,textvariable=Camvar)
    root.CamNum.grid(row=5,column=1,padx=10,pady=10)
    
    root.CAMBTN = Button(root, text="STOP CAMERA", command=StopCAM, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=13)
    root.CAMBTN.grid(row=4, column=2)
    

    root.previewlabel = Label(root, bg="steelblue", fg="white", text="IMAGE PREVIEW", font=('Comic Sans MS',20))
    root.previewlabel.grid(row=1, column=4, padx=10, pady=10, columnspan=2)

    root.imageLabel = Label(root, bg="steelblue", borderwidth=3, relief="groove")
    root.imageLabel.grid(row=2, column=4, padx=10, pady=10, columnspan=2)

    root.openImageEntry = Entry(root, width=55, textvariable=imagePath)
    root.openImageEntry.grid(row=3, column=4, padx=10, pady=10)
    
    root.RawNumEntry = Entry(root, width=10, textvariable=rawno)
    root.RawNumEntry.grid(row=4, column=4, padx=10, pady=10)
    
    root.RowSowBTN = Button(root, text="Show Rows", command=ShowRow)
    root.RowSowBTN.grid(row=4, column=5)

    root.openImageButton = Button(root, width=10, text="BROWSE", command=imageBrowse)
    root.openImageButton.grid(row=3, column=5, padx=10, pady=10)
    ShowFeed()

# Defining ShowFeed() function to display webcam feed in the cameraLabel;
def ShowFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()
    
    if ret:
        # Flipping the frame vertically
        #frame = cv2.flip(frame, 1)

        # Displaying date and time on the feed
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))

        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(cv2image)

        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image = videoImg)

        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)
        #root.text.insert(tk.END, "Just a text Widget\nin two lines\n")

        # Keeping a reference
        root.cameraLabel.imgtk = imgtk

        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image='')

def destBrowse():
    # Presenting user with a pop-up for directory selection. initialdir argument is optional
    # Retrieving the user-input destination directory and storing it in destinationDirectory
    # Setting the initialdir argument is optional. SET IT TO YOUR DIRECTORY PATH
    destDirectory = filedialog.askdirectory(initialdir="YOUR DIRECTORY PATH")

    # Displaying the directory in the directory textbox
    destPath.set(destDirectory)

def imageBrowse():
    # Presenting user with a pop-up for directory selection. initialdir argument is optional
    # Retrieving the user-input destination directory and storing it in destinationDirectory
    # Setting the initialdir argument is optional. SET IT TO YOUR DIRECTORY PATH
    openDirectory = filedialog.askopenfilename(initialdir="YOUR DIRECTORY PATH")

    # Displaying the directory in the directory textbox
    imagePath.set(openDirectory)

    # Opening the saved image using the open() of Image class which takes the saved image as the argument
    imageView = Image.open(openDirectory)

    # Resizing the image using Image.resize()
    imageResize = imageView.resize((640, 480), Image.ANTIALIAS)

    # Creating object of PhotoImage() class to display the frame
    imageDisplay = ImageTk.PhotoImage(imageResize)

    # Configuring the label to display the frame
    root.imageLabel.config(image=imageDisplay)

    # Keeping a reference
    root.imageLabel.photo = imageDisplay

# Defining Capture() to capture and save the image and display the image in the imageLabel
def Capture():
    # Storing the date in the mentioned format in the image_name variable
    image_name = datetime.now().strftime('%d-%m-%Y %H-%M-%S')

    # If the user has selected the destination directory, then get the directory and save it in image_path
    if destPath.get() != '':
        image_path = destPath.get()
    # If the user has not selected any destination directory, then set the image_path to default directory
    else:
        messagebox.showerror("ERROR", "NO DIRECTORY SELECTED TO STORE IMAGE!!")

    # Concatenating the image_path with image_name and with .jpg extension and saving it in imgName variable
    imgName = image_path + '/' + image_name + ".jpg"

    # Capturing the frame
    ret, frame = root.cap.read()

    # Displaying date and time on the frame
    #cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (430,460), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))

    # Writing the image with the captured frame. Function returns a Boolean Value which is stored in success variable
    success = cv2.imwrite(imgName, frame)

    # Opening the saved image using the open() of Image class which takes the saved image as the argument
    saved_image = Image.open(imgName)

    # Creating object of PhotoImage() class to display the frame
    saved_image = ImageTk.PhotoImage(saved_image)

    # Configuring the label to display the frame
    root.imageLabel.config(image=saved_image)

    # Keeping a reference
    root.imageLabel.photo = saved_image
    imagePath.set(imgName)

    # Displaying messagebox
    if success :
        messagebox.showinfo("SUCCESS", "IMAGE CAPTURED AND SAVED IN " + imgName)
    


# Defining StopCAM() to stop WEBCAM Preview
def StopCAM():
    # Stopping the camera using release() method of cv2.VideoCapture()
    root.cap.release()

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="START CAMERA", command=StartCAM)

    # Displaying text message in the camera label
    root.cameraLabel.config(text="OFF CAM", font=('Comic Sans MS',70))

def StartCAM():
    # Creating object of class VideoCapture with webcam index
    if Camvar.get().isnumeric():
        root.cap = cv2.VideoCapture(int(Camvar.get()))
    else:
        root.cap = cv2.VideoCapture(Camvar.get())
                                    

    # Setting width and height
    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="STOP CAMERA", command=StopCAM)

    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowFeed()
    
def ShowRow():
    rows = int(rawno.get())
    image = cv2.imread(imagePath.get())
    height, width = image.shape[1], image.shape[0]
    PerRow = int(height/rows)
    for i in range(0,height,PerRow):
        image = cv2.line(image, (0,i), (width,i), (0, 255, 0),1)
    cv2.imwrite("rowed.jpg", image)
    image = Image.open("rowed.jpg")
    image = image.resize((640, 480), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    root.imageLabel.config(image=image)
    root.imageLabel.photo = image


def translate():
    if imagePath.get() != '':
        if rawno.get() != '':
            imPath = imagePath.get()
            raws = int(rawno.get())
            filename = imPath.replace(".jpg","").replace(".png","").replace(".jpeg","")
            listPath = filename+"_"
            if raws != 1:
                slice(imPath,number_tiles=None,col=1,row=raws) 
                output = []
                beautyfied = []
                for i in range(1,raws+1):
                    fi = listPath+str(i).zfill(2)+"_01.png"
                    data = infer(model, fi)
                    bbt = spell(data) 
                    output.append(data)
                    beautyfied.append(bbt)
                    os.remove(fi)
                print(output)
                translated.set(output)
                uptranslated.set(beautyfied)
            else:
                tr = infer(model, imPath)
                translated.set(tr)
                text = spell(tr).replace("{","\n").replace("}","")
                uptranslated.set(text)
        else:
            messagebox.showerror("ERROR", "NO ROW NO DEFINED!!")
    else:
        messagebox.showerror("ERROR", "NO DIRECTORY SELECTED TO STORE IMAGE!!")

    
    
# Creating object of tk class
root = tk.Tk()

# Creating object of class VideoCapture with webcam index
root.cap = cv2.VideoCapture(0)

# Setting width and height
width, height = 640, 480
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Setting the title, window size, background color and disabling the resizing property
root.title("Hnadwriting CamApp")
root.geometry("1200x700")
root.resizable(True, True)

# Creating tkinter variables
destPath = StringVar()
imagePath = StringVar()
translated = StringVar()
uptranslated = StringVar()
rawno = StringVar()
Camvar = StringVar()
Camvar.set(0)

rawno.set("10")
createwidgets()
root.mainloop()






