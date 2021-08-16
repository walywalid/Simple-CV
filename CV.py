from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
import cv2
import random
import numpy as np

def load_image():
    global orig_img
    path_image = path_directory.get()
    if path_image == '':
        messagebox.showerror("Error", "Enter a valid path of image with extension")
        return
    try:
        orig_img = cv2.cvtColor(cv2.imread(path_image),cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img,(400,400))
        img1 = Image.fromarray(orig_img)
        global img_box1
        img_box1 = ImageTk.PhotoImage(img1)
        global label1
        label1 = Label(frame_img1,image = img_box1, bg = 'blue')
        label1.grid(row = 0, column = 0)
        path.configure(state = 'disable')
    except:
        messagebox.showerror("Error", "Entered path of image or extension is incorrect")

def load_video():
    path_video = path_directory.get()
    if path_video == '':
        messagebox.showerror("Error", "Enter a valid path of video with extension")
        return
    try:
        vid = cv2.VideoCapture(path_video)
        ret, frame = vid.read()
        k = 0
        while k != 27:
            ret, frame = vid.read()
            cv2.imshow('Video Playing',frame)
            k = cv2.waitKey(40)
    except:
        messagebox.showerror("Error", "Entered path of video or extension is incorrect")


def blur_img():
    blur = cv2.GaussianBlur(orig_img,(5,5),2)
    img2 = Image.fromarray(blur)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def erode():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(orig_img,kernel,iterations = 2)
    img2 = Image.fromarray(erosion)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def dilate():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilate = cv2.dilate(orig_img,kernel,iterations = 2)
    img2 = Image.fromarray(dilate)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def morph_opening():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opening = cv2.morphologyEx(orig_img,cv2.MORPH_OPEN,kernel,iterations = 2)
    img2 = Image.fromarray(opening)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def morph_closing():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    closing = cv2.morphologyEx(orig_img,cv2.MORPH_CLOSE,kernel,iterations = 2)
    img2 = Image.fromarray(closing)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def add_noise():
    img = orig_img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    row,col = img.shape

    number_of_pixels = random.randint(300, 1000)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 255

    number_of_pixels = random.randint(300 , 1000)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 0

    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def edge_detection():
    img = orig_img.copy()
    img = cv2.Canny(img,100,150)
    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def median_filter():
    img = orig_img.copy()
    img = cv2.medianBlur(img,5)
    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def lap_filter():
    img = orig_img.copy()
    img = cv2.Laplacian(img,cv2.CV_8UC3)
    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def hist_equalize():
    img = orig_img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def corner_extract():
    img = orig_img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 255, 0]

    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def fast_features():
    img = orig_img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(False)
    kp = fast.detect(gray_img, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))

    img2 = Image.fromarray(kp_img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def sift_features():
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray_img, None)
        kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2 = Image.fromarray(kp_img)
        global img_box2
        img_box2 = ImageTk.PhotoImage(img2)
        global label2
        label2 = Label(frame_img2,image = img_box2, bg = 'green')
        label2.grid(row = 0, column = 1)

    except:
        messagebox.showerror("Error", "SIFT is not available in this version of OpenCV, Install 3.4.2")

def orb_features():
    img = orig_img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(gray_img, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    img2 = Image.fromarray(kp_img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def lines_detection():
    img = orig_img.copy()
    dst = cv2.Canny(img, 50, 200, None, 3)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)

    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def circles_detection():
    img = orig_img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5),1)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 80,param2 = 60, minRadius = 20, maxRadius = 150)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)

    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def up_sampling():
    img = orig_img.copy()
    img = cv2.resize(img,(450,450))

    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def down_sample():
    img = orig_img.copy()
    img = cv2.resize(img,(300,300))

    img2 = Image.fromarray(img)
    global img_box2
    img_box2 = ImageTk.PhotoImage(img2)
    global label2
    label2 = Label(frame_img2,image = img_box2, bg = 'green')
    label2.grid(row = 0, column = 1)

def reset():
    label2.grid_forget()
    label1.grid_forget()
    global path
    path.configure(state = 'normal')



root = Tk()
root.title('Image Processing')
root.geometry('810x610')
#root.resizable(0,0)
root.configure(bg = 'white')

global frame_img1, frame_img2, frame_btns, frame_features, img_box1, img_box2


frame_img1 = Frame(root,width = 400,height = 400, bd = 2)
frame_img2 = Frame(root,width = 400, height = 400, bd = 2)
frame_btns = Frame(root,width = 400,height = 200, bd = 2)
frame_features = Frame(root,width = 400,height = 200, bd = 2)

btn_reset = Button(frame_features,command = reset, text = 'Reset')
btn_imageload = Button(frame_btns,command = load_image, text = 'Load Image')
btn_videoload = Button(frame_btns,command = load_video, text = 'Load Video')
btn_noise = Button(frame_btns, command = add_noise, text = 'Add Noise')
btn_erode = Button(frame_btns, command = erode, text = 'Erosion')
btn_dilate = Button(frame_btns,command = dilate, text = 'Dilation')
btn_opening = Button(frame_btns,command = morph_opening, text = 'Opening')
btn_closing = Button(frame_btns,command = morph_closing, text = 'Closing')
btn_median = Button(frame_btns, command = median_filter, text = 'Median Filter')
btn_blur = Button(frame_btns,command = blur_img, text = 'Blur')
btn_lap = Button(frame_btns, command = lap_filter ,text = 'Laplacian')
btn_edges = Button(frame_btns,command = edge_detection, text = 'Edges')
btn_hist = Button(frame_btns,command = hist_equalize, text = 'Histogram Eq')
btn_corners = Button(frame_btns,command = corner_extract, text = 'Corners')
btn_fast = Button(frame_btns,command = fast_features, text = 'FAST')
btn_sift = Button(frame_btns,command = sift_features, text = 'SIFT')
btn_orb = Button(frame_btns,command = orb_features, text = 'ORB')
btn_lines = Button(frame_btns,command = lines_detection, text = 'Hough Lines')
btn_circles = Button(frame_btns,command = circles_detection, text = 'Hough Circles')
btn_upsample = Button(frame_btns,command = up_sampling, text = 'Up Sample')
btn_downsample = Button(frame_btns,command = down_sample, text = 'Down Sample')


#Labels Creation
pathlabel = Label(frame_features,text = 'Path')


# Field for entering values
path_directory = StringVar()
global path
path = Entry(frame_features,width = 20,textvariable = path_directory)
path.focus()


# Frames creations
frame_img1.grid(row=0,column=0)
frame_img2.grid(row=0,column=1)
frame_btns.grid(row=1,column=0)
frame_features.grid(row=1,column=1)

#Button Setting
btn_imageload.grid(row=0,column=0)
btn_videoload.grid(row=0,column=1)
btn_noise.grid(row=0,column=2)

btn_upsample.grid(row = 1, column = 0)
btn_downsample.grid(row = 1, column = 1)
btn_erode.grid(row=1,column=2)

btn_opening.grid(row=2,column=0)
btn_closing.grid(row=2,column=1)
btn_median.grid(row=2,column=2)

btn_lap.grid(row=3,column=0)
btn_hist.grid(row = 3, column = 1)
btn_corners.grid(row = 3, column = 2)

btn_lines.grid(row = 4, column = 0)
btn_circles.grid(row = 4, column = 1)
btn_sift.grid(row = 4, column = 2)

btn_edges.grid(row = 5, column = 0)
btn_dilate.grid(row=5,column=1)
btn_blur.grid(row=5,column=2)
btn_fast.grid(row = 6, column = 0)
btn_orb.grid(row = 6, column = 1)


btn_reset.grid(row=5,column=1)


#labels setting
pathlabel.grid(row = 0, column = 0)

# Fields
path.grid(row = 0, column = 1)
#vidpath.grid(row = 0, column = 1)


root.mainloop()
