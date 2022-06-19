import cv2
import tkinter as tk
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image


window = tk.Tk()
window.geometry('850x350+370+170')
window.iconbitmap('logo/icon.ico')

#stop resize
window.resizable(width=False, height=False)
window.title("AI PROJECT - EMOTION RECOGNITION - TR.KIEN")

label_1 = tk.Label(window, text="PROJECT CUỐI KÌ MÔN TRÍ TUỆ NHÂN TẠO")
label_1.configure(font=("Times New Roman", 20, "bold")) 
label_1.pack()

label_2 = tk.Label(window, text="SVTH: TRẦN TRUNG KIÊN")
label_2.configure(font=("Times New Roman", 20, "bold"))
label_2.pack()

label_3 = tk.Label(window, text="MSSV: 19146346")
label_3.configure(font=("Times New Roman", 20, "bold"))
label_3.pack()

#SPKT logo
img = Image.open("logo/logo.png")
logo_rz=img.resize((105,120))
logo = ImageTk.PhotoImage(logo_rz)
label_4 = tk.Label(image=logo) 
label_4.place(relx=0.01, rely=0.02, anchor= NW)

#CKM logo
img1 = Image.open("logo/fme.png")
logo_rz1=img1.resize((116,116))
logo1 = ImageTk.PhotoImage(logo_rz1)
label_5 = tk.Label(image=logo1) 
label_5.place(relx=0.99, rely=0.02, anchor= NE)

#Load models
model = model_from_json(open("model/model_v2/model_arch_v2.json", "r").read())
model.load_weights('model/model_v2/my_model_v2.h5')
face_haar_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

#Test camera
def cam():
    cap =cv2.VideoCapture(0)
    while True:
      ret,frame=cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('AI PROJECT - EMOTION DETECTION - TR.KIEN - WEBCAM', frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
    cap.release()
    cv2.destroyAllWindows()

#Open camera & detect
def detect():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        res, frame = cap.read()
        height, width, channel = frame.shape
        sub_img = frame[0:int(height / 6), 0:int(width)]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 1, black_rect, 0.23, 0) #0.77
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        FONT_THICKNESS = 1
        lable_color = (0, 255, 0)  # parameter
        lable = "..."
        lable_dimension = cv2.getTextSize(lable, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        textX = int((res.shape[1] - lable_dimension[0]) / 2)
        textY = int((res.shape[0] + lable_dimension[1]) / 2)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try:
            for (x, y, w, h) in faces:
                # property detect
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y - 5:y + h + 5, x - 5:x + w + 5]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe') 
                emotion_prediction = emotion_detection[max_index]
                #text detect
                cv2.putText(res, "TrangThai: {}".format(emotion_prediction), (3, textY), FONT, 0.7,
                            lable_color, 2)
        except:
            pass
        frame[0:int(height / 6), 0:int(width)] = res
        cv2.imshow('AI PROJECT - EMOTION DETECTION - TR.KIEN - DETECT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#Detect & record
def detectRec():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    op = cv2.VideoWriter('videorec.mp4', fourcc, 9.0, (640, 480))
    while cap.isOpened():
        res, frame = cap.read()
        height, width, channel = frame.shape
        sub_img = frame[0:int(height / 6), 0:int(width)]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 1, black_rect, 0.23, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        FONT_THICKNESS = 1
        lable_color = (0, 255, 0)  # parameter
        lable = "..."
        lable_dimension = cv2.getTextSize(lable, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        textX = int((res.shape[1] - lable_dimension[0]) / 2)
        textY = int((res.shape[0] + lable_dimension[1]) / 2)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try:
            for (x, y, w, h) in faces:
                # property detect
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y - 5:y + h + 5, x - 5:x + w + 5]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(res, "TrangThai: {}".format(emotion_prediction), (3, textY), FONT, 0.7,
                            lable_color, 2)
        except:
            pass
        frame[0:int(height / 6), 0:int(width)] = res
        op.write(frame)
        cv2.imshow('AI PROJECT - EMOTION DETECTION - TR.KIEN- DETECT & RECORD', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    op.release()
    cap.release()
    cv2.destroyAllWindows()

#exit program
def exitt():
   exit()

#Button Test camera
but1=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=cam,text='Test Camera',font=('helvetica 15 bold'))
but1.place(relx=0.5,rely=0.4, anchor= CENTER)

#Button only detect
but2=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=detect,text='Open Camera & Detect',font=('helvetica 15 bold'))
but2.place(relx=0.5,rely=0.55, anchor= CENTER)

#Button detect & record
but3=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=detectRec,text='Detect & Record',font=('helvetica 15 bold'))
but3.place(relx=0.5,rely=0.7, anchor= CENTER)

#Button exit
but4=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but4.place(relx=0.5,rely=0.85, anchor= CENTER)

window.mainloop()

