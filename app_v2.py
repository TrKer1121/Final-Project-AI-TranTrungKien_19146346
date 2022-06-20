from email import contentmanager
import cv2
import tkinter as tk
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from sklearn.tree import export_text

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tkinter import filedialog


window = tk.Tk()
window.geometry('850x430+370+170')
window.iconbitmap('logo/icon.ico')

#stop resize
window.resizable(width=False, height=False)
window.title("AI PROJECT - EMOTION RECOGNITION - TR.KIEN")

label_1 = tk.Label(window, text="PROJECT CUỐI KÌ MÔN TRÍ TUỆ NHÂN TẠO")
label_1.configure(font=("Times New Roman", 20, "bold")) 
label_1.pack()

label_4 = tk.Label(window, text="ĐỀ TÀI: NHẬN DIỆN CẢM XÚC BẢN THÂN")
label_4.configure(font=("Times New Roman", 20, "bold"))
label_4.pack()

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
model = model_from_json(open("model/model_v3/model_arch_v3.json", "r").read())
model.load_weights('model/model_v3/my_model_v3.h5')
face_haar_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')


def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        res, frame = cap.read()
        #frame = cv2.resize(frame,(480,480))
        height, width, channel = frame.shape
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try: 
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(frame, emotion_prediction, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0, 255, 0), 1, cv2.LINE_AA)
        except:
            pass
        frame[0:int(height/1000), 0:int(width)] = res
        cv2.imshow('AI PROJECT - EMOTION DETECTION - TR.KIEN - DETECT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 

    
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
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try: 
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(frame, emotion_prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            pass
        frame[0:int(height/20), 0:int(width)] = res
        cv2.imshow('AI PROJECT - EMOTION DETECTION - TR.KIEN - DETECT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 


def detect():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    op = cv2.VideoWriter('videorec.mp4', fourcc, 9.0, (640, 480))
    while cap.isOpened():
        res, frame = cap.read()
        height, width, channel = frame.shape
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try: 
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(frame, emotion_prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            pass
        frame[0:int(height/20), 0:int(width)] = res
        op.write(frame)
        cv2.imshow('AI PROJECT - EMOTION DETECTION - TR.KIEN - DETECT & RECORD', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    op.release()
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
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try: 
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(frame, emotion_prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            pass
        frame[0:int(height/20), 0:int(width)] = res
        op.write(frame)
        cv2.imshow('AI PROJECT - EMOTION DETECTION - TR.KIEN - DETECT & RECORD', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    op.release()
    cap.release()
    cv2.destroyAllWindows() 

#exit program
def exitt():
   exit()

#Button import file and recog
but1=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,text='Import File & Recognition',command=UploadAction,font=('helvetica 15 bold'))
but1.place(relx=0.5,rely=0.44, anchor= CENTER)

#Button Test camera
but2=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=cam,text='Test Camera',font=('helvetica 15 bold'))
but2.place(relx=0.5,rely=0.56, anchor= CENTER)

#Button only detect
but3=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=detect,text='Open Camera & Recognition',font=('helvetica 15 bold'))
but3.place(relx=0.5,rely=0.68, anchor= CENTER)

#Button detect & record
but4=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=detectRec,text='Recognition & Record',font=('helvetica 15 bold'))
but4.place(relx=0.5,rely=0.8, anchor= CENTER)

#Button exit
but5=Button(window,padx=5,pady=5,width=30,bg='white',fg='red',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but5.place(relx=0.5,rely=0.92, anchor= CENTER)


window.mainloop()

