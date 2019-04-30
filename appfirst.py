from keras import models
from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import pandas
global graph
graph = tf.get_default_graph()
data= pandas.read_excel(r"D:\data case\data.xlsx")  
model=models.load_model("model.h5")

def getdata (EID):
    for i in range(0,len(data)):
        if EID in data["EID"][i]:
            c= data["Case"][i]
            ct= data["Case Type"][i]
            return c + " " +ct


labels = ["A1","A2", "A3" , "A4" , "A5"]

fd = cv2.CascadeClassifier(r"D:\kaizen_training\kaizen\haarcascade_frontalface_alt.xml")

app=Flask(__name__)

def get_face(img):
    corners = fd.detectMultiScale(img,1.3,4)
    if len(corners)==0:
        return None,None
    else:
        (x,y,w,h) = corners[0]
        img = img[y:y+w,x:x+h] #cropping the image
        img = cv2.resize(img,(100,100))
        return(x,y,w,h),img
    

@app.route('/')
def index():
    return render_template("index.html")


def gen():
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        corner,img2 = get_face(img)
        if corner != None:
            (x,y,w,h)= corner
            with graph.as_default():
                output = model.predict_classes(img2.reshape(1,100,100,3))
            EID = labels[output[0]]
            details= getdata(EID)
            cv2.putText(img,EID + " "+ details,(100,100),cv2.FONT_HERSHEY_COMPLEX, 1.0,(0,0,255),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
        ret,jpeg = cv2.imencode(".jpg",img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n')

       
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__=="__main__":
    app.run(debug=True,port=8000)
    