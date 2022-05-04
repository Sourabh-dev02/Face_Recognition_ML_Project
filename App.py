import cv2
import numpy as np
import face_recognition
import os
import streamlit as st

from datetime import datetime
path='images'
def detect(path):

    images=[]
    personName=[]
    myList=os.listdir(path)
    print(myList)
    for cu_img in myList:
       current_img= cv2.imread(f'{path}/{cu_img}')
       images.append(current_img)
       personName.append(os.path.splitext(cu_img)[0])
    print(personName)

    def faceEncodings(images):
        encodeList=[]
        for img in images:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode=face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    encodeListKnown=faceEncodings(images) #hog algo for encoding

    print("ALL ENCODINGS COMPLETED!!!!!!!!!!!")

    cap=cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()
        faces=cv2.resize(frame,(0,0),None,0.25,0.25)
        faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)

        facesCurrFrame=face_recognition.face_locations(faces)
        encodesCurrFrame=face_recognition.face_encodings(faces,facesCurrFrame)
        for encodeFace, faceloc in zip(encodesCurrFrame,facesCurrFrame):
            matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)

            matchIndex=np.argmin(faceDis)

            if matches[matchIndex]:
                name=personName[matchIndex].upper()
                # print(name)
                y1,x2,y2,x1=faceloc
                y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(frame,name,(x1+6,y2+6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow("Camera",frame)
        if cv2.waitKey(10)==13:
            break
    cap.release()
    cv2.destroyAllWindows()
path='images'
detect(path)


# def main():
#     st.title("Face Regcognition App:smile:")
#     st.write("by Sourabh Mahindrakar")
#     st.write("------------------------------")
#     st.write("**Using the Haar Cascade Classifier**")
#     activities=["Home","contact"]
#     choice=st.sidebar.selectbox("Menu",activities)
#     if choice=="Home":
#         st.write("Amigo! please upload a face photos....")
#         st.write("note:photo uploaded should be in a format name.jpg")






