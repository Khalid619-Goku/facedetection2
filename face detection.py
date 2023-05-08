#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries(MTCNN=multi_task cascaded convutional neural network)
import facenet_pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os


# In[2]:


# initializing MTCNN and InceptionResnetV1 

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval() 


# In[3]:


# read data from folder
dataset =datasets.ImageFolder(r"D:\lol\pics")
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = []
name_list = []  #list of names correspondence to cropped photos
embedding_list = []    #list of embedded matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob>0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])
        
# save data
data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file


# In[25]:


#using webcam to detect face
load_data=torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]
names = []


cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture('vtest.avi')
# cam = cv2.imread("geeksforgeeks.png", cv2.IMREAD_COLOR)
nameDetected = None

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame, try again")
        break
        
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob = True)
    
    if img_cropped_list is not None:
        boxes, _=mtcnn.detect(img)
        
        
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()
                
                dist_list = [] # list of amtched distances minimum distances is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)
                    
                min_dist = min(dist_list)#get min dist value
                min_dist_idx = dist_list.index(min_dist) # get min dist index
                name = name_list[min_dist_idx] # get name corresponding to minimum dist
                nameDetected = name
               # print("show",nameDetected)
                names.append(nameDetected)
                
                box = boxes[i]
                original_frame = frame.copy() #storing copy of frames before drawing on it
                pt1 = (int(box[0]),int(box[1]))
                pt2 = (int(box[2]),int(box[3]))
#                 pt1 = tuple([int(j) for j in box[0]],[int(j) for j in box[1]] )
#                 pt2 = tuple([int(j) for j in box[2]],[int(j) for j in box[3]])
#                 pt1 = tuple([int(j) for j in boxes[i][0]])
#                 pt2 = tuple([int(j) for j in boxes[i][1]])
                
                if min_dist<0.90:
                    frame = cv2.putText(frame, name, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                
                
#                 if min_dist<0.90:
#                     frame=cv2.putText(frame, name+' '+str(min_dist), (box[0],box[1])< cv2.FRONT_HERSHEY_SIMPLEX, 1)
#                 print(box)
                frame = cv2.rectangle(frame, pt1, pt2,(255,0,0),2)
#                 frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
#                 print(frame)
        cv2.imshow("IMG",frame)
#         cv2.imshow("IMG")
        
        k = cv2.waitKey(1)
#         esc
        if k%256==27: 
            print('closing...')
        
            break
            
#             space
        elif k%256==32:
            print("enter your name: ")
            name = input()
            
            #create directory if not exists 
            if not os.path.exists('D:\lol\pics/'+name):
                os.mkdir('D:\lol\pics/'+name)
            
#             img
                
            img_name = "D:\lol\pics/{}/{}.jpg".format(name, int(time.time()))
            cv2.imwrite(img_name, original_frame)
            print(" saved: {}".format(img_name))
            
cam.release()
cv2.destroyAllWindows()
#print(names)


# In[26]:


del names[:10]
print(set(names))


# In[27]:


name_list = names

recognized_faces = []
for  i in  (name_list):
    recognized_faces.append(names)

new_list =[]
for sublist in recognized_faces:
    for item in sublist:
        new_list.append(item)

# Print the list of recognized face names
from collections import OrderedDict
print(list(set(new_list)))
# print(new_list)
list(set(new_list))


# In[28]:


import pandas as pd


# In[29]:


data = [["Khalid", "male", "a"], ["Shahid", "male", "a"],["Tanya","Female","a"],["Jintu","male","a"]]


# In[30]:


df = pd.DataFrame(data, columns = ["Name", "Gender", "Attendance"])
print(df)


# In[31]:


df.to_csv("data.csv", index = False)


# In[32]:


records = pd.read_csv("data.csv").values.tolist()
print(records)


# In[ ]:


new_list1 = []
for record in records:
    for items in range (len(new_list)):
        if record[0].lower() == item:
            print("Here")
            record[2] = "p"
            print(record)
            new_list.append(record)
            print(record[0])


# In[ ]:


print(records)


# In[ ]:


records = pd.DataFrame(records, columns = ["Name", "Gender", "Attendance"])
print(records)


# In[21]:


records.to_csv("data.csv", index = False)

