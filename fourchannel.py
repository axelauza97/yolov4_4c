import json
import sys
import cv2
import numpy as np

def search(busca):
    for data in jsonObject:
        name=data["filename"].replace("/home/luis/Escritorio/MI/dataSet2/noche/rgbca/test/images/","")
        if name == busca:
            return data
    return None

with open(sys.argv[1]) as f:
  jsonObject = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}


file = open(sys.argv[2],"r")
imgs=[]
for img in file:
      imgs.append(img.rstrip())

file.close()
imgs.sort(reverse=False)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outI = cv2.VideoWriter("videoThermal"+'.avi',fourcc, 20, (640+640,512))

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1


print("Writing")
for img in imgs:
    name = img.replace("/home/axelauza/Documentos/MI/RGBC/test/images/","")
    data=search(name)
    print(name)
    print(data)
    image=cv2.imread(img,cv2.IMREAD_UNCHANGED)
    (R,G,B,A)=cv2.split(image)
    rgb=cv2.merge([R,G,B])
    thermal = cv2.applyColorMap(A, cv2.COLORMAP_HOT)
    #blank_image = np.zeros((640+640,512,3), np.uint8)
    blank_image = np.zeros((512,640+640,3), np.uint8)
    blank_image[0:512,0:640]=rgb

    blank_image[0:512,640:640+640]=thermal

    objects=data["objects"]
    for object in objects:
        color=(0,255,0)
        classs="Person "
        if(object["class_id"]==1):
            color=(0,0,255)
            classs="People "
        if(object["class_id"]==2):
            color=(255,0,0)
            classs="Cyclist "
        coords=object["relative_coordinates"]
        cx=coords["center_x"]*640
        cy=coords["center_y"]*512
        w=coords["width"]*640//2
        h=coords["height"]*512//2
        cv2.rectangle(blank_image,( int(cx-w),int(cy+h)) , (int(cx+w),int(cy-h)) ,color)
        cv2.putText(blank_image,classs+ str( int(object["confidence"]*100)) +"%", 
         (int(cx-w),int(cy-h)), 
        font, 
        fontScale,
        color,
        lineType)
        cv2.rectangle(blank_image,( int(cx-w+640),int(cy+h)) , (int(cx+w+640),int(cy-h)) ,color)
        cv2.putText(blank_image,classs+ str( int(object["confidence"]*100)) +"%", 
         (int(cx-w+640),int(cy-h)), 
        font, 
        fontScale,
        (255,255,255),
        lineType)
    outI.write(blank_image)
outI.release()
