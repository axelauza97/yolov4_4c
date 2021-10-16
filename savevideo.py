import cv2
import sys
import threading


#usage savevideo.py <path a csv>
def saveVideoRGB(files,name,box):	
    #cnt=0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outI = cv2.VideoWriter(name+'.avi',fourcc, 20, (640,512))
    print("Writing")
    for file in files:
        img=cv2.imread(file)
        if(box):
            boxes=file.replace(".jpg",".txt")
            boxes=open(boxes,"r")
            for box in boxes:
                parts=box.split(" ")
                #print(parts)
                color=(0,255,0)
                if(int(parts[0])==1):
                    color=(0,0,255)
                if(int(parts[0])==2):
                    color=(255,0,0)
                cx=float(parts[1].rstrip())*640
                cy=float(parts[2].rstrip())*512
                w=float(parts[3].rstrip())*640//2
                h=float(parts[4].rstrip())*512//2
                
                cv2.rectangle(img,( int(cx-w),int(cy+h)) , (int(cx+w),int(cy-h)) ,color)
            
        print(file)
        
        outI.write(img)
        #cnt=cnt+1
        #if(cnt==100):
        #    break
    
    outI.release()
    print("Finish")


def saveVideoFOUR(files,name,box):	
    cnt=0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outI = cv2.VideoWriter(name+'.mov',fourcc, 30, (640,512))
    print("Writing")
    for file in files:
        img=cv2.imread(file,cv2.IMREAD_UNCHANGED)
        if(box):
            boxes=file.replace(".png",".txt")
            boxes=open(boxes,"r")
            for box in boxes:
                parts=box.split(" ")
                #print(parts)
                color=(0,255,0,0)
                if(int(parts[0])==1):
                    color=(0,0,255,0)
                if(int(parts[0])==2):
                    color=(255,0,0,0)
                cx=float(parts[1].rstrip())*640
                cy=float(parts[2].rstrip())*512
                w=float(parts[3].rstrip())*640
                h=float(parts[4].rstrip())*512
                
                cv2.rectangle(img,( int(cx-w),int(cy+h)) , (int(cx+w),int(cy-h)) ,color)
            
        print(file)
        
        outI.write(img)
        cnt=cnt+1
        if(cnt==100):
            break
    
    outI.release()
    print("Finish")


file = open(sys.argv[1],"r")
file2 = open("new.csv","w")
imgs=[]
#verify data augmentation
data_aug=["r.jpg","cr.jpg","sh.jpg","fh.jpg","sc.jpg","co.jpg"]
for img in file:
  if not ((data_aug[0] in img )
  or (data_aug[1] in img )
  or (data_aug[2] in img )
  or (data_aug[3] in img )
  or (data_aug[4] in img )
  or (data_aug[5] in img )):
      imgs.append(img.rstrip())
      file2.write(img)
file.close()
file2.close()
imgs.sort(reverse=False)
print(imgs)

x1 = threading.Thread(target=saveVideoRGB, args=(imgs,'videobox',True))
x1.start()
x2 = threading.Thread(target=saveVideoRGB, args=(imgs,'video',False))
x2.start()

x1.join()
x2.join()

#saveVideoRGB(imgs,'videobox',True)
#saveVideoRGB(imgs,'video',False)
#saveVideoFOUR(imgs,'videoPNG',True)
#saveVideoFOUR(imgs,'videoPNGbox',False)

