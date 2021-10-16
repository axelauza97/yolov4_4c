#!/usr/bin/python
import sys,getopt,cv2,os,shutil,math,random,traceback
import numpy as np
import xml.etree.ElementTree as ET
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmenters.arithmetic import cutout
from skimage.transform import resize
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import threading



def dataset(dir_dataset,ndataset,dir_in,dir_out,augment,typee,fourch,dd):
	dclass = {'person':'0','people':'1','cyclist':'2'}
	print(dir_dataset,ndataset,dir_in,dir_out,augment,typee,fourch,dd)
	directory_dataset = os.listdir(dir_dataset)
	ddataset = dd
	for i in directory_dataset:
		path = os.listdir(dir_dataset+'/'+i)
		for j in path:
			tree = ET.parse(dir_dataset+'/'+i+'/'+j)
			root = tree.getroot()
			for obj in root.findall('object'):
				class_name = obj.find('name').text.replace('?','')
				#if not class_name in dclass:
				#	continue
				xmin = float(obj.find('bndbox').find('xmin').text)
				ymin = float(obj.find('bndbox').find('ymin').text)
				xmax = float(obj.find('bndbox').find('xmax').text)
				ymax = float(obj.find('bndbox').find('ymax').text)
				w,h,cx,cy,xmin,ymin,xmax,ymax = tranformAnnotation(xmin,ymin,xmax,ymax)
				if(fourch):
					if(ndataset+i+j.replace('.xml','.jpg') not in ddataset):
						ddataset[ndataset+i+j.replace('.xml','.jpg')] = [[dclass[class_name],str(cx),str(cy),str(w),str(h),ndataset+i+j.replace('.xml','.png'),dir_in+'/'+ndataset+'/'+i+'/visible/'+j.replace('.xml','.jpg'),dir_in+'/'+ndataset+'/'+i+'/lwir/'+j.replace('.xml','.jpg'),str(xmin),str(ymin),str(xmax),str(ymax)]]
					else:
						ddataset[ndataset+i+j.replace('.xml','.jpg')].append([dclass[class_name],str(cx),str(cy),str(w),str(h),str(xmin),str(ymin),str(xmax),str(ymax)])
				else:
					if(ndataset+i+j.replace('.xml','.jpg') not in ddataset):
						ddataset[ndataset+i+j.replace('.xml','.jpg')] = [[dclass[class_name],str(cx),str(cy),str(w),str(h),ndataset+i+j.replace('.xml','.jpg'),dir_in+'/'+ndataset+'/'+i+'/visible/'+j.replace('.xml','.jpg'),dir_in+'/'+ndataset+'/'+i+'/lwir/'+j.replace('.xml','.jpg'),str(xmin),str(ymin),str(xmax),str(ymax)]]
					else:
						ddataset[ndataset+i+j.replace('.xml','.jpg')].append([dclass[class_name],str(cx),str(cy),str(w),str(h),str(xmin),str(ymin),str(xmax),str(ymax)])
	csv = open(dir_out+'/'+typee+'/'+typee+'.csv','a+')
	for k,v in ddataset.items():
		f = open(dir_out+'/'+typee+'/images/'+k.replace('.jpg','.txt'),'a+')
		imagen = None
		imagen2 = None
		l_bbox = []
		l_class = []
		dest = None
		for z in v:
			if len(z)==12:
				imagen = z[6]
				
				csv.write(dir_out+'/'+typee+'/images/'+z[5]+'\n')
				if(fourch):
					imagen2 = z[7]
					img = cv2.imread(z[6])
					img2 = cv2.imread(z[7],cv2.IMREAD_GRAYSCALE)
					four = addChannel(img,img2)
					imageio.imwrite(dir_out+'/'+typee+'/images/'+z[5],four)
				else:
					shutil.copy(z[6], dir_out+'/'+typee+'/images/'+z[5])
				if(z[0]=="2"):
					#f.write(z[0]+' '+z[1]+' '+z[2]+' '+z[3]+' '+z[4]+'\n')
					f.write('0 '+z[1]+' '+z[2]+' '+z[3]+' '+z[4]+'\n')
				l_bbox.append(BoundingBox(x1=int(float(z[8])), x2=int(float(z[10])), y1=int(float(z[9])), y2=int(float(z[11]))))
				print(z[0])
				l_class.append(z[0])
				dest = dir_out+'/'+typee+'/images/'+z[5]
			else:
				if(z[0]=="2"):
					#f.write(z[0]+' '+z[1]+' '+z[2]+' '+z[3]+' '+z[4]+'\n')
					f.write('0 '+z[1]+' '+z[2]+' '+z[3]+' '+z[4]+'\n')
				l_class.append(z[0])
				l_bbox.append(BoundingBox(x1=int(float(z[5])), x2=int(float(z[7])), y1=int(float(z[6])), y2=int(float(z[8]))))
		
		#print(four)
		#print(augment)
		if(fourch and augment):
			augmentRGBC(augment,imagen,imagen2,l_bbox,dest,dir_out+'/'+typee+'/images/'+k.replace('.jpg','.txt'),csv,l_class)
		else:
			augmentRGB(augment,imagen,l_bbox,dest,dir_out+'/'+typee+'/images/'+k.replace('.jpg','.txt'),csv,l_class)
		f.close()
	csv.close()
	return ddataset

def main(argv):
	try:
		opts, args = getopt.getopt(argv,"",["train=","test=","in=","out=","n_tr_set=","n_te_set=","4c=","augment="])
		if len(opts)==0:
			usage()
		elif not (int(opts[6][1])):
			print("3 canales")
			if "--train" in opts[0] and "--test" in opts[1] and "--in" in opts[2] and "--out" in opts[3]and "--n_tr_set" in opts[4] and "--n_te_set" in opts[5] and "--4c" in opts[6] and "--augment" in opts[7]:
				processRgb(opts[0][1],opts[1][1],opts[2][1],opts[3][1],opts[4][1],opts[5][1],int(opts[6][1]),int(opts[7][1]))
		else:
			print("4 canales")
			if "--train" in opts[0] and "--test" in opts[1] and "--in" in opts[2] and "--out" in opts[3]and "--n_tr_set" in opts[4] and "--n_te_set" in opts[5] and "--4c" in opts[6] and "--augment" in opts[7]:
				processRgbc(opts[0][1],opts[1][1],opts[2][1],opts[3][1],opts[4][1],opts[5][1],int(opts[6][1]),int(opts[7][1]))
		
	except Exception as e:
		print('Failed to read directory: '+ str(e))
		print(traceback.format_exc())
		sys.exit(2)

def usage():
	print('Usage: annotationreformat2.py --train <train annotation> --test <test annotation> --in <in annotation> --out <output annotation> --n_tr_set<train set name> --n_tr_set<test set name> --augment<true>')
	print('Usage: annotationreformat2.py --train <train annotation> --test <test annotation> --in <in annotation> --out <output annotation> --n_tr_set<train set name> --n_tr_set<test set name> --4c<true> --augment<true>')

def transform(xmin,ymin,xmax,ymax):
	shapex,shapey = 640.0,512.0
	cX = (xmin+xmax)/2
	cY = (ymin+ymax)/2
	w = xmax-xmin
	h = ymax-ymin
	return w/shapex,h/shapey,cX/shapex,cY/shapey

def tranformAnnotation(xmin,ymin,xmax,ymax):
	shapex,shapey = 640.0,512.0
	black = np.zeros((int(shapey),int(shapex),1), np.uint8)
	cv2.rectangle(black,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,255,255),-1)
	contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	M = cv2.moments(contours[0])
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = 0, 0
	w = xmax-xmin
	h = ymax-ymin
	return w/shapex,h/shapey,cX/shapex,cY/shapey,xmin,ymin,xmax,ymax
	
def createDir(path):
	dir_train_exist = os.path.isdir(path+'/train')
	dir_test_exist = os.path.isdir(path+'/test')
	if not(dir_train_exist and dir_test_exist):
		os.makedirs(path+'/train')
		os.makedirs(path+'/train/images')
		os.makedirs(path+'/test')
		os.makedirs(path+'/test/images')
	else:
		shutil.rmtree(path+'/train')
		shutil.rmtree(path+'/test')
		os.makedirs(path+'/train')
		os.makedirs(path+'/train/images')
		os.makedirs(path+'/test')
		os.makedirs(path+'/test/images')

def augmentRGB(augment,img,bbss,out,txt,csv,lclass):
	if(augment):
		image = imageio.imread(img)
		bbs = BoundingBoxesOnImage(bbss,shape=image.shape)
		#son funciones
		rotate = iaa.Affine(rotate=(25, -25))
		crop = iaa.Crop(percent=(0, 0.3))
		shear  = iaa.Affine(shear=(0,20))
		fliph = iaa.Fliplr(p=1.0)
		scale = iaa.Affine(scale={"x": (1.3, 1.3), "y": (1.3, 1.3)})
		cutouti = iaa.Cutout(nb_iterations=2)
		#aug_imgx es la imagen con data augmentation y bbs_aug son los boxes ya aumentados
		aug_img1, bbs_aug1 = rotate(image=image, bounding_boxes=bbs)
		aug_img2, bbs_aug2 = crop(image=image, bounding_boxes=bbs)
		aug_img3, bbs_aug3 = shear(image=image, bounding_boxes=bbs)
		aug_img4, bbs_aug4 = fliph(image=image, bounding_boxes=bbs)
		aug_img6, bbs_aug6 = scale(image=image, bounding_boxes=bbs)
		aug_img7, bbs_aug7 = cutouti(image=image, bounding_boxes=bbs)

		bbs_aug1 = bbs_aug1.remove_out_of_image(fully=True).clip_out_of_image()
		bbs_aug2 = bbs_aug2.remove_out_of_image(fully=True).clip_out_of_image()
		bbs_aug3 = bbs_aug3.remove_out_of_image(fully=True).clip_out_of_image()
		bbs_aug4 = bbs_aug4.remove_out_of_image(fully=True).clip_out_of_image()
		bbs_aug6 = bbs_aug6.remove_out_of_image(fully=True).clip_out_of_image()
		bbs_aug7 = bbs_aug7.remove_out_of_image(fully=True).clip_out_of_image()
		
		#Guardar imagenes
		imageio.imwrite(out.replace(".jpg","r.jpg"),aug_img1)
		imageio.imwrite(out.replace(".jpg","cr.jpg"),aug_img2)
		imageio.imwrite(out.replace(".jpg","sh.jpg"),aug_img3)
		imageio.imwrite(out.replace(".jpg","fh.jpg"),aug_img4)
		imageio.imwrite(out.replace(".jpg","sc.jpg"),aug_img6)
		imageio.imwrite(out.replace(".jpg","co.jpg"),aug_img7)
		csv.write(out.replace(".jpg","r.jpg")+'\n')
		csv.write(out.replace(".jpg","cr.jpg")+'\n')
		csv.write(out.replace(".jpg","sh.jpg")+'\n')
		csv.write(out.replace(".jpg","fh.jpg")+'\n')
		csv.write(out.replace(".jpg","sc.jpg")+'\n')
		csv.write(out.replace(".jpg","co.jpg")+'\n')
		#Recorro los n boxes para graficarlos en una imagen
		f1 = open(txt.replace(".txt","r.txt"),'a+')
		cont = 0
		#rotate
		for bb in bbs_aug1.bounding_boxes:
			cX = (int(bb.x1)+int(bb.x2))/2
			cY = (int(bb.y1)+int(bb.y2))/2
			#cv2.rectangle(aug_img1,(int(bb.x1),int(bb.y1)),(int(bb.x2),int(bb.y2)),(0,255,0,255),1)
			#cv2.circle(aug_img1,(int(cX),int(cY)),2,(0,255,0),-1)
			#cv2.imshow("Display window", aug_img1)
			#cv2.waitKey(0)
			
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f1.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f1.close()
		f2 = open(txt.replace(".txt","cr.txt"),'a+')
		cont = 0
		#crop
		for bb in bbs_aug2.bounding_boxes:
			
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f2.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f2.close()
		f3 = open(txt.replace(".txt","sh.txt"),'a+')
		cont = 0
		#shear
		for bb in bbs_aug3.bounding_boxes:
			
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f3.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f3.close()
		f4 = open(txt.replace(".txt","fh.txt"),'a+')
		cont = 0
		#fliph
		for bb in bbs_aug4.bounding_boxes:
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f4.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f4.close()
		f6 = open(txt.replace(".txt","sc.txt"),'a+')
		cont = 0
		#scale
		for bb in bbs_aug6.bounding_boxes:
			
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f6.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f6.close()
		f7 = open(txt.replace(".txt","co.txt"),'a+')
		cont = 0
		#cutout
		for bb in bbs_aug7.bounding_boxes:
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f7.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f7.close()
		
def augmentRGBC(augment,img,img2,bbss,out,txt,csv,lclass):
	if(augment):
		imageRGB = cv2.imread(img,cv2.IMREAD_UNCHANGED)
		imageFLIR = cv2.imread(img2,cv2.IMREAD_GRAYSCALE)
		(B,G,R) =cv2.split(imageRGB)
		image= cv2.merge((B,G,R,imageFLIR))
		bbs = BoundingBoxesOnImage(bbss,shape=image.shape)
		#son funciones
		rotate = iaa.Affine(rotate=(-25, 25))
		crop = iaa.Crop(percent=(0, 0.3))
		shear  = iaa.Affine(shear=(0,20))
		fliph = iaa.Fliplr(p=1.0)
		scale = iaa.Affine(scale={"x": (1.3, 1.3), "y": (1.3, 1.3)})
		cutouti = iaa.Cutout(nb_iterations=2)
		#aug_imgx es la imagen con data augmentation y bbs_aug son los boxes ya aumentados
		aug_img1, bbs_aug1 = rotate(image=image, bounding_boxes=bbs)
		aug_img2, bbs_aug2 = crop(image=image, bounding_boxes=bbs)
		aug_img3, bbs_aug3 = shear(image=image, bounding_boxes=bbs)
		aug_img4, bbs_aug4 = fliph(image=image, bounding_boxes=bbs)
		aug_img6, bbs_aug6 = scale(image=image, bounding_boxes=bbs)
		aug_img7, bbs_aug7 = cutouti(image=image, bounding_boxes=bbs)
		#Guardar imagenes
		imageio.imwrite(out.replace(".png","r.png"),aug_img1)
		imageio.imwrite(out.replace(".png","cr.png"),aug_img2)
		imageio.imwrite(out.replace(".png","sh.png"),aug_img3)
		imageio.imwrite(out.replace(".png","fh.png"),aug_img4)
		imageio.imwrite(out.replace(".png","sc.png"),aug_img6)
		imageio.imwrite(out.replace(".png","co.png"),aug_img7)
		csv.write(out.replace(".png","r.png")+'\n')
		csv.write(out.replace(".png","cr.png")+'\n')
		csv.write(out.replace(".png","sh.png")+'\n')
		csv.write(out.replace(".png","fh.png")+'\n')
		csv.write(out.replace(".png","sc.png")+'\n')
		csv.write(out.replace(".png","co.png")+'\n')
		#Recorro los n boxes para graficarlos en una imagen
		f1 = open(txt.replace(".txt","r.txt"),'a+')
		cont = 0
		for bb in bbs_aug1.bounding_boxes:
			cX = (int(bb.x1)+int(bb.x2))/2
			cY = (int(bb.y1)+int(bb.y2))/2
			#cv2.rectangle(aug_img1,(int(bb.x1),int(bb.y1)),(int(bb.x2),int(bb.y2)),(0,255,0,255),1)
			#cv2.circle(aug_img1,(int(cX),int(cY)),2,(0,255,0),-1)
			#cv2.imshow("Display window", aug_img1)
			#cv2.waitKey(0)
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f1.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f1.close()
		f2 = open(txt.replace(".txt","cr.txt"),'a+')
		cont = 0
		for bb in bbs_aug2.bounding_boxes:
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f2.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f2.close()
		f3 = open(txt.replace(".txt","sh.txt"),'a+')
		cont = 0
		for bb in bbs_aug3.bounding_boxes:
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f3.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f3.close()
		f4 = open(txt.replace(".txt","fh.txt"),'a+')
		cont = 0
		for bb in bbs_aug4.bounding_boxes:
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f4.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f4.close()
		f6 = open(txt.replace(".txt","sc.txt"),'a+')
		cont = 0
		for bb in bbs_aug6.bounding_boxes:
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f6.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f6.close()
		f7 = open(txt.replace(".txt","co.txt"),'a+')
		cont = 0
		for bb in bbs_aug7.bounding_boxes:
			w,h,cx,cy = transform(int(bb.x1),int(bb.y1),int(bb.x2),int(bb.y2))
			f7.write(lclass[cont]+' '+str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+'\n')
			cont+=1
		f7.close()

def addChannel(img1,img2):
	imgRGB=img1
	alpha=img2
	(B,G,R)=cv2.split(imgRGB)
	imgFour=cv2.merge([B,G,R,alpha])
	return imgFour

def processRgb(dir_train,dir_test,dir_in,dir_out,ntrain,ntest,four,augment):
	dclass = {'person':'0','people':'1','cyclist':'2'}
	dtrain = {}
	dtest = {}
	exist_train = os.path.isdir(dir_train)
	exist_test = os.path.isdir(dir_test)
	if(exist_train and exist_test):
		#createDir(dir_out)
		x1 = threading.Thread(target=dataset, args=(dir_train,ntrain,dir_in,dir_out,augment,'train',four,dtrain))
		x1.start()
		x2 = threading.Thread(target=dataset, args=(dir_test,ntest,dir_in,dir_out,augment,'test',four,dtest))
		x2.start()
		x1.join()
		x2.join()

def processRgbc(dir_train,dir_test,dir_in,dir_out,ntrain,ntest,four,augment):
	dclass = {'person':'0','people':'1','cyclist':'2'}
	dtrain = {}
	dtest = {}
	exist_train = os.path.isdir(dir_train)
	exist_test = os.path.isdir(dir_test)
	if(exist_train and exist_test):
		#createDir(dir_out)
		x1 = threading.Thread(target=dataset, args=(dir_train,ntrain,dir_in,dir_out,augment,'train',four,dtrain))
		x1.start()
		x2 = threading.Thread(target=dataset, args=(dir_test,ntest,dir_in,dir_out,augment,'test',four,dtest))
		x2.start()
		x1.join()
		x2.join()

if __name__ == "__main__":
   main(sys.argv[1:])
