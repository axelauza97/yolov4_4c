# yolov4_4c

1. annotationreformat:
Script make the division for train and test folders required for training at darknet yolo

`<path-to-train-dataset-lwir-visible>` -> KAIST 

`<path-to-test-dataset-lwir-visible>` -> KAIST

4c->
-    0 for 3 channel

-	1 for 4 channel

augment->	
- 0 for without data augmentation
		
- 1  with data augmentation
		

2. fourchannel:
Script that takes result json from yolov4_4c detection* and create a video of double width resolution where left side is RGB and right side is pseudocolor LWIR.


3. savevideo:
Script that creates groundtruth tag video and video without tags 

	
4. mosaic:
Alternate way of creating mosaic data augmentation technique. Is recomended to use Darknet YOLOv4 mosaic flag

# USAGE
python3 annotationreformat.py --train=`<path-to-train-xml-annotation>` --test=`<path-to-test-xml-annotation> --in=<path-to-train-dataset> --out=<path-store-yolo-folders-test-train-img-txt>` --n_tr_set=`<train-set-#>` --n_te_set=`<test-set-#>` --4c=`<bool>` --augment=`<bool>`


python fourchannel.py <detection-json-from-darknetyolo> <csv-path-to-images> 


python3 savevideo.py `<path-to-test-csv>`


# EXAMPLE
python3 annotationreformat.py --train=/home/axelauza/annotations-xml/set03 --test=/home/axelauza/annotations-xml/set09 --in=/home/axelauza/Documentos/MI/dataset --out=/home/axelauza/Documentos/MI/tes --n_tr_set=set03 --n_te_set=set09 --4c=0 --augment=1


python fourchannel.py Descargas/result.json new.csv


python3 savevideo.py Documentos/MI/RGBC/test/test.csv


python3 mosaic.py /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-2.jpg  /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-3.jpg  /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-4.jpg  /home/axelauza/Descargas/FIRMEZA/SENSITIVO1711_2/corteRgb/Arandano\ 1-5.jpg



*darknet.exe detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < data/train.txt


# Demostration Videos
- https://espolec-my.sharepoint.com/:f:/g/personal/aaauza_espol_edu_ec/Euo_yn7kqzRIvx6rC3q8DSgBafi1SaPn_xtyZPWVufGAWg
