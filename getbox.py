from PIL import ImageDraw,Image
import numpy as np
import os

from threading import Thread
from multiprocessing import Process
import time          
  
def getnewimage():

    a = 12

    image = Image.open("/home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/Example/zuijiandan/00001_lr-0.png")
    mask_data = np.load("/home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/Example/zuijiandan/00001_lr-0.npz")
    mask = mask_data['mask']
 
    for c in range(13):
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                if mask[c][x][y] == True:
                    image.putpixel((x, y), (255, 255, 255, 255))
    image.save("/home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/Example/zuijiandan/new1.png")
    
    image1 = Image.open("/home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/Example/zuijiandan/00001_lr-0wrinkles.png")
    mask_data1 = np.load("/home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/Example/zuijiandan/00001_lr-0wrinkles.npz")
    mask1 = mask_data1['mask']
 
    for c in range(13):
        for x in range(image1.size[0]):
            for y in range(image1.size[1]):
                if mask1[c][x][y] == True:
                    image1.putpixel((x, y), (255, 255, 255, 255))
    image1.save("/home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/Example/zuijiandan/new2.png")
#getnewimage()






def getnewimage1(image_path,mask_path):
    img = Image.open(image_path)
    try:
        mask_data = np.load(mask_path)
    except Exception as e:
        print(e)
        return
    mask = mask_data['mask']
    for c in range(13):
        image = Image.open(image_path)
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                if mask[c][x][y] == True:
                    image.putpixel((x, y), (255, 255, 255, 255))
        image.save("/data/0shared_data/ptb__ECGImage/ptbxl_data_new_change_with_log/test/"+image_path[-14:-4]+ str(c) +".png")



path = "/data/0shared_data/ptb__ECGImage/ptbxl_data_new_change_with_log/records100/00000/"
images = os.listdir(path)
images.sort()
n = 0
for index, image in enumerate(images):

    if len(image) != 14 or "png" not in image:
        continue
 
    image_path = path + image
    mask_path = "/data/0shared_data/ptb__ECGImage/ptbxl_data_new_change_with_log/newmask/" + image[:-4] + ".npz"
    getnewimage1(image_path,mask_path)
    n = n + 1


    
    

'''
def work1():
    start_time = time.time()
    #os.system("python /home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/gen_ecg_images_from_data_batch.py -i /data/0shared_data/ptb__ECGImage/test/putin -o /data/0shared_data/ptb__ECGImage/test/output --random_add_header 0.5 --num_columns 2 --random_dc 0.5 --bbox --random_grid_color --store_text_bounding_box --wrinkles")
    os.system("python /home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/gen_ecg_images_from_data_batch.py -i /data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/records100 -o /data/0shared_data/ptb__ECGImage/test/100output --random_add_header 0.5 --num_columns 2 --random_dc 0.5 --bbox --random_grid_color --store_text_bounding_box --wrinkles")
    os.system("python /home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/gen_ecg_images_from_data_batch.py -i /data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/records500 -o /data/0shared_data/ptb__ECGImage/test/500output --random_add_header 0.5 --num_columns 2 --random_dc 0.5 --bbox --random_grid_color --store_text_bounding_box --wrinkles")

    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time)) 

work1() 
#path = "/data/0shared_data/ptb__ECGImage/test/output"
#paths = os.listdir(path)
#for i in paths:
 #   if ".npz" in i:
  #      print(i,i[:-4])
   #     getnewimage1(i[:-4])

'''

'''
def work2():
    for i in range(22):
        if i <= -1:
            continue
        print(str(i) + "begin")
        if i <= 9:
            num = "0" + str(i)
        else:
            num = str(i)
        os.system("python /home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/gen_ecg_images_from_data_batch.py -i /data/0shared_data/ptb-xl/physionet.org/files/ptb-xl/1.0.3/records100/" + num + "000 -o /data/0shared_data/ptb__ECGImage/ptbxl_data_new_change_with_log/records100/" + num + "000 --random_add_header 0.5 --num_columns 2 --random_dc 0.5 --bbox --random_grid_color --store_text_bounding_box --wrinkles")

work2()
'''