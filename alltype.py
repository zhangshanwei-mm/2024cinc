import os


for i in range(100):
    os.system("python /home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/gen_ecg_images_from_data_batch.py -i /home/wangkexin/ecg-image-kit-main/codes/ecg-image-generator/Example/RawData -o /data/0shared_data/AI_ECG_IMAGE_MAIN_Train_val_test/alltype/num_3/" + str(i) + " --pad_inches 1 --print_header --random_dc 1 --random_grid_color --wrinkles -ca 180 -nv 20 -nh 20 --augment -rot 10 -noise 100 -c 0.1 -t 70000 --hw_text  -n 15 --x_offset 1000 --y_offset 1000")