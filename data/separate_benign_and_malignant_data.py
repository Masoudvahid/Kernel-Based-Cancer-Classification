# place this script in the same hierarchy as "MIASimages" directory

import os
import sys
from PIL import Image
import shutil

# path to all images
#all_dataset_path = "D:/PhD and Research/Research_Related/PhD PDPU/DC/DC 5/Implementation/LatestProcessesMIAS/Enhanced/"

#all_dataset_path = "D:/Drive D/PhD/DC/Dataset/Dataset to be published/RajivImages_Final/New folder/parita/Final Dataset/Enhanced/"
all_dataset_path ="/home/masoud/Uni/healthCareResearch/data/TIFF Images/all/"

# path to save benign images 
#cleaned_dataset_path_b = "D:/Drive D/PhD/DC/Dataset/Dataset to be published/RajivImages_Final/New folder/parita/Final Dataset/Enhanced/Benign/"
cleaned_dataset_path_b = "/home/masoud/Uni/healthCareResearch/data/TIFF Images/Benign/"

# path to save malignant images 
#cleaned_dataset_path_m = "D:/Drive D/PhD/DC/Dataset/Dataset to be published/RajivImages_Final/New folder/parita/Final Dataset/Enhanced/Malignant/"
cleaned_dataset_path_m = "/home/masoud/Uni/healthCareResearch/data/TIFF Images/Malignant/"
# path to save normal images 
#cleaned_dataset_path_n = "D:/Drive D/PhD/DC/Dataset/Dataset to be published/RajivImages_Final/New folder/parita/Final Dataset/Enhanced/Normal/"
cleaned_dataset_path_n = "/home/masoud/Uni/healthCareResearch/data/TIFF Images/Normal/"

cleaned_dataset_path_Notn = "/home/masoud/Uni/healthCareResearch/data/TIFF Images/notNormal/"

Norm=0
N=0
B=0
M=0

# open 'annotation.txt'
# annotation.txt is created using the information about the
# images from the MIAS database
print('Processing...')
with open('Info (copy).txt', 'r') as file:
#    line = file.readlines()  	
    for line in file:
        words = line.split()
        if 'NORM' in words or 'B' in words:
            # source file from "all_mias"  directory
            src2 = all_dataset_path + str(line.replace('\t','  ').split('  ')[0]) + '.tif'

            # destination file for malignant images            
            dst2 = cleaned_dataset_path_n + str(line.replace('\t','  ').split('  ')[0]) + '.tif'

            # copy normal images to 'dataset/normal/'      
            shutil.copy2(src2, dst2)
            Norm += 1
            continue
        elif 'N' not in words and 'B' not in words:
            src2 = all_dataset_path + str(line.replace('\t','  ').split('  ')[0]) + '.tif'

            dst2 = cleaned_dataset_path_Notn + str(line.replace('\t','  ').split('  ')[0]) + '.tif'

            shutil.copy2(src2, dst2)
            N += 1



        if 'B' in words:
            # source file from "MIASimages"  directory
            #src = all_dataset_path + str(line.split(' ')[0]) + '.tiff'
            src = all_dataset_path + str(line.replace('\t','  ').split('  ')[0]) + '.tif'
            

            # destination file for benign images
            dst = cleaned_dataset_path_b + str(line.replace('\t','  ').split('  ')[0]) + '_benign.tif'

            # copy benign images to 'dataset/benign/'                                       
            shutil.copy2(src, dst)
            B += 1
                                                    
        

        if 'M' in words:
            # source file from "all_mias"  directory
            src1 = all_dataset_path + str(line.replace('\t','  ').split('  ')[0]) + '.tif'

            # destination file for malignant images            
            dst1 = cleaned_dataset_path_m + str(line.replace('\t','  ').split('  ')[0]) + '_malignant.tif'

            # copy malignant images to 'dataset/malignant/'      
            shutil.copy2(src1, dst1)
            M += 1

print(f'{Norm=}, {N=}, {B=}, {M=}')
print('Finished processing!')
