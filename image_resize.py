import os
import sys
import shutil
from PIL import Image
from resizeimage import resizeimage
import pandas as pd

# python image_resize.py [the size you want] [how many percentage you want to crop]

def resize_file(csv_file, source_dir, target_dir, target_size, ratio):
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        file_name = row['FileName']

        file_path = os.path.join(target_dir, file_name)
        source_path = os.path.join(source_dir, file_name)

        with open(source_path, 'r+b') as f:
            with Image.open(f) as image:
                width = int(image.width * ratio)
                image = resizeimage.resize_crop(image, [width, width])
                cover = resizeimage.resize_width(image, target_size)

                cover.save(file_path, image.format)


size = 227
ratio = 1.0
source = '/scratch/liaoi/images'

if len(sys.argv) == 2:
    size = int(sys.argv[1])
if len(sys.argv) == 3:
    ratio = float(sys.argv[2])
    size = int(sys.argv[1])

target = 'images_all' + str(ratio) + '_' + str(size)

source_dir = os.path.normpath(source)
parent_dir = os.path.dirname(source_dir)
target_dir = os.path.join(parent_dir, target)

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

os.makedirs(target_dir)

resize_file('./train_all.csv', source_dir, target_dir, size, ratio)
resize_file('./test_all.csv', source_dir, target_dir, size, ratio)
