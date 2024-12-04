from icrawler.builtin import BingImageCrawler
import cv2
import numpy as np
import os

def add_pic():
    number = 100
    bing = BingImageCrawler(storage={'root_dir': 'your_image_dir'})
    bing.crawl(keyword='dog images', filters=None, max_num=number, offset=0)

def store_raw_images():
    directory = '/media/dreamer/K2LIB/data/n'
    pic_num = 1

    for filename in os.listdir(directory):
        try:
            print(pic_num)
            path = os.path.join(directory, filename)
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (100, 100))
            cv2.imwrite("/home/dreamer/haar_workspace/neg/" + str(pic_num) + ".jpg", img)
            pic_num += 1
        except Exception as e:
            print(str(e))


def add_file():
    directory = '/home/dreamer/haar_workspace/neg'

    for img in os.listdir(directory):
        line = 'neg/' + img +'\n'
        with open('bg.txt', 'a') as f:
            f.write(line)

if __name__ == '__main__':
    add_file()