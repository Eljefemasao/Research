import re
import glob

with open("path_and_label_train.txt", "w") as f:

    
    l=glob.glob("/home/seimei/Graduation_Research/dataset/hare/class1-1/*.jpg")
    for i in l:
        f.write(i+" "+"0"+"\n")

    l=glob.glob("/home/seimei/Graduation_Research/dataset/hare/class1-3/class1-3_video1_cropped/*.jpg")
    for i in l:
        f.write(i+" "+"0"+"\n")

    l=glob.glob("/home/seimei/Graduation_Research/dataset/kumori/class2-2/*.jpg")
    for i in l:
        f.write(i+" "+"1"+"\n")

    l=glob.glob("/home/seimei/Graduation_Research/dataset/kumori/class2-3/*.jpg")
    for i in l:
        f.write(i+" "+"1"+"\n")

