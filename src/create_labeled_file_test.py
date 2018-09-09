import re
import glob

with open("path_and_label_test.txt", "w") as f:

    l=glob.glob("/home/seimei/Graduation_Research/dataset_valid/kumori/class4-5/*jpg")
    for i in l:
        f.write(i+" "+"1"+"\n")

    l = glob.glob("/home/seimei/Graduation_Research/dataset_valid/kumori/class4-6/*.jpg")
    for i in l:
        f.write(i+" "+"1"+"\n")

    l = glob.glob("/home/seimei/Graduation_Research/dataset_valid/hare/class3-1/*.jpg")
    for i in l:
        f.write(i+" "+"0"+"\n")

    
    l=glob.glob("/home/seimei/Graduation_Research/dataset_valid/hare/class3-2/*.jpg")
    for i in l:
        f.write(i+" "+"0"+"\n")


