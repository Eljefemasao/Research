
with open("path_and_label.txt", "w") as f:

    count = 0
    for count in range(1, 191):
        f.write("/Users/masaaki/akamineresearch/dataset/class1-1/class1-1_video1_cropped/image_" + str(0) + str(count).zfill(3) + ".jpg" + " "+ "1" + "\n")

    count = 0
    for count in range(1,183):
        f.write("/Users/masaaki/akamineresearch/dataset/class1-3/class1-3_video1_cropped/image_" + str(0) + str(count).zfill(3)+ ".jpg"+ " " + "1" + "\n")

    count = 0
    for count in range(1,184):
        f.write("/Users/masaaki/akamineresearch/dataset/class3-2/class3-2_video1_cropped/image_" + str(0) + str(count).zfill(3)+ ".jpg"+ " " + "1" + "\n")

    count = 0
    for count in range(1,148):
        f.write("/Users/masaaki/akamineresearch/dataset/class4-2/class4-2_video1_cropped/image_" + str(0) + str(count).zfill(3)+ ".jpg"+ " " + "1" + "\n")
        
    count = 0
    for count in range(1,112):
        f.write("/Users/masaaki/akamineresearch/dataset/negative_cropped/image_" + str(0) + str(count).zfill(3)+ ".jpg"+ " " + "0" + "\n")
    
    
