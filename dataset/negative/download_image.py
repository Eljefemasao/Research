import urllib.request
import io
from PIL import Image

import os
from http.client import HTTPException
from ssl import CertificateError
from keras.preprocessing.image import img_to_array

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
MINIMUM_FILE_SIZE = 300


def download_image(url, download_path):
    """
    Downloads a single image from a url to a specific path
    :param url: url of image
    :param download_path: full path of saved image file
    :return: true if successfully downloaded, false otherwise
    """

    print("Downloading from " + url)

    try:
        fd = urllib.request.urlopen(url, timeout=3)
        image_file = io.BytesIO(fd.read())
        image = Image.open(image_file)

        size = image.size
        if size[0] < IMAGE_WIDTH or size[1] < IMAGE_HEIGHT:  # Image too small
            return False

        #resized = image.rezise((IMAGE_WIDTH, IMAGE_HEIGHT))
        image.save(download_path, 'jpeg')
    except (IOError, HTTPException, CertificateError) as e:
        print(e)
        return False

    # Check if photo meets minimum size requirement
    size = os.path.getsize(download_path)
    if size < MINIMUM_FILE_SIZE:
        os.remove(download_path)
        print("Invalid Image: " + url)
        return False

    # Try opening as array to see if there are any errors
    try:
        img_to_array(download_path)
    except ValueError as e:
        os.remove(download_path)
        return False

    return True


def main():
    s=download_image("http://farm3.static.flickr.com/2208/1969759803_94b1f1c09d.jpg", "/Users/masaaki/akamineresearch/dataset/negative/dog/1969759803_94b1f1c09d.jpg")
    print(s)


if __name__ == "__main__":
    main()