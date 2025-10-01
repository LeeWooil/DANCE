import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res


def crop(infile, height, width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def averagePixels(path):
    r, g, b = 0, 0, 0
    count = 0
    pic = Image.open(path)
    for x in range(pic.size[0]):
        for y in range(pic.size[1]):
            imgData = pic.load()
            tempr, tempg, tempb = imgData[x, y]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    return (r / count), (g / count), (b / count), count

def convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray


def prepare_dirs(keyframePath):
    if not os.path.exists(keyframePath):
        os.makedirs(keyframePath)
def prepare_csv_dirs(csvPath):
    if not os.path.exists(csvPath):
        os.makedirs(csvPath)

def plot_metrics(indices, lstfrm, lstdiffMag):
    y = np.array(lstdiffMag)
    plt.plot(indices, y[indices], "x")
    l = plt.plot(lstfrm, lstdiffMag, 'r-')
    plt.xlabel('frames')
    plt.ylabel('pixel difference')
    plt.title("Pixel value differences from frame to frame and the peak values")
    plt.show()

def save_keyframes_as_video(keyframePath, output_video, frame_size, fps=50):
    images = sorted(
        [img for img in os.listdir(keyframePath) if img.endswith(".jpg")],
        key=lambda x: int(x.replace("keyframe", "").replace(".jpg", ""))
    )
    
    if not images:
        print("⚠️ No saved keyframes found.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    for img_name in images:
        img_path = os.path.join(keyframePath, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        out.write(frame)

    out.release()