import os
import cv2
import numpy as np
import time
import peakutils
import argparse
from Concept_exraction.utils.keyframe_selection_utils import convert_frame_to_grayscale, prepare_dirs

def keyframeDetection(source, dest, Thres ,plotMetrics=False, verbose=False, thres_len=70, thres_len2=200):
    video_name = os.path.splitext(os.path.basename(source))[0] 
    video_dest = os.path.join(dest, video_name)  

    keyframePath = os.path.join(video_dest, 'keyFrames')
    imageGridsPath = os.path.join(video_dest, 'imageGrids')
    csvPath = os.path.join(video_dest, 'csvFile')
    path2file = os.path.join(csvPath, 'output.csv')
    txt_file = os.path.join(csvPath, f'{video_name}.txt')
    prepare_dirs(keyframePath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print(f"❌ Error opening video file: {source}")
        return

    lstfrm, lstdiffMag, timeSpans, images, full_color = [], [], [], [], []
    lastFrame = None
    start_time = time.process_time()

    for _ in range(length):
        ret, frame = cap.read()
        if not ret:
            break
        
        grayframe, blur_gray = convert_frame_to_grayscale(frame)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)

        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.absdiff(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        timeSpans.append(time.process_time() - start_time)
        lastFrame = blur_gray

    cap.release()

    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    if length > thres_len :
        Thres = Thres + 0.2
    elif length <= thres_len:
        pass
    indices = peakutils.indexes(y - base, Thres, min_dist=1)



    for cnt, x in enumerate(indices, start=1):
        cv2.imwrite(os.path.join(keyframePath, f'keyframe{cnt}.jpg'), full_color[x])
        log_message = [f'keyframe {cnt} happened at {timeSpans[x]} sec.']
        if verbose:
            print(log_message[0])




    cv2.destroyAllWindows()


def process_all_videos(input_folder, dest, Thres, plotMetrics, verbose, thres_len):
    video_files = []

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print("⚠️ No MP4 videos found in the input folder.")
        return

    print(f"📂 Processing {len(video_files)} videos...")

    for video in video_files:
        video_path = os.path.join(input_folder, video)
        keyframeDetection(video_path, dest, Thres, plotMetrics, verbose,thres_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Keyframe Detection for Multiple Videos")

    parser.add_argument("--source", type=str, help="Path to the input video file")
    parser.add_argument("--dest", type=str, help="Path to the output directory")
    parser.add_argument("--Thres", type=float, help="Threshold for keyframe detection")
    parser.add_argument("--plot", type=bool, help="Enable plotting metrics")
    parser.add_argument("--verbose", type=bool, help="Print keyframe logs")
    parser.add_argument("--thres_len", type=int, help="Length for thresholding")
    args = parser.parse_args()
    t0 = time.time()
    process_all_videos(args.source, args.dest, args.Thres,
                       args.plot, args.verbose, args.thres_len)
    t1 = time.time()
    elapsed = t1 - t0
    print(f"⏱️ Keyframe JPG saving time: {elapsed:.2f} sec")
