import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.style as style
import matplotlib.ticker as ticker
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Image rectifier function
def image_rectifier(img_path, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height):
    img = cv2.imread(img_path)  # read image in visible light
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # transform image to grayscale
    img = img[img_ymin:img_ymax, img_xmin:img_xmax]  # crop image next to stageboard
    tr = cv2.getPerspectiveTransform(ort_src, ort_dst)  # get transformation matrix from manual points
    img_rect = cv2.warpPerspective(img, tr, (warp_width, warp_height))  # orthorectify image
    th = np.percentile(img_rect, 75)  # chosen percentile of the image gets above threshold
    return img_rect, th


# Stageboard reader function
def stage_reader(img, img_path, real_width, th):
    # Get time data
    timestamp = os.path.basename(img_path)[-18:-4]
    ts_date = '-'.join([timestamp[0:4], timestamp[4:6], timestamp[6:8]])
    ts_time = ':'.join([timestamp[8:10], timestamp[10:12], timestamp[12:]])
    # Create pandas datetime object
    timestamp_dt = pd.to_datetime(ts_date + ' ' + ts_time)
    # Threshold (binary threshold at 95% of maximum value) image
    thresh = cv2.threshold(img, th, np.max(img), cv2.THRESH_BINARY)[1]
    # Merge stageboard into single object
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35)))
    # Retrieve contours of the image
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(contours, key=cv2.contourArea)  # get contour of only the largest object (stageboard)
    except ValueError:
        stage = -100
        return {'Timestamp': timestamp_dt, 'Stage': stage}
    # Construct output image
    out = np.zeros(img.shape, np.uint8)  # empty output
    # Draw filled contour on output image
    cv2.drawContours(out, [cnt], -1, 150, cv2.FILLED)
    # Obtain minimum bounding box (angled, not corrected for distortion)
    min_bb = cv2.minAreaRect(cnt)
    bb_points = cv2.boxPoints(min_bb)  # corners of rectangle
    box = np.int0(bb_points)  # create box shape
    cv2.drawContours(out, [box], -1, 200, 1)  # cv2.FILLED)  # draw min bounding rect on output
    # Get bounding rectangle dimensions
    # Length of two sides and diagonal
    distances = [np.int(np.round(np.linalg.norm(point - bb_points[0]))) for point in bb_points]
    distances.sort()  # ascending sort
    # Get width and height (second and third items, as first is 0 and fourth is diagonal in sorted list)
    width, height = distances[1], distances[2]  # get edges
    # Get result (100 cm full stageboard length, 11 cm stageboard width
    try:
        stage = 100 - (height / width * real_width)  # 11 cm assumed stageboard width
    except ZeroDivisionError:
        stage = -100
    print(f'The stage on this image at {timestamp_dt} is {stage} cm!')
    return {'Timestamp': timestamp_dt, 'Stage': stage}


# Main function, defines parameters and calls other functions
def main() -> None:
    # Site-specific camera calibration and orthorectification variables
    # Image crop extent variables
    img_xmin = 1350
    img_xmax = 1700
    img_ymin = 500
    img_ymax = 1540
    # Perspecitve transformation point lists
    ort_src = np.float32([[206, 54], [156, 578], [239, 610], [291, 73]])
    ort_dst = np.float32([[43, 0], [43, 435], [130, 435], [130, 0]])
    # Perspective transformation image width and length
    warp_width = 173
    warp_height = 900
    # Stageboard width (cm)
    st_width = 11
    # Set filepath
    fpath = r"F:\FIELD\FIELD_WESSENDEN_2"
    # Number of images to start and end with
    start, end = 4, len(os.listdir(fpath))
    # Make list of images in folder
    flist = os.listdir(fpath)[start:end]
    # Create empty dataframe for results
    results = pd.DataFrame(columns=['Timestamp', 'Stage'])
    for fname in flist:
        # Construct image path
        img_path = os.path.join(fpath, fname)
        # Crop and orthorectify image
        img_rect, th = \
            image_rectifier(
                img_path, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height)
        # Obtain result
        res_dict = stage_reader(img_rect, img_path, st_width, th)
        results = results.append(res_dict, ignore_index=True)
    results.to_csv(
        'F:\\RESULTS\\GENERATED' + os.path.basename(fpath) + '_' + str(start) + '_' + str(end) + '_box.csv', index=False
    )
    # Plot rough graph of results
    locator = mdates.DayLocator()
    style.use("classic")
    fig, (ax1) = plt.subplots(nrows=1, figsize=(10, 12), facecolor="white")
    ax1.plot(results["Timestamp"], results["Stage"], color="blue")
    # ax1.plot(board_exposed["DATE"], board_exposed["EXPOSED"], color="blue")
    ax1.set_ylabel("STAGE (cm)", fontsize=12)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax1.xaxis.set_major_locator(locator)
    ax1.set_ylim(0, 100)
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    main()
