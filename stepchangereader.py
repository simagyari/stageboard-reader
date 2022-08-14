import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.style as style
import matplotlib.ticker as ticker
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Get image data and orthorectify it
# Image rectifier function
def image_rectifier(img_path, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height):
    img = cv2.imread(img_path)  # read image in visible light
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # transform image to grayscale
    img = img[img_ymin:img_ymax, img_xmin:img_xmax]  # crop image next to stageboard
    tr = cv2.getPerspectiveTransform(ort_src, ort_dst)  # get transformation matrix from manual points
    img_rect = cv2.warpPerspective(img, tr, (warp_width, warp_height))  # orthorectify image
    th = np.percentile(img_rect, 75)  # 70th percentile of image value as threshold
    return img_rect, th


# Stageboard reader
def stage_reader(img_path, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height,
                 real_width, real_length):
    # Read timestamp from image name
    timestamp = os.path.basename(img_path)[-18:-4]
    ts_date = '-'.join([timestamp[0:4], timestamp[4:6], timestamp[6:8]])
    ts_time = ':'.join([timestamp[8:10], timestamp[10:12], timestamp[12:]])
    # Create pandas datetime object
    timestamp_dt = pd.to_datetime(ts_date + ' ' + ts_time)
    print(timestamp_dt)
    # Call image_rectifier to rectify and crop image
    img_rect, th = \
        image_rectifier(img_path, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height)
    # Threshold
    thresh = cv2.threshold(img_rect, th, np.max(img_rect), cv2.THRESH_BINARY)[1]
    whites = [np.count_nonzero(thresh[:, i]) for i in range(0, len(thresh[0]))]
    # Get largest step change in whites
    # Adapted from https://stackoverflow.com/a/48001937/18668457
    white_diffs = np.array([i - np.mean(whites) for i in whites])  # differences from the mean
    white_diffs -= np.average(white_diffs)  # subtract average from differences
    step = np.hstack((np.ones(len(white_diffs)), -1 * np.ones(len(white_diffs))))
    white_step = np.convolve(white_diffs, step, mode='valid')
    step_index = np.argmax(white_step)  # get highest step
    step_index_2 = np.argmin(white_step)  # get the lowest step (highest step down)
    # Get white rows inside the columns
    white_rows = [np.count_nonzero(thresh[k, step_index:step_index_2]) for k in range(0, len(thresh))]
    # Get row indices where stageboard is white (more than 60% of stageboard width is exposed)
    white_row_indices = [idx for idx, val in enumerate(white_rows) if val > ((step_index_2 - step_index) * 0.6)]
    # Iterate over the row indices in the list to discover the larger gaps
    try:
        for nr in range(0, (len(white_row_indices) - 15)):
            # Only engage in the lower half of the image to counteract low reflectance in the upper image part
            if nr > len(img_rect)/2:
                nrs_should = set(range(white_row_indices[nr], white_row_indices[nr] + 15))  # set of theoretical next values
                nrs_are = set(white_row_indices[nr:nr + 15])  # set of actual next values
                # Find common values in two sets
                if len(nrs_are & nrs_should) > 1:
                    continue
                else:
                    white_row_indices = white_row_indices[:nr]  # cut off list at last row that is part of the stageboard
                    print(f'Stageboard ends at pixel {white_row_indices[nr - 1]}!')
                    break
        # Get width to height ratio
        ratio = white_row_indices[-1] / (step_index_2 - step_index)
        # Get stage with real-life stageboard width and length
        stage = real_length - (real_width * ratio)
    # if the whites are returned empty due to some vegetation altering reflection, stage is set to -100
    except IndexError:
        stage = -100
    return {'Timestamp': timestamp_dt, 'Stage': stage}


# Main function
def main() -> None:
    # Site-specific camera calibration and orthorectification variables
    img_xmin = 1350
    img_xmax = 1700
    img_ymin = 500
    img_ymax = 1540
    ort_src = np.float32([[206, 54], [156, 578], [239, 610], [291, 73]])
    ort_dst = np.float32([[43, 0], [43, 435], [130, 435], [130, 0]])
    warp_width = 173
    warp_height = 900
    # Stageboard width and length (cm)
    real_width = 11
    real_length = 100
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
        # Crop and orthorectify image, obtain result
        res_dict = \
            stage_reader(
                img_path, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height,
                real_width, real_length)
        results = results.append(res_dict, ignore_index=True)
    outpath = 'F:\\RESULTS\\GENERATED' + os.path.basename(fpath) + '_' + str(start) + '_' + str(end) + '_stepchange.csv'
    results.to_csv(outpath, index=False)
    print(f'Results are in {outpath} file!')
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
