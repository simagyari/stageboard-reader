import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sp
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import cv2 as cv
import os
from numpy import asarray
from PIL import Image
from datetime import datetime


# Image rectifier function
def image_rectifier(img_path, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height):
    img = cv.imread(img_path)  # read image in visible light
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # transform image to grayscale
    img = img[img_ymin:img_ymax, img_xmin:img_xmax]  # crop image next to stageboard
    gray = gray[img_ymin:img_ymax, img_xmin:img_xmax]  # crop gray image to stageboard
    tr = cv.getPerspectiveTransform(ort_src, ort_dst)  # get transformation matrix from manual points
    img_rect = cv.warpPerspective(img, tr, (warp_width, warp_height))  # orthorectify image
    gray_rect = cv.warpPerspective(gray, tr, (warp_width, warp_height))  # orthorectify grayscale image
    th = np.percentile(img_rect, 75)  # chosen percentile of the image gets above threshold
    return img_rect, gray_rect, th


# String containing location where photos are
# direct = "C://Users//minda//Desktop//WORK_2020//PYTHON_EXP//RIVER_GAUGE_EXP//EXP_IMAGES_WESS_06_CIARA"
direct = r"F:\FIELD\FIELD_WESSENDEN_2"
os.chdir(direct)

# Conversion to account for angle of photo/distortion etc. This is a rough
# and ready estimate and needs to carried out properly in field for new data
# and need to be carried out more extensively for our old photos.
# This is quick estimate only based on few truthing points

# U = (633 - 41) / 50

# Change length of list to discover
start, end = 4, len(os.listdir(direct))

# Change directory to that where files are
file_list = os.listdir()[start:end]

# Use to examine any one photo in sequence
look = 2

# Create series of empty lists to employ later
True_colour_image_list = []  # list of true colour images
Greyscale_image_list = []  # list of grayscale images

range_max = []
range_min = []
date_times = []
top_70 = []

# Set parameters for image rectification
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
# Width of stageboard in pixels
pixel_width = 87
# Width of stageboard in cm
real_width = 11

# For each picture in the folder open with Open cv and get the date/time taken
# and add to list
for file in file_list:
    img_rect, gray_rect, th = \
        image_rectifier(file, img_xmin, img_xmax, img_ymin, img_ymax, ort_src, ort_dst, warp_width, warp_height)
    # img = cv.imread(file)
    # Read timestamp from image name
    timestamp = os.path.basename(file)[-18:-4]
    ts_date = '-'.join([timestamp[0:4], timestamp[4:6], timestamp[6:8]])
    ts_time = ':'.join([timestamp[8:10], timestamp[10:12], timestamp[12:]])
    # Create pandas datetime object
    timestamp_dt = pd.to_datetime(ts_date + ' ' + ts_time)
    date_times.append(timestamp_dt)

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Make list of cropped photos numbers here are the coords to crop to
    True_colour_image_list.append(img_rect.copy())
    Greyscale_image_list.append(gray_rect.copy())

# shows a particular image from sequence requested by 'look'
plt.clf()
plt.imshow(True_colour_image_list[look])

for r, image in enumerate(True_colour_image_list):

    # image = x
    # #IMAGE_3[IMAGE_3 <200] = 0

    # Array of means of the 3 values making up RGB for each pixel
    # Turn it to grayscale and check differences if there are any
    image_m = np.mean(image, axis=2)

    # What is the 75 percentile value on the mean from each pixel in array
    image_perc = np.percentile(image_m, 75)  # 75th percentile of the mean brightness values across the three bands

    # Pick out pixels where the value is greater than the percentile specified
    image_x = np.where(image_m >= image_perc)  # x coords of above-th pixels

    # What is the Y coordinate for each pixel with high RGB (most white like)
    image_y_coord = [t for t in image_x[0]]  # y coords of above-th pixels

    # Give counts of the Y axis locations to show frequency at that coordinate
    y_count = np.unique(image_y_coord, return_counts=True)  # counts the number of pixels in each line with high values
    y_count_list = list(y_count[1])  # list of counts
    y_height = list(y_count[0])  # list of unique values

    # Create empty lists for some ranges
    range_1 = []
    range_2 = []
    range_3 = []

    # For each Y axis height if more than width of width_var pixels across
    # (stage board is much wider but there are the black marking on it
    # (not problem for our boards in field now but is for the earlier photos)
    # OR EACH Y AXIS HEIGHT IF MORE THAN 35 PIXELS ACROSS DEEM GOOD (HAVE STAGE)
    # OTHERWISE DEEM AS BAD (NO STAGE)

    # Number of pixels required across to be as white as the threshold stated
    # above. Here is 20 but stage in these photos is about 50 across
    width_var = pixel_width / 2.5

    # Go through the y coordinates and the number of whites in their rows
    for y_count_lists, y_heights in zip(y_count_list, y_height):

        # If higher, append height to stage constituent list, and quality should be good, with the count in the third
        if y_count_lists > width_var:
            range_1.append(y_heights)
            range_2.append("GOOD")
            range_3.append("GOOD" + "_" + str(y_count_lists))
            # e+=1
        # If lower or equal, add them to stage range, but bad and count in third
        else:
            range_1.append(y_heights)
            range_2.append("BAD")
            range_3.append("BAD" + "_" + str(y_count_lists))

    # Create dataframe of above
    data = pd.DataFrame(list(zip(range_1, range_2, range_3)), columns=["HEIGHT", "QUALITY", "GROUPS"]).reset_index()
    # Group data in the dataframe by the quality of the groups, then get their counts
    data["COUNT"] = data.groupby("GROUPS")["QUALITY"]. \
        transform(lambda x: x.count())

    # Empty list for binary quality values
    quality = []

    # Give a 1 for good and a zero for bad quality
    for x in data["QUALITY"]:
        if x == "BAD":
            quality.append(0)
        else:
            quality.append(1)

    # Append binary quality list back to dataframe
    data["QUALITY_NUM"] = quality

    # See where major gaps are in heights
    data["HEIGHT_2"] = data["HEIGHT"].shift(-1) - data["HEIGHT"]

    # Create rolling means of quality centred on 10, 20 and 30 steps
    data["QUALITY_NUM_ROLL_10"] = data["QUALITY_NUM"]. \
        rolling(10, center=True).mean()
    data["QUALITY_NUM_ROLL_20"] = data["QUALITY_NUM"]. \
        rolling(20, center=True).mean()
    data["QUALITY_NUM_ROLL_30"] = data["QUALITY_NUM"]. \
        rolling(30, center=True).mean()

    a = -1
    Q = 1

    quality_num = np.array(data["QUALITY_NUM"])
    quality_g_group_30 = np.zeros(len(data["HEIGHT"]))
    quality_g_group_20 = np.zeros(len(data["HEIGHT"]))
    quality_g_group_10 = np.zeros(len(data["HEIGHT"]))

    # Give group numbers to bad quality or leave as 0 if good
    for x in np.arange(0, len(data["HEIGHT"]), 1):
        a += 1

        if quality_num[a] == 0:
            quality_g_group_10[a] = Q
            quality_g_group_20[a] = Q
            quality_g_group_30[a] = Q

        else:
            quality_g_group_10[a] = 0
            quality_g_group_20[a] = 0
            quality_g_group_30[a] = 0
            Q += 1

    data["QUALITY_G_GROUP_30"] = quality_g_group_30
    data["QUALITY_G_GROUP_20"] = quality_g_group_20
    data["QUALITY_G_GROUP_10"] = quality_g_group_10

    data["ZEROS_30"] = 0
    data["ZEROS_20"] = 0
    data["ZEROS_10"] = 0

    w = 1

    QNR_30_L = []
    QNR_30 = np.array(data["QUALITY_NUM_ROLL_30"])

    # Show where rolling quality at 30 reaches zero
    for x in QNR_30:
        if x == 0:
            QNR_30_L.append(w)
        else:
            QNR_30_L.append(0)
            w += 1
        # break

    data["ZEROS_30"] = QNR_30_L
    data_30 = data[data["ZEROS_30"] > 30]
    data["BASE_30"] = data.groupby("QUALITY_G_GROUP_30")["ZEROS_30"]. \
        transform(lambda x: x.max())
    data["BASE_H_30"] = data.groupby("BASE_30")["HEIGHT"]. \
        transform(lambda x: x.min())

    w = 1

    # Show were rolling quality at 20 reaches zero
    QNR_20_L = []
    QNR_20 = np.array(data["QUALITY_NUM_ROLL_20"])

    for x in QNR_20:
        if x == 0:
            QNR_20_L.append(w)
        else:
            QNR_20_L.append(0)
            w += 1
            # break

    data["ZEROS_20"] = QNR_20_L
    data["BASE_20"] = data.groupby("QUALITY_G_GROUP_20")["ZEROS_20"]. \
        transform(lambda x: x.max())
    data["BASE_H_20"] = data.groupby("BASE_20")["HEIGHT"]. \
        transform(lambda x: x.min())

    w = 1

    # Show were rolling quality at 10 reaches zero
    QNR_10_L = []
    QNR_10 = np.array(data["QUALITY_NUM_ROLL_10"])
    for x in QNR_10:
        if x == 0:
            QNR_10_L.append(w)
        else:
            QNR_10_L.append(0)
            w += 1
            # break

    data["ZEROS_10"] = QNR_10_L
    data["BASE_10"] = data.groupby("QUALITY_G_GROUP_10")["ZEROS_10"]. \
        transform(lambda x: x.max())
    data["BASE_H_10"] = data.groupby("BASE_10")["HEIGHT"]. \
        transform(lambda x: x.min())

    # Create a dataframe with data for photo specified in 'look' variable above
    # if want to see a problem photo
    if r == look:
        data_2 = data

    # Choose the 30 location marker first but if not the 20 then 10 etc
    try:
        range_base_10 = data[data["BASE_10"] > 20].reset_index()
        range_base_10 = range_base_10.loc[0, "BASE_H_10"]
    except:
        range_base_10 = np.nan
    try:
        range_base_20 = data[data["BASE_20"] > 20].reset_index()
        range_base_20 = range_base_20.loc[0, "BASE_H_20"]
    except:
        range_base_20 = np.nan

    try:
        range_base_30 = data[data["BASE_30"] > 20].reset_index()
        range_base_30 = range_base_30.loc[0, "BASE_H_30"]
    except:
        range_base_30 = np.nan

    if pd.isnull(range_base_10):
        range_base = np.nan

    if pd.notnull(range_base_10):
        range_base = range_base_10

    if pd.notnull(range_base_20):
        range_base = range_base_20

    if pd.notnull(range_base_30):
        range_base = range_base_30

    if pd.isnull(range_base_30):
        h = data["QUALITY_G_GROUP_30"].max()
        h = data[data["QUALITY_G_GROUP_30"] == h].reset_index()
        range_base = h["HEIGHT"][0]

    # Find the max and min of the range which should be the height of the
    # stage board exposed from water to top of board
    range_max.append(range_base)

    range_min.append(min(data["HEIGHT"]))

    if r == look:
        print(str(range_base) + "  " + str(min(data["HEIGHT"])))

    # if r == look:
    #     break

# Take away the max from the min to get the board exposed distance and apply
# Get stage by getting exposed height in cm and subtracting it from 100
board_exposed = [100 - ((x - y) / pixel_width * real_width) for x, y in zip(range_max, range_min)]
board_exposed = pd.DataFrame(list(board_exposed), columns=["EXPOSED"])

board_exposed["SHIFT_1"] = board_exposed["EXPOSED"].shift(-1) - \
                           board_exposed["EXPOSED"]

b_list = []

for x, y in zip(board_exposed["SHIFT_1"], board_exposed["EXPOSED"]):
    if x > 5 or x < -5:
        b_list.append(np.nan)
    else:
        b_list.append(y)

board_exposed["SHIFT_2"] = b_list

board_exposed["FILL"] = board_exposed["SHIFT_2"].interpolate(method='linear')
board_exposed["SMOOTH_5"] = board_exposed["FILL"].rolling(10, center=True) \
    .mean()

board_exposed["NUMBER"] = [x for x in range(0, len(board_exposed["SMOOTH_5"]))]
board_exposed["DATE"] = date_times

# Write result to csv file at the set location
outpath = 'F:\\RESULTS\\GENERATED' + os.path.basename(direct) + '_' + str(start) + '_' + str(end) + '_exposed.csv'
board_exposed.iloc[:, [6, 0]].to_csv(outpath, index=False)

# Plot rough graph of results
locator = mdates.DayLocator()
style.use("classic")
fig, (ax1) = plt.subplots(nrows=1, figsize=(10, 12), facecolor="white")
ax1.plot(board_exposed["DATE"], board_exposed["SMOOTH_5"], color="red")
ax1.plot(board_exposed["DATE"], board_exposed["EXPOSED"], color="blue")
ax1.set_ylabel("STAGE (cm)", fontsize=12)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
# ax1.xaxis.set_major_locator(locator)
ax1.set_ylim(0, 100)
plt.gcf().autofmt_xdate()
plt.show()
