from PIL import Image
import os
import shutil

# Do all this when the files are still on the SD card, cause after that the date of change might be modified!!!!
folder = r"F:\FIELD\FIELD_WESSENDEN"  # source folder
dst_folder = r'F:\FIELD\FIELD_WESSENDEN_1'  # destination folder

# List files in directory (you can hash the list the usual way after the closing parenthesis
files = os.listdir(folder)

# Loop through the files
for file in files:
    # Open the images and get their modification time from the exif
    im = Image.open(os.path.join(folder, file))
    exif = im.getexif()
    time = exif.get(306)
    print(time)
    im.close()
    # Assign new name based on modification time (here it is the same as creation time since we haven't moved the files)
    new_name = 'FIELD_MARS_1_' + time.replace(':', '').replace(' ', '') + '.jpg'
    # Copy file into destination folder
    shutil.copyfile(os.path.join(folder, file), os.path.join(dst_folder, new_name))
