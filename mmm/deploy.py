import shutil
import os
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
all_items = os.listdir(current_directory)
files_in_directory = [item for item in all_items if os.path.isfile(os.path.join(current_directory, item))]

deploy_files = [f for f in files_in_directory if f.startswith("bc") and f.endswith(".py")]

for f in deploy_files:
    scr_dat = f"./mmm/{f}"
    dst_dat = f"C:\\Users\\christianb\\python\\WPy64-31110\\python-3.11.1.amd64\\Lib\\site-packages\\{f}"
    shutil.copy(scr_dat, dst_dat)

time.sleep(3)
