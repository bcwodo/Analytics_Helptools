import shutil
import os
import time

files = [f for f in os.listdir('.') if os.path.isfile(f)]
deploy_files = [f for f in files if f.startswith("bc") and f.endswith(".py")]


for f in deploy_files:
    scr_dat = f
    dst_dat = f"C:\\Users\\christianb\\python\\WPy64-31110\\python-3.11.1.amd64\\Lib\\site-packages\\{f}"
    shutil.copy(scr_dat, dst_dat)

time.sleep(3)
