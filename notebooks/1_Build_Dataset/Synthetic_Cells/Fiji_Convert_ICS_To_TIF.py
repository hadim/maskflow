# @File dir_path(style="directory")
# @IOService io

import os

dir_path = dir_path.toString()

for fname in os.listdir(dir_path):
	if fname.endswith(".ics"):
		full_original_path = os.path.join(dir_path, fname)
		im = io.open(full_original_path)
		io.save(im, full_original_path.replace(".ics", ".tif"))