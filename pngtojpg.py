from PIL import Image
from tqdm import tqdm
import os

TEMP_DIR = 'train_jpg'

for img in tqdm(os.listdir(TEMP_DIR)):
	path = os.path.join(TEMP_DIR, img)
	new_file_name = img.split('.')[0] + '.' + img.split('.')[1] + ".jpg"
	im = Image.open(path)
	bg = Image.new("RGB", im.size, (255,255,255))
	bg.paste(im,im)
	bg.save(os.path.join(TEMP_DIR, new_file_name))
	#print(os.path.join(TEMP_DIR, new_file_name))