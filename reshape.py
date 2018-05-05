from PIL import Image
import PIL
import os
dirname = "Warwick QU Dataset (Released 2016_07_08)/"
reshapedirname = "reshaped_warwick/"
final_height = 400
final_width = 600
try:
	os.stat(reshapedirname)
except:
	os.mkdir(reshapedirname)
for subdirname in ["segments_train","segments_test","segments_train_eval","images_train","images_test","images_train_eval"]:
	try:
		os.stat(reshapedirname+subdirname)
	except:
		os.mkdir(reshapedirname+subdirname)
prefix_name = ["images","segments"]
suffix_name = ["train","test"]
for file in os.listdir(dirname):
	if file.endswith(".bmp"):
		img = Image.open(dirname+file)
		width, height = img.size
		istest,isanno = False,False
		if file[:4] == "test":
			istest = True
		if file.find("anno") != -1:
			isanno = True
		if isanno:
			# change the image to saturated
			img = img.point(lambda i: i * 255)
		reshaped_image = img = img.resize((final_width, final_height), PIL.Image.ANTIALIAS)
		newfilename = reshapedirname+prefix_name[isanno]+"_"+suffix_name[istest]+"/"+file.split('.')[0]
		newfilenamefull = newfilename+".png"
		reshaped_image.save(newfilenamefull,"PNG")