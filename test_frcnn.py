from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model,load_model
from keras_frcnn import roi_helpers
from keras.applications.mobilenet import preprocess_input
import math,random

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--write", dest="write", help="to write out the image with detections or not.", action='store_true')
parser.add_option("--load", dest="load", help="specify model path.", default=None)
(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# we will use resnet. may change to vgg
if options.network == 'vgg':
	C.network = 'vgg16'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
elif options.network == 'vgg19':
	from keras_frcnn import vgg19 as nn
	C.network = 'vgg19'
elif options.network == 'mobilenetv1':
	from keras_frcnn import mobilenetv1 as nn
	C.network = 'mobilenetv1'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_05':
	from keras_frcnn import mobilenetv1_05 as nn
	C.network = 'mobilenetv1_05'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_25':
	from keras_frcnn import mobilenetv1_25 as nn
	C.network = 'mobilenetv1_25'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	C.network = 'mobilenetv2'
else:
	print('Not a valid model')
	raise ValueError

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path


def get_proposed_regions(region):
    minX = region[0]
    minY = region[1]
    maxX = region[2]
    maxY = region[3]

    proposed_list = [region]
    scale = 2
    proposed_list.append([minX,minY,int(maxX/scale),int(maxY/scale)])
    proposed_list.append([minX,minY,int(maxX*scale),int(maxY*scale)])
    proposed_list.append([int(minX/scale),int(minY/scale),maxX,maxY])
    proposed_list.append([int(minX*scale),int(minY*scale),maxX,maxY])
    proposed_list.append([int(minX*scale),minY,int(maxX*scale),maxY])
    proposed_list.append([int(minX/scale),minY,int(maxX/scale),maxY])
    proposed_list.append([minX,int(minY*scale),maxX,int(maxY*scale)])
    proposed_list.append([minX,int(minY/scale),maxX,int(maxY/scale)])
    proposed_list.append([minX+(scale*10),minY+(scale*10),maxX-(scale*10),maxY-(scale*10)])

    for i,prop in enumerate(proposed_list): #fix going out of bounds
        if prop[2] > maxX:
            proposed_list[i][2] = maxX
        if prop[3] > maxY:
            proposed_list[i][3] = maxY

    return proposed_list

def get_roi_coords(model1,model2,image):
	X_size = 800 #part1
	Y_size = 64 #part1

	pTwo_size = 600 #part2
	cuts_labels = 60 #part2
	label_precision = 8

	y_fail_num = 2

	image_label = []
	input_roi=[]
	num_roi = 10 #fixed number of rois

	pixel_data = cv2.imread(image, 0)
	original_pixel_data_255 = pixel_data.copy()
	pixel_data = cv2.normalize(pixel_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	original_pixel_data = pixel_data.copy()

	height, width = pixel_data.shape
	scale = X_size/width

	pixel_data = cv2.resize(pixel_data, (X_size, int(height*scale))) #X, then Y
	bordered_pixel_data = cv2.copyMakeBorder(pixel_data,top=int(Y_size/4),bottom=int(Y_size/4),left=0,right=0,borderType=cv2.BORDER_CONSTANT,value=1)

	slice_skip_size = int(Y_size/2)
	iter = 0
	slices = []
	while((iter*slice_skip_size + Y_size) < int(height*scale+Y_size/2)):
		s_iter = iter*slice_skip_size
		slices.append(bordered_pixel_data[int(s_iter):int(s_iter+Y_size)])
		iter += 1

	slices = np.array(np.expand_dims(slices,  axis = -1))

	data = model1.predict(slices)

	conc_data = []
	for single_array in data:
		for single_data in single_array:
			conc_data.append(single_data)
	conc_data += [0 for i in range(y_fail_num+1)] #Still needed
	groups = []
	fail = y_fail_num
	group_start = 1 #start at 1 to prevent numbers below zero in groups
	for iter in range(len(conc_data)-1):
		if(conc_data[iter] < .5):
			fail += 1
		else:
			fail = 0

		if(fail >= y_fail_num):
			if(iter - group_start >= 4):
				groups.append((int((group_start-1)*label_precision/scale), int((iter+1-y_fail_num)*label_precision/scale)))
			group_start = iter



	groups2 = []
	for group in groups:
		temp_final_original = cv2.resize(original_pixel_data[group[0]:group[1]], (pTwo_size, pTwo_size))
		temp_final = np.expand_dims(np.expand_dims(temp_final_original,  axis = 0), axis = -1)
		data_final = model2.predict(temp_final)

		hor_start = -1
		hor_finish = 10000
		pointless, original_width = original_pixel_data.shape

		for iter in range(len(data_final[0])):
			if(data_final[0][iter] > .5 and hor_start == -1):
				if(iter > 0):
					hor_start = int((iter-0.5)*original_width/cuts_labels)
				else:
					hor_start = int(iter*original_width/cuts_labels)

			if(data_final[0][iter] > .5):
				hor_finish = int((iter+0.5)*original_width/cuts_labels)

		if(0 and hor_finish - hor_start > (0.7 * original_width)): #Fix for tables that cover the entire image
			groups2.append((0, original_width))
		else:
			groups2.append((hor_start, hor_finish))

	data_shared = 0
	all_roi_coords=[]
	start_ind = len(image_label)
    #add generated coordinates
	for iter in range(len(groups)):
		final_split = original_pixel_data_255[groups[iter][0]:groups[iter][1], groups2[iter][0]:groups2[iter][1]]
		all_roi_coords.append([groups2[iter][0],groups[iter][0],groups2[iter][1],groups[iter][1]])
		if(0):
			cv2.imshow('image', final_split)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imshow('image', original_pixel_data_255[xml_locs[0][2]:xml_locs[0][3], xml_locs[0][0]:xml_locs[0][1]])
			cv2.waitKey(0)
			cv2.destroyAllWindows()
    #for each location in all roi coords, propose 10 scaled regions around location
	'''
	proposed_regions=[]
	for region in all_roi_coords:
		copy_image = original_pixel_data
		proposed_regions.append(get_proposed_regions(region))
	'''
	return all_roi_coords

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape

	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network =="mobilenetv2":
	num_features = 320
else:
	# may need to fix this up with your backbone..!
	print("backbone is not resnet50. number of features chosen is 512")
	num_features = 512

if K.image_data_format() == 'channels_first':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([feature_map_input, roi_input], classifier)
model1 = load_model("/Users/serafinakamp/Desktop/TableExt/opt_branch/datasheet-scrubber/src/cnn_models/stage1.h5")
model2 = load_model("/Users/serafinakamp/Desktop/TableExt/opt_branch/datasheet-scrubber/src/cnn_models/stage2.h5")

# model loading
if options.load == None:
  print('Loading weights from {}'.format(C.model_path))
  model_rpn.load_weights(C.model_path, by_name=True)
  model_classifier.load_weights(C.model_path, by_name=True)
else:
  print('Loading weights from {}'.format(options.load))
  model_rpn.load_weights(options.load, by_name=True)
  model_classifier.load_weights(options.load, by_name=True)

#model_rpn.compile(optimizer='adam', loss='mse')
#model_classifier.compile(optimizer='adam', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.5

visualise = True

num_rois = C.num_rois

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)

    # preprocess image
	X, ratio = format_img(img, C)
	img_scaled = (np.transpose(X[0,:,:,:],(1,2,0)) + 127.5).astype('uint8')
	if K.image_data_format() == 'channels_last':
		X = np.transpose(X, (0, 2, 3, 1))
	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	print(np.shape(F))


	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.3)
	print(R.shape)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}
	for jk in range(R.shape[0]//num_rois + 1):
		ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1),:],axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:,:curr_shape[1],:] = ROIs
			ROIs_padded[0,curr_shape[1]:,:] = ROIs[0,0,:]
			ROIs = ROIs_padded
		maxY = np.shape(img_scaled)[0]
		maxX = np.shape(img_scaled)[1]
		F_y = np.shape(F)[1]
		F_x = np.shape(F)[2]
		Y_ratio = maxY/F_y
		X_ratio = maxX/F_x
		print("Y ratio", maxY/F_y)
		print("X ratio", maxX/F_x)
		print("ratio", ratio)
		rois = get_roi_coords(model1,model2,os.path.join(img_path,img_name))
		all_areas=[]
		for region in rois:
			x1,y1,x2,y2 = get_real_coordinates(1/ratio,region[0],region[1],region[2],region[3])
			#if x2>maxX:
				#x2=maxX
			#if y2>maxY:
				#y2=maxY
			x1 /= X_ratio
			y1 /= Y_ratio
			x2 -=x1
			x2 /= X_ratio #width
			y2 -= y1
			y2 /= Y_ratio #height
			all_areas.append([x1,y1,x2,y2])
		print(all_areas)
		print(np.shape(img_scaled))
		im = np.ascontiguousarray(img_scaled)
		for roi in ROIs[0]:
			x1 = roi[0]*X_ratio
			y1 = roi[1]*Y_ratio
			x2 = x1+roi[2]*X_ratio
			y2 = y1+roi[3]*Y_ratio
			rand_255_r = math.floor(random.random()*255)
			rand_255_g = math.floor(random.random()*255)
			rand_255_b = math.floor(random.random()*255)

			cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(rand_255_r, rand_255_g, rand_255_b), 2)

		cv2.imshow("image",im)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	


		arr_areas = np.array(np.expand_dims(all_areas,axis=0))
		print(arr_areas)
		print(ROIs)
		[P_cls,P_regr] = model_classifier.predict([F, arr_areas])
		print(P_cls)

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0,ii,:]) < 0.8 or np.argmax(P_cls[0,ii,:]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0,ii,:])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []
			(x,y,w,h) = ROIs[0,ii,:]

			bboxes[cls_name].append([16*x,16*y,16*(x+w),16*(y+h)])
			probs[cls_name].append(np.max(P_cls[0,ii,:]))

	all_dets = []

	for key in bboxes:
		#print(key)
		#print(len(bboxes[key]))
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.3)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	print('Elapsed time = {}'.format(time.time() - st))
	print(all_dets)
	print(bboxes)
    # enable if you want to show pics
	if options.write:
           import os
           if not os.path.isdir("results"):
              os.mkdir("results")
           cv2.imwrite('./results/{}.png'.format(idx),img)
