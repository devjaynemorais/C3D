# Copy of https://github.com/dolongbien/HumanBehaviorBKU/blob/master/C3D/generate_inout_file.py

#cd /content/gdrive/My\ Drive/C3D/C3D-v1.0/examples/c3d_feature_extraction/input/Traning_Normal_Part1_Video
import os, os.path

# import the necessary packages
#from ..convenience import is_cv3

import cv2
import glob
import math

f_input = open('prototxt/input_list_video.txt', 'w')
f_output = open('prototxt/output_list_video_prefix.txt','w')
sh_out = open('c3d_sport1m_feature_extraction_video.sh', 'w')

interval_v = 16

cmd1 = 'time GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_video.prototxt conv3d_deepnetA_sport1m_iter_1900000'

cmd2 = 'prototxt/output_list_video_prefix.txt fc6-1'

gpu_id = 0
batch_size = 50
batch_num = 1
max_batch_num = 1
result = 0

# read file path name in alphabet order
file_list_v = sorted(glob.glob1("/content/C3D/C3D-v1.0/examples/c3d_feature_extraction/input", "*.avi"))

print('files.....')
print(file_list_v)

# each video, generate input/output file
for index in range(len(file_list_v)):
	# file path of video
	loc = file_list_v[index]
	cap=cv2.VideoCapture('input/'+loc)

	# generate .sh file
	sh_out.write('rm -rf output/c3d/' + loc[:-4])
	sh_out.write('\n')
	sh_out.write('mkdir -p output/c3d/' + loc[:-4])
	sh_out.write('\n')

	# number of frames
	total_frames = int(cap.get(7))
	print(index+1, total_frames, sep='\tframes=')

	# calculate batch number
	curr_batch_num = math.ceil(total_frames/batch_size)
	max_batch_num = batch_num
	if (curr_batch_num >= max_batch_num):
		batch_num = curr_batch_num
		print('IF - ', batch_num)
	else:
		batch_num = max_batch_num
		print('ELSE - ', batch_num)

	result = result + curr_batch_num

	print('result ', result)

	counter = 0

	# generate file
	while (counter <= total_frames-interval_v):
		f_input.write('input/' + loc + ' ' + str(counter) + ' ' + str(0))
		f_input.write('\n')
		f_output.write('output/c3d/' + loc[:-4] + '/' + str(counter).zfill(6) )
		f_output.write('\n')
		counter = counter + interval_v

# write command to bash file
parameter = ' ' + str(gpu_id) + ' ' + str(batch_size) + ' ' + str(result) + ' '
sh_out.write(cmd1 + parameter + cmd2)

print('-------- PARAMETER --------')
print('GPU ID is', gpu_id)
print('Batch size is', batch_size)
print('Number of batch is', result)

f_input.close()
f_output.close()
sh_out.close()
