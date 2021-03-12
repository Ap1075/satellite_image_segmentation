import cv2
import json
import numpy as np
import os

def draw_over_canvas(canvas, structure):
	"""Draws structures with corresponding pixel values over empty canvas to serve as mask for training.
	"""
	poly_type = structure["name"]
	for polygon in structure["annotations"]:
		polygon['segmentation'] = [round(i) for i in polygon['segmentation']]
		polygon['segmentation'] = list(zip(*[iter(polygon['segmentation'])] * 2))

		if poly_type == "Houses" :
			cv2.fillConvexPoly(canvas, np.array(polygon['segmentation']), 1)
		elif poly_type == "Buildings" :
			cv2.fillConvexPoly(canvas, np.array(polygon['segmentation']), 2)
		else:
			cv2.fillConvexPoly(canvas, np.array(polygon['segmentation']), 3)

	return canvas


def mask_from_json(path_to_annotations,path_to_masks):
	"""Creates mask from json files. Parses json annotation files and passes to draw_over_canvas to draw structures.
	"""

	json_files = [pos_json for pos_json in os.listdir(path_to_annotations) if pos_json.endswith('.json')]
	for n,file in enumerate(json_files):
		full_filename = "%s/%s" % (path_to_annotations, file)
		with open(full_filename,'r') as fi:
			data = json.load(fi)
		img_width = data['width']
		img_height = data['height']
		houses, buildings, sheds = data["labels"]
		canvas = np.zeros((img_height,img_width,1), np.int32)
		for structure in [houses, buildings, sheds]:
			canvas = draw_over_canvas(canvas, structure)
		output_name = "%s/%s%s" % (path_to_masks, file.split('.')[0],'.png')

		cv2.imwrite(output_name, canvas)

if __name__=='__main__':
	mask_from_json("./annotations","./annotation_masks")