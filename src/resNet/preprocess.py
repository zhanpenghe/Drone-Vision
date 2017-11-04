import os
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler

def loadRecord(filepath, col_selection):
	result = {}
	with open(filepath, 'r') as file:
		for line in file:
			splited_str = line.split("\t")
			row = []
			for i in range(len(splited_str)):
				if i == 8:
					continue
				elif i == 0 and i in col_selection:
					row.append(int(splited_str[i]))
				elif i in col_selection:
					row.append(float(splited_str[i]))
			result[splited_str[8].strip()] = row
	return result

def loadImages(img_dir_path, records):
	X = []
	y = []
	for filename in os.listdir(img_dir_path):
		if records[filename]!=None:
			img = ndimage.imread(img_dir_path+'/'+filename, mode='RGB')
			record = records[filename]
			X.append(img)
			y.append(record)

	return np.array(X), np.array(y)


#load images and records to X, a vector of images, and y, a vector of records
def loadData(data_path):
	try:
		if data_path[-1]!='/':
			data_path+='/'
		records = loadRecord(data_path+'airsim_rec.txt', [4, 5])
		return loadImages(data_path+'images', records)
	except Exception as e:
		print(e)

def preprocess_y(y_train, y_test):
	scaler = MinMaxScaler()
	y_train = scaler.fit_transform(y_train)
	y_test = scaler.transform(y_test)
	return y_train, y_test, scaler
