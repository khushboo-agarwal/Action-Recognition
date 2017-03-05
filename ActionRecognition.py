from __future__ import division
import math
import cv2
import numpy as np 
from PIL import Image
from matplotlib import pyplot as plt 
import sys, os
import random as rn 
import time
import scipy
import sklearn
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def ReadText(path):
	labels = []

	a = os.listdir(path)
	for i in a:
		if i != '.DS_Store':
			b = os.path.join(path, i)
			c = os.listdir(b)
			for j in c:
				if j != '.DS_Store':
					p


		

path = '/Users/administrator/PDFS/MachineLearning/Fall2016/CAP5415/PA3_new/ucf_sports'
ReadText(path)