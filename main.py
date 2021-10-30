import cv2
import numpy as np
import pywt
import time
import os
import pickle

PREPROCESSXDIM = 128
PREPROCESSYDIM = 128
WAVE = 'db8'
LEVEL = 4
INDEX_NAME = "preProcessIndexFile"

class preTransformIndex:
    def __init__(self,wave,level):
        self.wave = wave
        self.level = level
        self.arr = []

class record:
    def __init__(self,Wcoeffs,stdev,WcoeffsPlus):
        self.Wcoeffs = Wcoeffs
        self.stdev = stdev
        self.WcoeffsPlus = WcoeffsPlus



job = input("1 - Search, 2 - Preprocess")

if job == "2" : 
    preprocessImagesPath = input("Path to images that will be preprocessed")
    start = time.process_time()

    index = preTransformIndex(WAVE,LEVEL)

    for imageFile in os.listdir(preprocessImagesPath):
        image = cv2.imread(os.path.join(preprocessImagesPath,imageFile))
        dim = (PREPROCESSXDIM, PREPROCESSYDIM)
        image = cv2.resize(image,dim)
        cv2.imwrite(os.path.join("/database",imageFile), image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        Wcoeffs = pywt.wavedecn(image, WAVE, level=LEVEL, axes=(0, 1))
        stdev = np.std(Wcoeffs[0], axis=(0, 1))
        WcoeffsPlus = pywt.wavedecn(image, WAVE, level=LEVEL+1, axes=(0, 1))
        tempRecord = record(Wcoeffs[:2].copy(), stdev.copy(), WcoeffsPlus[0].copy())
        index.arr.append(tempRecord)

    with open(os.path.join("/database", INDEX_NAME), 'wb') as f:
        pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)
    end = time.process_time()
    print("Time Took : ")
    print(end - start)



