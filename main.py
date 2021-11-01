import cv2
import numpy as np
import pywt
import time
import os
import pickle
from operator import itemgetter
import shutil

PREPROCESSXDIM = 128
PREPROCESSYDIM = 128
WAVE = 'db8'
LEVEL = 4
INDEX_NAME = ".wavedatabase"
PERCENT = 50
BETAACCEPTANCE = 1 - (PERCENT / 100)
L5_FACTOR = 5
WC1 = 1
WC2 = 1
WC3 = 1
W11 = 1
W12 = 1
W21 = 1
W22 = 1


class preTransformIndex:
    def __init__(self,wave,level):
        self.wave = wave
        self.level = level
        self.arr = []



class record:
    def __init__(self,Wcoeffs,stdev,WcoeffsPlus,path):
        self.Wcoeffs = Wcoeffs
        self.stdev = stdev
        self.WcoeffsPlus = WcoeffsPlus
        self.path = path



job = input("1 - Search, 2 - Preprocess\n")

if job == "1" :
    ImagePath = input("Path to image to be searched\n")
    numberOfMatchedToReturn = int(input("How many matches needs to be returned?\n"))
    index = preTransformIndex(WAVE,LEVEL)
    with open(os.path.join("database", INDEX_NAME), 'rb') as f:
        index = pickle.load(f)

    dim = (PREPROCESSXDIM, PREPROCESSYDIM)

    image = cv2.imread(ImagePath)
    

    start = time.process_time()

    image = cv2.resize(image, dim)
    results = []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    Wcoeffs = pywt.wavedecn(image, WAVE, level=LEVEL, axes=(0, 1))
    stdev = np.std(Wcoeffs[0], axis=(0, 1))
    WcoeffsPlus = pywt.wavedecn(image, WAVE, level=LEVEL+1, axes=(0, 1))
    Wcoeffs = Wcoeffs[:2]
    WcoeffsPlus = WcoeffsPlus[0]

    for i in index.arr:
        if stdev[0] * BETAACCEPTANCE < i.stdev[0] < stdev[0] / BETAACCEPTANCE or (((stdev[1] * BETAACCEPTANCE < i.stdev[1] < stdev[1] / BETAACCEPTANCE) and (stdev[2] * BETAACCEPTANCE < i.stdev[2] < stdev[2] / BETAACCEPTANCE))):
            mn = np.min(WcoeffsPlus)
            mx = np.max(WcoeffsPlus)
            temp1 = (WcoeffsPlus - np.min(WcoeffsPlus)) * (1.0 / (np.max(WcoeffsPlus) - np.min(WcoeffsPlus)))
            temp2 = (i.WcoeffsPlus - np.min(i.WcoeffsPlus)) * (1.0 / (np.max(i.WcoeffsPlus) - np.min(i.WcoeffsPlus)))
            if np.sqrt(np.sum((temp1-temp2)**2)) < L5_FACTOR:
                
                wcoef = np.array([WC1, WC2, WC3])

                iWC11 = i.Wcoeffs[0]
                WC11 = Wcoeffs[0]

                iWC12 = i.Wcoeffs[1]['da']
                WC12 = Wcoeffs[1]['da']

                iWC21 = i.Wcoeffs[1]['ad']
                WC21 = Wcoeffs[1]['ad']

                iWC22 = i.Wcoeffs[1]['dd']
                WC22 = Wcoeffs[1]['dd']

                calc = W11 * np.sum(wcoef * np.sqrt(np.sum((WC11-iWC11)**2))) + W12 * np.sum(wcoef * np.sqrt(np.sum((WC12-iWC12)**2))) + W21 * np.sum(wcoef * np.sqrt(np.sum((WC21-iWC21)**2))) + W22 * np.sum(wcoef * np.sqrt(np.sum((WC22-iWC22)**2)))
                results.append((calc, i))
    

    sortedResults = sorted(results, key=itemgetter(0),reverse = False)
    end = time.process_time()

    print("Time Took to Search: ")
    print("{0:.2f}s".format(end - start))

    tempSave = input("Do you wanna save the results y|n \n")
    c=0
    if tempSave == "y":
        for (calc, i) in sortedResults:
            if c<numberOfMatchedToReturn:
                shutil.copy2(i.path,os.path.join("SavedImages", os.path.basename(i.path)))
                c+=1
        print("Images are saved!\n")

    tempShow = input("Do you wanna see the results y|n \n")
    c=0
    if tempShow == "y":
        arrImagesToShow = []
        cv2.imshow('Searched Image', cv2.resize(cv2.imread(ImagePath),(500,500)))
        if numberOfMatchedToReturn > len(sortedResults):
            tempnumberOfMatchedToReturn = len(sortedResults)

        for calc, i in sortedResults:
            if c<numberOfMatchedToReturn:
                cv2.imshow('Found Images '+str(c+1), cv2.resize(cv2.imread(i.path),(500,500)))
                c+=1           

        cv2.waitKey(100000)

        




    

if job == "2" : 
    preprocessImagesPath = input("Path to images that will be preprocessed\n")
    start = time.process_time()

    index = preTransformIndex(WAVE,LEVEL)

    for imageFile in os.listdir(preprocessImagesPath):
        image = cv2.imread(os.path.join(preprocessImagesPath,imageFile))
        dim = (PREPROCESSXDIM, PREPROCESSYDIM)
        image = cv2.resize(image,dim)
        cv2.imwrite(os.path.join("database",imageFile), image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        Wcoeffs = pywt.wavedecn(image, WAVE, level=LEVEL, axes=(0, 1))
        stdev = np.std(Wcoeffs[0], axis=(0, 1))
        WcoeffsPlus = pywt.wavedecn(image, WAVE, level=LEVEL+1, axes=(0, 1))
        tempRecord = record(Wcoeffs[:2].copy(), stdev.copy(), WcoeffsPlus[0].copy(),os.path.realpath(os.path.join(preprocessImagesPath,imageFile)))
        index.arr.append(tempRecord)

    with open(os.path.join("database", INDEX_NAME), 'wb') as f:
        pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)
    end = time.process_time()
    print("Time Took to PreProcess: ")
    print("{0:.2f}s".format(end - start))



