import cv2
import numpy as np
import pywt
import time
import os
import pickle
from operator import itemgetter
import shutil

#Constants
#Pre processing image dimensions
PREPROCESSXDIM = 128
PREPROCESSYDIM = 128
#Wavelet type
WAVE = 'db8'
#Wavelet level
LEVEL = 4
#database path and name
INDEX_NAME = ".wavedatabase"
#Beta acceptance criteria
#Set to 50 from paper
PERCENT = 50
BETAACCEPTANCE = 1 - (PERCENT / 100)
#level 5 passing factor
L5_FACTOR = 4
#coefficients
#Set to 1 from paper
WC1 = 1
WC2 = 1
WC3 = 1
W11 = 1
W12 = 1
W21 = 1
W22 = 1


# The preTransformIndex class is a container for the wavelet transform of a signal.
# It contains the wavelet transform, the level of the transform, and the array containing record class
# 
# Args:
#   self: the instance of the class.
#   wave: the wavelet to be used
#   level: the level of the transform.

class preTransformIndex:
    def __init__(self,wave,level):
        self.wave = wave
        self.level = level
        self.arr = []



# The class record is a class that stores the Wcoeffs, stdev, WcoeffsPlus, and path of a image.
class record:
    def __init__(self,Wcoeffs,stdev,WcoeffsPlus,path):
        self.Wcoeffs = Wcoeffs
        self.stdev = stdev
        self.WcoeffsPlus = WcoeffsPlus
        self.path = path


#Is system gonna preprocess or Search?
job = input("1 - Search, 2 - Preprocess\n")
#Searching in database
if job == "1" :
    #Path of the queried image
    ImagePath = input("Path to image to be searched\n")
    #Number of found images to return
    numberOfMatchedToReturn = int(input("How many matches needs to be returned?\n"))
    #loading in database
    index = preTransformIndex(WAVE,LEVEL)
    with open(os.path.join("database", INDEX_NAME), 'rb') as f:
        index = pickle.load(f)

    dim = (PREPROCESSXDIM, PREPROCESSYDIM)
    #reading image queried image
    image = cv2.imread(ImagePath)
    start = time.process_time()
    #Resizing queried image
    image = cv2.resize(image, dim)
    results = []

    #Chaning RGB color to component color space (from paper)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            tempB = image[i][j][0]
            tempG = image[i][j][1]
            tempR = image[i][j][2]

            image[i][j][0] = (tempB + tempG + tempR)/3
            image[i][j][1] = (tempR+(255-tempB))/2
            image[i][j][2] = (tempR+2*(255-tempG)+tempB)/4

    #calculating wavelet coefficients
    Wcoeffs = pywt.wavedecn(image, WAVE, level=LEVEL, axes=(0, 1))
    stdev = np.std(Wcoeffs[0], axis=(0, 1))
    WcoeffsPlus = pywt.wavedecn(image, WAVE, level=LEVEL+1, axes=(0, 1))
    Wcoeffs = Wcoeffs[:2]
    WcoeffsPlus = WcoeffsPlus[0]

    #Going through database
    for i in index.arr:
        #Beta acceptance criteria
        if stdev[0] * BETAACCEPTANCE < i.stdev[0] < stdev[0] / BETAACCEPTANCE or (((stdev[1] * BETAACCEPTANCE < i.stdev[1] < stdev[1] / BETAACCEPTANCE) and (stdev[2] * BETAACCEPTANCE < i.stdev[2] < stdev[2] / BETAACCEPTANCE))):
            #Level 5 passing criteria
            mn = np.min(WcoeffsPlus)
            mx = np.max(WcoeffsPlus)
            temp1 = (WcoeffsPlus - np.min(WcoeffsPlus)) * (1.0 / (np.max(WcoeffsPlus) - np.min(WcoeffsPlus)))
            temp2 = (i.WcoeffsPlus - np.min(i.WcoeffsPlus)) * (1.0 / (np.max(i.WcoeffsPlus) - np.min(i.WcoeffsPlus)))
            if np.sqrt(np.sum((temp1-temp2)**2)) < L5_FACTOR:
                #final calculation using 768 feature vector from level 4 wave transform                
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
                #Adding the final calculating to array
                results.append((calc, i))
    
    #Sorting array according to calculation
    sortedResults = sorted(results, key=itemgetter(0),reverse = False)
    end = time.process_time()

    print("Time Took to Search: ")
    print("{0:.2f}s".format(end - start))

    #Should we save the results?
    tempSave = input("Do you wanna save the results y|n \n")
    c=0
    if tempSave == "y":
        for (calc, i) in sortedResults:
            if c<numberOfMatchedToReturn:
                shutil.copy2(i.path,os.path.join("SavedImages", os.path.basename(i.path)))
                c+=1
        print("Images are saved!\n")

    #Should we show the results
    tempShow = input("Do you wanna see the results y|n \n")
    c=0
    if tempShow == "y":
        arrImagesToShow = []
        cv2.imshow('Searched Image', cv2.resize(cv2.imread(ImagePath),(250,250)))
        if numberOfMatchedToReturn > len(sortedResults):
            tempnumberOfMatchedToReturn = len(sortedResults)

        for calc, i in sortedResults:
            if c<numberOfMatchedToReturn:
                cv2.imshow('Found Images '+str(c), cv2.resize(cv2.imread(i.path),(250,250)))
                c+=1           

        cv2.waitKey(100000)

#Preprocessing to create database
if job == "2" :
    #Datasets folder path
    preprocessImagesPath = input("Path to images that will be preprocessed\n")
    
    start = time.process_time()
    #Database object
    index = preTransformIndex(WAVE,LEVEL)
    #f = 0
    for imageFile in os.listdir(preprocessImagesPath):
        try:
            #Uncomment to have a loading bar
            #print(f)
            #print(imageFile)
            #f+=1
            #Read Image
            image = cv2.imread(os.path.join(preprocessImagesPath,imageFile))
            dim = (PREPROCESSXDIM, PREPROCESSYDIM)
            #Resize the image
            image = cv2.resize(image,dim)
            #Recolor the image
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    tempB = image[i][j][0]
                    tempG = image[i][j][1]
                    tempR = image[i][j][2]
                    image[i][j][0] = (tempB + tempG + tempR)/3
                    image[i][j][1] = (tempR+(255-tempB))/2
                    image[i][j][2] = (tempR+2*(255-tempG)+tempB)/4
            #Get wavelet coefficients of the image
            Wcoeffs = pywt.wavedecn(image, WAVE, level=LEVEL, axes=(0, 1))
            stdev = np.std(Wcoeffs[0], axis=(0, 1))
            WcoeffsPlus = pywt.wavedecn(image, WAVE, level=LEVEL+1, axes=(0, 1))
            tempRecord = record(Wcoeffs[:2].copy(), stdev.copy(), WcoeffsPlus[0].copy(),os.path.realpath(os.path.join(preprocessImagesPath,imageFile)))
            #Add image feautures to database
            index.arr.append(tempRecord)
            #Uncomment if dont wanna go through whole dataset
            #if f == 500:
            #    break;

        except Exception as e:
            print(str(e))

    #Saving database to the disk
    with open(os.path.join("database", INDEX_NAME), 'wb') as f:
        pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)
    end = time.process_time()
    print("Time Took to PreProcess: ")
    print("{0:.2f}s".format(end - start))



