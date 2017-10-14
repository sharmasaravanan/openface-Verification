#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import re
from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface
import csv 
import pprint

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GMM

fileDir = os.path.dirname(os.path.realpath(__file__))

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')



def getRep(imgPath):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
   
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()
    faces = align.getAllFaceBoundingBoxes(rgbImg)
 
    reparray = []
    facerectangle = []
    print(" size of faces :",len(faces))
    for i in range(0,len(faces)):
        if faces[i] is None:
           raise Exception("Unable to find a face: {}".format(imgPath))
        if args.verbose:
           print("Face detection took {} seconds.".format(time.time() - start))

        start = time.time()
        alignedFace = align.align(args.imgDim, rgbImg, faces[i],
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
           raise Exception("Unable to align image: {}".format(imgPath))
        if args.verbose:
           print("Alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        #s+=str(i)
        #s+=".jpg"
        #cv2.imwrite(s,alignedFace)
        reparray += [net.forward(alignedFace)]
        facerectangle +=[faces[i]]
        
        if args.verbose:
            print("Neural network forward pass took {} seconds.".format(time.time() - start))
    return reparray, facerectangle


def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if args.classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    elif args.classifier == 'GMM':
        clf = GMM(n_components=nClasses)

    if args.ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])

    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


def infer(args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)
    path_1 = []
    path_predict = []
    confidencescore = []
    originalpath = []
    img1 = ''.join(args.imgs)
    print(img1,'hiboss')
    dirt = os.getcwd()+'/'+img1  
    save=0
    #print("harishankar",[x[0] for x in os.walk(dirt)])
    for root, dirs, files in os.walk(dirt):
        print(files)
        for fi in files:
            img = os.path.join(root,fi)
            save+=1;
            start = time.time()
            print("\n=== {} ===".format(img))
            originalpath += [img]
            imgpath,imgbase = os.path.split(img)
            print(imgpath,"harshankar")
            a = img1+'/(\w+)'
            match = re.search(a,imgpath,re.IGNORECASE)
            match = match.group(0)
            print("match",str(match))
            try:
                [reparray, facerectangle] = getRep(img)
                source = cv2.imread(img)
            except:
                print("no face detected")
                    
            else:
                print(len(reparray))
                for i in range(0,len(reparray)):
                    rep = reparray[i].reshape(1, -1)
                    path_1 +=[match]
                    s="result"
                    #start = time.time()
                    predictions = clf.predict_proba(rep).ravel()
                    maxI = np.argmax(predictions)
                    person = le.inverse_transform(maxI)
                    confidence = predictions[maxI]
                    #if args.verbose:
                    a = str(facerectangle[i])                    
                    values = re.findall("\d+",a,re.IGNORECASE)
                    print(values)
                    x =int(values[0])
                    y = int(values[1])
                    w = int(values[2])
                    h =  int(values[3])

                    
                    if confidence > 0.39:
                       cv2.rectangle(source, (x, y), (w, h), (0,255, 0), 2)                    
                       cv2.putText(source,person, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)
		       #cv2.imwrite("./result-image/"+person+".jpg",source)
                    else:
                       cv2.rectangle(source, (x, y), (w, h), (0,0, 255), 2)                    
                       cv2.putText(source,"UnKnown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)
                       #cv2.imwrite("./result-image/results"+person+".jpg",source)
                    print("Prediction took {} seconds.".format(time.time() - start))
                    print("Predict {} with {:.2f} confidence.".format(person, confidence))
                    confidencescore += [confidence]
                    path_predict +=[person]
                    if isinstance(clf, GMM):
                        dist = np.linalg.norm(rep - clf.means_[maxI])
                        print("  + Distance from the mean: {}".format(dist))
                cv2.imshow("faces found", source)
                cv2.imwrite("./result-image/"+s+str(save)+".jpg",source)
                cv2.waitKey(5000)
    return path_1, path_predict, confidencescore, originalpath

def validation(path_1,path_predict,confidencescore, originalpath):
    c = csv.writer(open("Result.txt", "w"))
    print(path_1)
    print(path_predict)
    print(originalpath)
    string1 = []
    for word in path_1:
        match = re.findall("/(\w+)",word,re.IGNORECASE)
        match = ''.join(match)
        string1 += [match]
    print("string",string1)
    j = 0
    for i in range(len(string1)):
        if string1[i] == path_predict[i]:
           print(string1[i],path_predict[i])
           j = j+1
           print(j)
	   print("================") 
           print(path_1)
	   print("================")
 
	   c.writerow([string1[i],path_predict[i],confidencescore[i],"1"])
           #c.writerow(["originalpath"+originalpath[i], "testedpath"+pathnew])
           
        else:
           print("non-matches")
          
           j = j
           c.writerow([string1[i],path_predict[i],confidencescore[i],"0"]) 

    print "number of persons correctly predict",j
    print "total number of persons",len(path_1)
    percentage = j*100/len(path_1)
    print "percentage of prediction", percentage,"%"
   
   
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFacePredictor', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir,
                                             "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str,
                        help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument('--classifier', type=str,
                             choices=['LinearSvm', 'GMM'],
                             help='The type of classifier to use.',
                             default='LinearSvm')
    trainParser.add_argument('workDir', type=str,
                             help="The input work directory containing 'reps.csv' and 'labels.csv'. Obtained from aligning a directory with 'align-dlib' and getting the representations with 'batch-represent'.")

    inferParser = subparsers.add_parser('infer',
                                        help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel', type=str,
                             help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(time.time() - start))

    if args.mode == 'infer' and args.classifierModel.endswith(".t7"):
        raise Exception("""
Torch network model passed as the classification model,
which should be a Python pickle (.pkl)

See the documentation for the distinction between the Torch
network and classification models:

        http://cmusatyalab.github.io/openface/demo-3-classifier/
        http://cmusatyalab.github.io/openface/training-new-models/

Use `--networkModel` to set a non-standard Torch network model.""")
    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))
        start = time.time()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        #infer(args)
    	[path_1,path_predict,confidencescore,originalpath] = infer(args)
        
        validation(path_1,path_predict,confidencescore,originalpath)
              




    	
