#!/usr/bin/env python

import imutils
from imutils.object_detection import non_max_suppression
#import sys
import cv2
import numpy as np
import os

class diamond_finder:

    @staticmethod
    def find_diamond(templ, img):
        threshold = 0.90
        graderDict = {} 
        
        #Template processing

        # templ = cv2.imread("./src/team19_object_follower/src/template.jpg", cv2.IMREAD_COLOR)
        #templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
        
        (tH, tW) = templ.shape[:2]
        # print("Templ height: " + str(tH))
        # print("Templ Width: " + str(tW))
        ##cv2.imshow("Template", templ)
        
        ##print(f)
        ##img = cv2.imread(inp_dir + f, cv2.IMREAD_COLOR)
        # color_img = img.copy()
        #Convert to gray
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Array with possible matches
        rects_detailed = []
        

        #go over different scales
        for scale in np.linspace(0.5, 5, 6)[::1]:
            #image resize
            resized = imutils.resize(img, width = int(img.shape[1] * scale))
            r = img.shape[1] / float(resized.shape[1])
            #print("Resized dim: " + str(resized.shape))

            if resized.shape[0] < tH or resized.shape[1] < tW:
                continue

            edged = resized
            ## cv2.imshow("Canny", edged)
            
            #Match the template
            result = cv2.matchTemplate(edged, templ, cv2.TM_CCOEFF_NORMED)
            ##cv2.imshow("Result",result)
            ##(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            ##print("MaxVal: " + str(maxVal))
            
            # Indenify threshold exceeding regions
            (yCoords, xCoords) = np.where(result >= threshold)
            ##clone = img.copy()
            ##print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))

            if len(yCoords) != 0:
                for(x,y) in zip(xCoords, yCoords):
                    #cv2.rectangle(clone, (int(x*r), int(y*r)),(int((x + tW)*r),int((y + tH)*r)), (255, 0, 0),3)
                    rects_detailed.append((int(x*r), int(y*r),int((x + tW)*r),int((y + tH)*r))) #,result[y,x]))


            ##cv2.imshow("Before NMS", clone)
            ##cv2.waitKey(0)
        centroid = 0
        ## print("MaxVal: " + str(maxVal))
        if len(rects_detailed) != 0:
            #Non-max supp
            picks = non_max_suppression(np.array(rects_detailed))
            #print("[INFO] {} matched locations *after* NMS".format(len(picks)))
            graderList = np.delete(picks, [2,3], 1).tolist()
            #print(graderList)
            #graderDict[f] = graderList
            
            # for (startX, startY, endX, endY) in picks:
            #     cv2.rectangle(img, (startX, startY), (endX, endY), (0,215,255), 1)
            #     print("[" + f + " " + str(startX) + " " + str(startY) + "]")
            
            print(picks)

            mean_picks = np.mean(picks,axis=0)
            print(mean_picks)
            centroid = (int(round((mean_picks[0] + mean_picks[2])/2)),int(round((mean_picks[1] +mean_picks[3])/2)))
            print(centroid)
            #cv2.putText(img, "Diamonds Found: " + str(len(picks)), (10, 300),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1, cv2.LINE_AA)
            ##cv2.imshow("After NMS", img)
            return centroid
        else:
            #graderDict[f] = []
            #cv2.putText(img, "Diamonds Found: 0", (10, 300),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1, cv2.LINE_AA)
            return 0
        #print(self.graderDict)

        # Writes processed image to local directory
        #cv2.imwrite(out_dir + f,color_img)
        #cv2.imshow("Image", img)
        
        #cv2.waitKey(0)
        # return img
        
        
        ##cv2.destroyAllWindows()
    
#if __name__ == "__main__":
    #myDict = diamond_finder.find_diamond()

    #print(myDict)