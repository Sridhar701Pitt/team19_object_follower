''' 
-------------------------------------------------------------
Lab Partners : Sridhar Reddy Velagala and Yusuf Ali
-------------------------------------------------------------
'''

#!/usr/bin/env python

import cv2
import sys
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import os

class diamond_finder:

    @staticmethod
    def find_diamond(template_large, img):

        # Define "found" variable which tracks whether a better template match has been detected for an image 
        found = None

        # Define various scaling and template mathcing thresholds
        template_matching_threshold = 0.8
        min_template_scale = 0.1
        max_template_scale = 1
        template_scale_steps = 10
        # input_img_scale = 1
        # no_of_imgs = len(images)

        # resizing input image (can be done if required)
        resized = img
	#resized = imutils.resize(img, width = int(img.shape[1] * input_img_scale))

        # Main for loop which performs template matching
        # for i in range(0, no_of_imgs):     
        for scale in np.linspace(min_template_scale, max_template_scale, template_scale_steps)[::-1]:

            # resize the template image according to the scale, and keep track of the ratio of the resizing
            template = imutils.resize(template_large, width = int(template_large.shape[1] * scale))

            (tH, tW) = template.shape[:2]
            
            # matching to find the template in the image
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)

            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            if found is None or maxVal > found:
                found = maxVal

                # initialize our list of rectangles
                rects = (maxLoc[0], maxLoc[1], maxLoc[0] + tW, maxLoc[1] + tH)

            if maxVal > 0.95 or maxVal < found:
                break
        
        if len(rects) != 0 and maxVal > template_matching_threshold:
            mean_picks = rects
            
            # mean_picks = np.mean(rects, axis=0)
            centroid = (int(round((mean_picks[0] + mean_picks[2])/2)),int(round((mean_picks[1] +mean_picks[3])/2)))

            return centroid

        else:
            return 0
