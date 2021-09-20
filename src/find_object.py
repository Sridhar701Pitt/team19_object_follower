#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
#from sensor_msgs.msg import Image
import cv2
#from cv_bridge import CvBridge, CvBridgeError
#import the diamond finder script
from diamond_finder import diamond_finder
import numpy as np
import sys
from pympler.asizeof import asizeof

import matplotlib.pyplot as plt

#bridge = CvBridge()
templ = cv2.imread("./src/team19_object_follower/src/template.jpg", cv2.IMREAD_COLOR)
img = None


def callback(data):
	global img
	print("Delay:%6.3f" % (rospy.Time.now() - data.header.stamp).to_sec())
	#rospy.loginfo(rospy.get_caller_id() + " find_object input data: " + data.encoding)
	
	# try:
	# 	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	# except CvBridgeError as e:
	# 	rospy.logerror(e)

	np_arr = np.fromstring(data.data, np.uint8)
	cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


	cv_hsv=cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)


	# lower mask (0-10)
	lower_red = np.array([0,40,40])
	upper_red = np.array([14,255,255])
	mask0 = cv2.inRange(cv_hsv, lower_red, upper_red)

	# upper mask (170-180)
	lower_red = np.array([150,40,40])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(cv_hsv, lower_red, upper_red)

	# join my masks
	mask = cv2.dilate(mask0+mask1, np.ones((5,5), np.uint8))


	_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# print("Contours" + str(contours))
	centroid = 0
	if contours != []:
		boxes = []	
		for c in contours:
		    (x, y, w, h) = cv2.boundingRect(c)
		    boxes.append([x,y, x+w,y+h])

		boxes = np.asarray(boxes)
		left, top = np.min(boxes, axis=0)[:2]
		right, bottom = np.max(boxes, axis=0)[2:]

		cropped_cv = cv_image[top:bottom, left:right]
		cv2.imshow("cropped", cropped_cv)

		(tH, tW) = templ.shape[:2]
		(iH, iW) = cropped_cv.shape[:2]
		# print("th = " + str(tH) + "  tW = " + str(tW) + "  iH = " + str(iH) + "  iW = " + str(iW))
		if iH >= tH and iW >= tW:
			centroid = diamond_finder.find_diamond(templ, cropped_cv)
			processed_image = cv_image
			if centroid != 0:
				print(centroid)
				centroid = (centroid[0]+left,centroid[1]+top)
				cv2.putText(processed_image, "C", centroid,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1, cv2.LINE_AA)
			
		else:
			centroid = 0
			processed_image = cv_image


		cv2.rectangle(cv_image, (left,top), (right,bottom), (255, 0, 0), 2)
	else:
		centroid = 0
		processed_image = cv_image

	
	

	#Start detecting diamonds
	# processed_image = diamond_finder.find_diamond(templ, cv_image)
	# processed_image = cv_image

	#Display the frames
	cv2.imshow("mask", mask)
	cv2.imshow("Processed Stream", processed_image)

	# im = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

	# if img is None:
	# 	plt.figure(1)
	# 	img = plt.imshow(im)
	# 	plt.pause(1)
	# else:
	# 	img.set_data(im)
 #    # plt.pause(.1)
	# plt.draw()
	# plt.pause(.1)

	#Press Q on keyboard tp exit
	if cv2.waitKey(25) & 0Xff == ord('q'):
		cv2.destroyAllWindows()
		manual_shutdown()


def listener():
	rospy.init_node('find_object_node', anonymous=True)

	rospy.Subscriber("/raspicam_node/image/compressed",CompressedImage,callback, queue_size = 1, buff_size=2**24)

	rospy.spin()

def manual_shutdown():
	rospy.signal_shutdown("Process Finished")


if __name__ == '__main__':

	listener()