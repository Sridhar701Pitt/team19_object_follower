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
#from pympler.asizeof import asizeof
import os
import imutils
# import matplotlib.pyplot as plt

from geometry_msgs.msg import Point
__location__ = os.path.realpath(
	os.path.join(os.getcwd(), os.path.dirname(__file__)))
#bridge = CvBridge()
templ = cv2.imread(os.path.join(__location__, 'template.jpg'), cv2.IMREAD_COLOR)

input_img_scale = 0.5
templ = imutils.resize(templ, width = int(templ.shape[1] * input_img_scale))

img = None

#print("Hello")

def callback(data):
	global img
#	print("Delay:%6.3f" % (rospy.Time.now() - data.header.stamp).to_sec())
	#rospy.loginfo(rospy.get_caller_id() + " find_object input data: " + data.encoding)
	
	# try:
	# 	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	# except CvBridgeError as e:
	# 	rospy.logerror(e)

	np_arr = np.fromstring(data.data, np.uint8)
	cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

	#Do some pre resizing
	cv_image = imutils.resize(cv_image, width = int(cv_image.shape[1] * input_img_scale))

	h, w, _ = cv_image.shape

#	print("Width & Height: ", w, h)

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
	centroid = (w/2 , h/2)
#	print(centroid)
	if contours != []:
		boxes = []
		for c in contours:
		    (x1, y1, x2, y2) = cv2.boundingRect(c)
		    boxes.append([x1, y1, x1 + x2, y1 + y2])

		boxes = np.asarray(boxes)
		left, top = np.min(boxes, axis=0)[:2]
		right, bottom = np.max(boxes, axis=0)[2:]

		cropped_cv = cv_image[top:bottom, left:right]
		#cv2.imshow("cropped", cropped_cv)

		(tH, tW, _) = templ.shape
		(iH, iW, _) = cropped_cv.shape
		# print("th = " + str(tH) + "  tW = " + str(tW) + "  iH = " + str(iH) + "  iW = " + str(iW))
		if iH >= tH and iW >= tW:
			centroid = diamond_finder.find_diamond(templ, cropped_cv)
#			print("centroid from df.py: ", centroid)
			processed_image = cv_image
			if centroid != 0:
				centroid = (centroid[0] + left, centroid[1] + top)
#				print("shifted centroid from df.py: ", centroid)
				#cv2.putText(processed_image, "C", centroid,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1, cv2.LINE_AA)
			else:
				centroid = (w/2, h/2)
		else:
			centroid = (w/2, h/2)
			processed_image = cv_image


		cv2.rectangle(cv_image, (left,top), (right,bottom), (255, 0, 0), 2)
	else:
		centroid = (w/2, h/2)
		processed_image = cv_image

	#NOW if centroid[0] is in the left half turn left, right turn right, else, stop

#	print("centroid after for loop: ", centroid)

#	print("scaling factors: ", w, h)

	centroid = (centroid[0] / float(w), centroid[1] / float(h))

	print("scaled centroid: ", centroid)

	centroid_coords = Point()
	centroid_coords.x = centroid[0]
	centroid_coords.y = centroid[1]

	rospy.loginfo(centroid_coords)
	pub.publish(centroid_coords)

	#Start detecting diamonds
	# processed_image = diamond_finder.find_diamond(templ, cv_image)
	# processed_image = cv_image

	#Display the frames
	#cv2.imshow("mask", mask)
	#cv2.imshow("Processed Stream", processed_image)

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

	global pub
	pub = rospy.Publisher("/robot/centroid", Point, queue_size = 1)
	#rate = rospy.Rate(10) 

	rospy.spin()

def manual_shutdown():
	rospy.signal_shutdown("Process Finished")


if __name__ == '__main__':

	while not rospy.is_shutdown():	
		listener()
