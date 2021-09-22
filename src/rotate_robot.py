#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point, Twist



def callback(data):
	rospy.loginfo(str(data))

	# Twist message template to send to /cmd_vel topic
	move_cmd = Twist()

	k_p = 1.5

	error = (0.5 - data.x)

	if abs(error) < 0.1:
	    move_cmd.angular.z = 0
	
	else:
	    move_cmd.angular.z = k_p * error
	

	pubVel.publish(move_cmd)


def pubsub():
	global pubVel

	rospy.init_node('rotate_robot_node', anonymous=True)

	rospy.Subscriber('/robot/centroid',Point,callback)

	pubVel = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

	rospy.spin()

if __name__ == '__main__':

	pubsub()
