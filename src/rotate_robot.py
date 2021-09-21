#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point, Twist



def callback(data):
	rospy.loginfo(str(data))

	# Twist message template to send to /cmd_vel topic
	move_cmd = Twist()

	if data.x <= 0.4:
		move_cmd.angular.z = 0.1
		#Turn left
	elif data.x >= 0.6:
		#turn right
		move_cmd.angular.z = -0.1
	else:
		#Stop
		move_cmd.angular.z = 0.0

	pubVel.publish(move_cmd)


def pubsub():
	global pubVel

	rospy.init_node('rotate_robot_node', anonymous=True)

	rospy.Subscriber('/robot/centroid',Point,callback)

	pubVel = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

	rospy.spin()

if __name__ == '__main__':

	pubsub()