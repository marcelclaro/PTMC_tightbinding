import numpy as np
from math import pi

#calculate distance between 2 atoms
def distance(a1,a2,a3,pos1,pos2,cellpos2):
	return  np.linalg.norm((cellpos2[0]*a1+cellpos2[1]*a2+cellpos2[2]*a3+pos2)-pos1)

#calculate angle (phi in xy-plane and theta - with +z-axis) between 2 atoms
def angles(a1,a2,a3,ref,pos1,pos2,cellpos2):
	x=ref[0]*a1+ref[1]*a2+ref[2]*a3
	x=x/np.linalg.norm(x)
	z=np.array([0,0,1])
	y=np.cross(z,x)
	d=(cellpos2[0]*a1+cellpos2[1]*a2+cellpos2[2]*a3+pos2)-pos1
	dxy=d-np.dot(d,z)*z
	signal=1
	if (d[0] == 0.0 and d[1] == 0.0):
		theta = 0
	else:
		if np.dot(dxy,y) < 0:
			signal = -1
		theta=signal*180*np.arccos(np.dot(dxy,x)/np.linalg.norm(dxy))/pi
	phi=90-180*np.arccos(np.dot(d,z)/np.linalg.norm(d))/pi
	return (theta,phi)