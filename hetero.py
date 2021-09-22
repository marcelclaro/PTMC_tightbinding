import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, cos, sin

pb.pltutils.use_style()


"""Hopping projections
"""
def s_px(hopping,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return hopping*cos(a)*cos(b)

def s_py(hopping,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return -hopping*sin(a)*cos(b)
	
def s_pz(hopping,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return hopping*sin(b)
	
def pxpx(h_sigma,h_pi,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return h_sigma*cos(a)**2*cos(b)**2+h_pi*(sin(a)**2+cos(a)**2*sin(b)**2)
	
def pypx(h_sigma,h_pi,angle=(0,0)):
	return pxpy(h_sigma,h_pi,angle)
	
def pxpy(h_sigma,h_pi,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return -h_sigma*cos(a)*sin(a)*cos(b)**2+h_pi*(sin(a)*cos(a)+sin(a)*sin(b)-cos(a)*cos(b))

def pxpz(h_sigma,h_pi,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return h_sigma*cos(a)*sin(a)*sin(b)-h_pi*cos(a)*cos(b)*sin(b)
	
def pzpx(h_sigma,h_pi,angle=(0,0)):
	return pxpz(h_sigma,h_pi,angle)
	
def pypy(h_sigma,h_pi,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return h_sigma*sin(a)**2*cos(b)**2+h_pi*(cos(a)**2+sin(a)**2*sin(b)**2)
	
def pypz(h_sigma,h_pi,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return -h_sigma*sin(a)*cos(b)*sin(b)+h_pi*cos(b)*sin(b)*sin(a)
	
def pzpz(h_sigma,h_pi,angle=(0,0)):
	a=2*pi*angle[0]/360
	b=2*pi*angle[1]/360
	return h_sigma*sin(b)**2+h_pi*cos(b)**2

def pzpy(h_sigma,h_pi,angle=(0,0)): 
	return pypz(h_sigma,h_pi,angle)
	
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

#interatomic distance
def distance(a1,a2,a3,pos1,pos2,cellpos2):
	return  np.linalg.norm((cellpos2[0]*a1+cellpos2[1]*a2+cellpos2[2]*a3+pos2)-pos1)	

#hopping matrix for a general position	
def sp3s_hoppingmatrix(bondparameters,angle):
	return [[bondparameters[0],s_px(bondparameters[1],angle),s_py(bondparameters[1],angle),s_pz(bondparameters[1],angle),bondparameters[5]],
			[-s_px(bondparameters[1],angle),pxpx(bondparameters[2],bondparameters[3],angle),pxpy(bondparameters[2],bondparameters[3],angle),pxpz(bondparameters[2],bondparameters[3],angle),-s_px(bondparameters[4],angle)],
			[-s_py(bondparameters[1],angle),pypx(bondparameters[2],bondparameters[3],angle),pypy(bondparameters[2],bondparameters[3],angle),pypz(bondparameters[2],bondparameters[3],angle),-s_py(bondparameters[4],angle)],
			[-s_pz(bondparameters[1],angle),pzpx(bondparameters[2],bondparameters[3],angle),pzpy(bondparameters[2],bondparameters[3],angle),pzpz(bondparameters[2],bondparameters[3],angle),-s_pz(bondparameters[4],angle)],
			[bondparameters[5],s_px(bondparameters[4],angle),s_py(bondparameters[4],angle),s_pz(bondparameters[4],angle),bondparameters[6]]]


""" Define the heterostructure a=lattice parameter xy, c=lattice parameter z, shift=valence band offset factor
"""			
def Hetero(a,c,shift):


	InSe_a = 0.400
	InSe_c = 1.688
	InSe_z1 = 0.091
	InSe_z2 = 0.167
	
	InSe_M_Es = -12.97
	InSe_X_Es = -21.73
	InSe_M_Ep = -6.73
	InSe_X_Ep = -10.82
	InSe_M_Ese = -2.34
	InSe_X_Ese = -4.89
	
	InSe_d2=0.2802079999999998
	InSe_d1=0.2641801360385246
	InSe_d3=0.4
	InSe_d4=0.4
	InSe_d5=0.38433709681649686
	
	InSe_bondpar1=(-1.001,2.105,3.290,-0.421,0.913,0,0)
	InSe_bondpar2=(-1.701,1.493,2.327,-0.968,1.896,0,0)
	InSe_bondpar3=(-0.133,0.293,1.119,0.204,0.597,0,0)
	InSe_bondpar4=(0.082,0.233,0.114,-0.109, -0.220,0,0)
	InSe_bondpar5=(-0.055,-0.147,0.582,-0.159,-0.051,0,0)
	
		
	GaSe_a = 0.3755
	GaSe_c = 1.594
	GaSe_z1 = 0.1
	GaSe_z2 = 0.173
	
	#include bands offset
	GaSe_M_Es = -12.63-shift
	GaSe_X_Es = -22.60-shift
	GaSe_M_Ep = -6.44-shift
	GaSe_X_Ep = -12.30-shift
	GaSe_M_Ese = -2.60-shift
	GaSe_X_Ese = -4.89-shift
	
	GaSe_d2=0.2454759999999998
	GaSe_d1=0.24604917877801047
	GaSe_d3=0.37549999999999994
	GaSe_d4=0.3755
	GaSe_d5=0.38553018472401523
	
	GaSe_bondpar1=(-0.988,2.057,2.803,-0.533, 0.822,-0.333,2.253)
	GaSe_bondpar2=(-2.241,1.881,2.462,-1.013,0.0,-0.279,-0.240)
	GaSe_bondpar3=(-0.102,0.085,0.774,-0.115,0.561,0.007,0.415)
	GaSe_bondpar4=(-0.133,0.242, 0.330,-0.075, 0.488,-0.386,1.110)
	GaSe_bondpar5=(-0.05,0.051,0.483,-0.149,0.249,-0.010,-0.125)
	
	
	mixed_bondpar5=(-0.052,-0.048,0.533,-0.155,0.044,-0.005,-0.062)
	
	a1l=np.array([a/2, (a/2) * sqrt(3),0])
	a2l=np.array([-a/2, (a/2) * sqrt(3),0]) 
	a3l=np.array([0,0,c])	
	X1pos = (2*(a1l+a2l)/3+InSe_z1*a3l)
	X2pos = (2*(a1l+a2l)/3+(1/2-InSe_z1)*a3l)
	X3pos =((a1l+a2l)/3+(1/2+InSe_z1)*a3l)
	X4pos = ((a1l+a2l)/3+(1-InSe_z1)*a3l)
	M1pos = (1*(a1l+a2l)/3+InSe_z2*a3l)
	M2pos = ((a1l+a2l)/3+(1/2-InSe_z2)*a3l)
	M3pos = (2*(a1l+a2l)/3+(1/2+InSe_z2)*a3l)
	M4pos = (2*(a1l+a2l)/3+(1-InSe_z2)*a3l)
	
	X1bpos = (2*(a1l+a2l)/3+GaSe_z1*a3l+a3l)
	X2bpos = (2*(a1l+a2l)/3+(1/2-GaSe_z1)*a3l+a3l)
	X3bpos =((a1l+a2l)/3+(1/2+GaSe_z1)*a3l+a3l)
	X4bpos = ((a1l+a2l)/3+(1-GaSe_z1)*a3l+a3l)
	M1bpos = (1*(a1l+a2l)/3+GaSe_z2*a3l+a3l)
	M2bpos = ((a1l+a2l)/3+(1/2-GaSe_z2)*a3l+a3l)
	M3bpos = (2*(a1l+a2l)/3+(1/2+GaSe_z2)*a3l+a3l)
	M4bpos = (2*(a1l+a2l)/3+(1-GaSe_z2)*a3l+a3l)
	
	"""strain correction based on the derivative of hopping parameter in the scaling law: Vnnn~d^-2
	"""
	InSe_bondpar1= np.array(InSe_bondpar1) *(1- 2*(distance(a1l,a2l,a3l,M3pos,M4pos,[0,0,0])-InSe_d2)/InSe_d2)
	InSe_bondpar2= np.array(InSe_bondpar2) *(1- 2*(distance(a1l,a2l,a3l,M3pos,M3pos,[0,-1,0])-InSe_d3)/InSe_d3)
	InSe_bondpar3=np.array(InSe_bondpar3)*(1- 2*(distance(a1l,a2l,a3l,X3pos,M3pos,[0,-1,0])-InSe_d1)/InSe_d1)
	InSe_bondpar4= np.array(InSe_bondpar4) *(1- 2*(distance(a1l,a2l,a3l,X3pos,X3pos,[0,1,0])-InSe_d4)/InSe_d4)
	InSe_bondpar5= np.array(InSe_bondpar5) *(1- 2*(distance(a1l,a2l,a3l,X2pos,X3pos,[0,1,0])-InSe_d5)/InSe_d5)
	
	GaSe_bondpar1= np.array(GaSe_bondpar1) *(1- 2*(distance(a1l,a2l,a3l,M3pos,M4pos,[0,0,0])-GaSe_d2)/GaSe_d2)
	GaSe_bondpar2= np.array(GaSe_bondpar2) *(1- 2*(distance(a1l,a2l,a3l,M3pos,M3pos,[0,-1,0])-GaSe_d3)/GaSe_d3)
	GaSe_bondpar3=np.array(GaSe_bondpar3)*(1- 2*(distance(a1l,a2l,a3l,X3pos,M3pos,[0,-1,0])-GaSe_d1)/GaSe_d1)
	GaSe_bondpar4= np.array(GaSe_bondpar4) *(1- 2*(distance(a1l,a2l,a3l,X3pos,X3pos,[0,1,0])-GaSe_d4)/GaSe_d4)
	GaSe_bondpar5= np.array(GaSe_bondpar5) *(1- 2*(distance(a1l,a2l,a3l,X2pos,X3pos,[0,1,0])-GaSe_d5)/GaSe_d5)
	
	lat = pb.Lattice(a1=a1l.tolist(), a2=a2l.tolist(), a3=(2*a3l).tolist())

	lat.add_sublattices(
		('X1', X1pos.tolist(), [[InSe_X_Es,0,0,0,0],
								[0,InSe_X_Ep,0,0,0],
								[0,0,InSe_X_Ep,0,0],
								[0,0,0,InSe_X_Ep,0],
								[0,0,0,0,InSe_X_Ese]]),
		('X2', X2pos.tolist(), [[InSe_X_Es,0,0,0,0],
								[0,InSe_X_Ep,0,0,0],
								[0,0,InSe_X_Ep,0,0],
								[0,0,0,InSe_X_Ep,0],
								[0,0,0,0,InSe_X_Ese]]),
		('M1', M1pos.tolist(), [[InSe_M_Es,0,0,0,0],
								[0,InSe_M_Ep,0,0,0],
								[0,0,InSe_M_Ep,0,0],
								[0,0,0,InSe_M_Ep,0],
								[0,0,0,0,InSe_M_Ese]]),
		('M2', M2pos.tolist(), [[InSe_M_Es,0,0,0,0],
								[0,InSe_M_Ep,0,0,0],
								[0,0,InSe_M_Ep,0,0],
								[0,0,0,InSe_M_Ep,0],
								[0,0,0,0,InSe_M_Ese]]),
		('X3', X3pos.tolist(), [[InSe_X_Es,0,0,0,0],
								[0,InSe_X_Ep,0,0,0],
								[0,0,InSe_X_Ep,0,0],
								[0,0,0,InSe_X_Ep,0],
								[0,0,0,0,InSe_X_Ese]]),
		('X4', X4pos.tolist(), [[InSe_X_Es,0,0,0,0],
								[0,InSe_X_Ep,0,0,0],
								[0,0,InSe_X_Ep,0,0],
								[0,0,0,InSe_X_Ep,0],
								[0,0,0,0,InSe_X_Ese]]),
		('M3', M3pos.tolist(), [[InSe_M_Es,0,0,0,0],
								[0,InSe_M_Ep,0,0,0],
								[0,0,InSe_M_Ep,0,0],
								[0,0,0,InSe_M_Ep,0],
								[0,0,0,0,InSe_M_Ese]]),
		('M4', M4pos.tolist(), [[InSe_M_Es,0,0,0,0],
								[0,InSe_M_Ep,0,0,0],
								[0,0,InSe_M_Ep,0,0],
								[0,0,0,InSe_M_Ep,0],
								[0,0,0,0,InSe_M_Ese]]),
								
		('X1b', X1bpos.tolist(), [[GaSe_X_Es,0,0,0,0],
								[0,GaSe_X_Ep,0,0,0],
								[0,0,GaSe_X_Ep,0,0],
								[0,0,0,GaSe_X_Ep,0],
								[0,0,0,0,GaSe_X_Ese]]),
		('X2b', X2bpos.tolist(), [[GaSe_X_Es,0,0,0,0],
								[0,GaSe_X_Ep,0,0,0],
								[0,0,GaSe_X_Ep,0,0],
								[0,0,0,GaSe_X_Ep,0],
								[0,0,0,0,GaSe_X_Ese]]),
		('M1b', M1bpos.tolist(), [[GaSe_M_Es,0,0,0,0],
								[0,GaSe_M_Ep,0,0,0],
								[0,0,GaSe_M_Ep,0,0],
								[0,0,0,GaSe_M_Ep,0],
								[0,0,0,0,GaSe_M_Ese]]),
		('M2b', M2bpos.tolist(), [[GaSe_M_Es,0,0,0,0],
								[0,GaSe_M_Ep,0,0,0],
								[0,0,GaSe_M_Ep,0,0],
								[0,0,0,GaSe_M_Ep,0],
								[0,0,0,0,GaSe_M_Ese]]),
		('X3b', X3bpos.tolist(), [[GaSe_X_Es,0,0,0,0],
								[0,GaSe_X_Ep,0,0,0],
								[0,0,GaSe_X_Ep,0,0],
								[0,0,0,GaSe_X_Ep,0],
								[0,0,0,0,GaSe_X_Ese]]),
		('X4b', X4bpos.tolist(), [[GaSe_X_Es,0,0,0,0],
								[0,GaSe_X_Ep,0,0,0],
								[0,0,GaSe_X_Ep,0,0],
								[0,0,0,GaSe_X_Ep,0],
								[0,0,0,0,GaSe_X_Ese]]),
		('M3b', M3bpos.tolist(), [[GaSe_M_Es,0,0,0,0],
								[0,GaSe_M_Ep,0,0,0],
								[0,0,GaSe_M_Ep,0,0],
								[0,0,0,GaSe_M_Ep,0],
								[0,0,0,0,GaSe_M_Ese]]),
		('M4b', M4bpos.tolist(), [[GaSe_M_Es,0,0,0,0],
								[0,GaSe_M_Ep,0,0,0],
								[0,0,GaSe_M_Ep,0,0],
								[0,0,0,GaSe_M_Ep,0],
								[0,0,0,0,GaSe_M_Ese]])
	)
	
	lat.add_hoppings(
		#Metal-Metal Bond 2
		([0, 0, 0], 'M3', 'M4', sp3s_hoppingmatrix(InSe_bondpar2,angles(a1l,a2l,a3l,[1,1,0],M3pos,M4pos,[0,0,0]))),
		

		# Ga-Se Bond 1
		([0, 0, 0], 'X3', 'M3', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X3pos,M3pos,[0,0,0]))),
		([0, -1, 0], 'X3', 'M3', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X3pos,M3pos,[0,-1,0]))),
		([-1, 0, 0], 'X3', 'M3', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X3pos,M3pos,[-1,0,0]))),
		([0, 0, 0], 'X4', 'M4', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X4pos,M4pos,[0,0,0]))),
		([-1, 0, 0], 'X4', 'M4', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X4pos,M4pos,[0,-1,0]))),
		([0, -1, 0], 'X4', 'M4', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X4pos,M4pos,[-1,0,0]))),
		
								
		# Ga-Ga Bond 3
		([0, -1, 0], 'M3', 'M3', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M3pos,M3pos,[0,-1,0]))),
		([1, 0, 0], 'M3', 'M3', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M3pos,M3pos,[1,0,0]))),		
		([1, -1, 0], 'M3', 'M3', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M3pos,M3pos,[1,-1,0]))),
		([0, -1, 0], 'M4', 'M4', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M4pos,M4pos,[0,-1,0]))),
		([1, 0, 0], 'M4', 'M4', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M4pos,M4pos,[1,0,0]))),		
		([1, -1, 0], 'M4', 'M4', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M4pos,M4pos,[1,-1,0]))),
		
							
		# Se-Se Bond 4
		([0, 1, 0], 'X3', 'X3', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X3pos,X3pos,[0,1,0]))),
		([1, 0, 0], 'X3', 'X3', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X3pos,X3pos,[1,0,0]))),
		([1, -1, 0], 'X3', 'X3', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X3pos,X3pos,[1,-1,0]))),
		([0, 1, 0], 'X4', 'X4', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X4pos,X4pos,[0,1,0]))),
		([1, 0, 0], 'X4', 'X4', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X4pos,X4pos,[1,0,0]))),
		([1, -1, 0], 'X4', 'X4', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X4pos,X4pos,[1,-1,0]))),
		
								
		#Ga-Ga Bond 2
		([0, 0, 0], 'M1', 'M2', sp3s_hoppingmatrix(InSe_bondpar2,angles(a1l,a2l,a3l,[1,1,0],M1pos,M2pos,[0,0,0]))),

		# Ga-Se Bond 1
		([0, 0, 0], 'X1', 'M1', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X1pos,M1pos,[0,0,0]))),
		([0, 1, 0], 'X1', 'M1', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X1pos,M1pos,[0,1,0]))),
		([1, 0, 0], 'X1', 'M1', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X1pos,M1pos,[1,0,0]))),
		([0, 0, 0], 'X2', 'M2', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X2pos,M2pos,[0,0,0]))),
		([1, 0, 0], 'X2', 'M2', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X2pos,M2pos,[0,1,0]))),
		([0, 1, 0], 'X2', 'M2', sp3s_hoppingmatrix(InSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X2pos,M2pos,[1,0,0]))),
		
		# Ga-Ga Bond 3
		([0, -1, 0], 'M1', 'M1', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M1pos,M1pos,[0,-1,0]))),
		([1, 0, 0], 'M1', 'M1', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M1pos,M1pos,[1,0,0]))),		
		([1, -1, 0], 'M1', 'M1', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M1pos,M1pos,[1,-1,0]))),
		([0, -1, 0], 'M2', 'M2', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M2pos,M2pos,[0,-1,0]))),
		([1, 0, 0], 'M2', 'M2', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M2pos,M2pos,[1,0,0]))),		
		([1, -1, 0], 'M2', 'M2', sp3s_hoppingmatrix(InSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M2pos,M2pos,[1,-1,0]))),
		
		# Se-Se Bond 4
		([0, 1, 0], 'X1', 'X1', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X1pos,X1pos,[0,1,0]))),
		([1, 0, 0], 'X1', 'X1', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X1pos,X1pos,[1,0,0]))),
		([1, -1, 0], 'X1', 'X1', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X1pos,X1pos,[1,-1,0]))),
		([0, 1, 0], 'X2', 'X2', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X2pos,X2pos,[0,1,0]))),
		([1, 0, 0], 'X2', 'X2', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X2pos,X2pos,[1,0,0]))),
		([1, -1, 0], 'X2', 'X2', sp3s_hoppingmatrix(InSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X2pos,X2pos,[1,-1,0]))),

		([0, 1, 0], 'X2', 'X3', sp3s_hoppingmatrix(InSe_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X2pos,X3pos,[0,1,0]))),
		([1, 0, 0], 'X2', 'X3', sp3s_hoppingmatrix(InSe_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X2pos,X3pos,[1,0,0]))),
		([0, 0, 0], 'X2', 'X3', sp3s_hoppingmatrix(InSe_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X2pos,X3pos,[0,0,0]))),
		([0, -1, 0], 'X4', 'X1b', sp3s_hoppingmatrix(mixed_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X4pos,X1bpos,[0,-1,0]))),
		([-1, 0, 0], 'X4', 'X1b', sp3s_hoppingmatrix(mixed_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X4pos,X1bpos,[-1,0,0]))),
		([0,0, 0], 'X4', 'X1b', sp3s_hoppingmatrix(mixed_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X4pos,X1bpos,[0,0,0]))),
		
		##GaSe 
		#Metal-Metal Bond 2
		([0, 0, 0], 'M3b', 'M4b', sp3s_hoppingmatrix(GaSe_bondpar2,angles(a1l,a2l,a3l,[1,1,0],M3bpos,M4bpos,[0,0,0]))),
		

		# Ga-Se Bond 1
		([0, 0, 0], 'X3b', 'M3b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X3bpos,M3bpos,[0,0,0]))),
		([0, -1, 0], 'X3b', 'M3b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X3bpos,M3bpos,[0,-1,0]))),
		([-1, 0, 0], 'X3b', 'M3b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X3bpos,M3bpos,[-1,0,0]))),
		([0, 0, 0], 'X4b', 'M4b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X4bpos,M4bpos,[0,0,0]))),
		([-1, 0, 0], 'X4b', 'M4b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X4bpos,M4bpos,[0,-1,0]))),
		([0, -1, 0], 'X4b', 'M4b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X4bpos,M4bpos,[-1,0,0]))),
		
								
		# Ga-Ga Bond 3
		([0, -1, 0], 'M3b', 'M3b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M3bpos,M3bpos,[0,-1,0]))),
		([1, 0, 0], 'M3b', 'M3b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M3bpos,M3bpos,[1,0,0]))),		
		([1, -1, 0], 'M3b', 'M3b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M3bpos,M3bpos,[1,-1,0]))),
		([0, -1, 0], 'M4b', 'M4b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M4bpos,M4bpos,[0,-1,0]))),
		([1, 0, 0], 'M4b', 'M4b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M4bpos,M4bpos,[1,0,0]))),		
		([1, -1, 0], 'M4b', 'M4b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M4bpos,M4bpos,[1,-1,0]))),
		
							
		# Se-Se Bond 4
		([0, 1, 0], 'X3b', 'X3b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X3bpos,X3bpos,[0,1,0]))),
		([1, 0, 0], 'X3b', 'X3b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X3bpos,X3bpos,[1,0,0]))),
		([1, -1, 0], 'X3b', 'X3b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X3bpos,X3bpos,[1,-1,0]))),
		([0, 1, 0], 'X4b', 'X4b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X4bpos,X4bpos,[0,1,0]))),
		([1, 0, 0], 'X4b', 'X4b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X4bpos,X4bpos,[1,0,0]))),
		([1, -1, 0], 'X4b', 'X4b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X4bpos,X4bpos,[1,-1,0]))),
		
								
		#Ga-Ga Bond 2
		([0, 0, 0], 'M1b', 'M2b', sp3s_hoppingmatrix(GaSe_bondpar2,angles(a1l,a2l,a3l,[1,1,0],M1bpos,M2bpos,[0,0,0]))),

		# Ga-Se Bond 1
		([0, 0, 0], 'X1b', 'M1b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X1bpos,M1bpos,[0,0,0]))),
		([0, 1, 0], 'X1b', 'M1b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X1bpos,M1bpos,[0,1,0]))),
		([1, 0, 0], 'X1b', 'M1b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X1bpos,M1bpos,[1,0,0]))),
		([0, 0, 0], 'X2b', 'M2b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X2bpos,M2bpos,[0,0,0]))),
		([1, 0, 0], 'X2b', 'M2b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X2bpos,M2bpos,[0,1,0]))),
		([0, 1, 0], 'X2b', 'M2b', sp3s_hoppingmatrix(GaSe_bondpar1,angles(a1l,a2l,a3l,[1,1,0],X2bpos,M2bpos,[1,0,0]))),
		
		# Ga-Ga Bond 3
		([0, -1, 0], 'M1b', 'M1b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M1bpos,M1bpos,[0,-1,0]))),
		([1, 0, 0], 'M1b', 'M1b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M1bpos,M1bpos,[1,0,0]))),		
		([1, -1, 0], 'M1b', 'M1b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M1bpos,M1bpos,[1,-1,0]))),
		([0, -1, 0], 'M2b', 'M2b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M2bpos,M2bpos,[0,-1,0]))),
		([1, 0, 0], 'M2b', 'M2b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M2bpos,M2bpos,[1,0,0]))),		
		([1, -1, 0], 'M2b', 'M2b', sp3s_hoppingmatrix(GaSe_bondpar3,angles(a1l,a2l,a3l,[1,1,0],M2bpos,M2bpos,[1,-1,0]))),
		
		# Se-Se Bond 4
		([0, 1, 0], 'X1b', 'X1b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X1bpos,X1bpos,[0,1,0]))),
		([1, 0, 0], 'X1b', 'X1b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X1bpos,X1bpos,[1,0,0]))),
		([1, -1, 0], 'X1b', 'X1b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X1bpos,X1bpos,[1,-1,0]))),
		([0, 1, 0], 'X2b', 'X2b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X2bpos,X2bpos,[0,1,0]))),
		([1, 0, 0], 'X2b', 'X2b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X2bpos,X2bpos,[1,0,0]))),
		([1, -1, 0], 'X2b', 'X2b', sp3s_hoppingmatrix(GaSe_bondpar4,angles(a1l,a2l,a3l,[1,1,0],X2bpos,X2bpos,[1,-1,0]))),

		([0, 1, 0], 'X2b', 'X3b', sp3s_hoppingmatrix(GaSe_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X2bpos,X3bpos,[0,1,0]))),
		([1, 0, 0], 'X2b', 'X3b', sp3s_hoppingmatrix(GaSe_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X2bpos,X3bpos,[1,0,0]))),
		([0, 0, 0], 'X2b', 'X3b', sp3s_hoppingmatrix(GaSe_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X2bpos,X3bpos,[0,0,0]))),
		([0, -1, 1], 'X4b', 'X1', sp3s_hoppingmatrix(mixed_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X4bpos,X1pos,[0,-1,1]))),
		([-1, 0, 1], 'X4b', 'X1', sp3s_hoppingmatrix(mixed_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X4bpos,X1pos,[-1,0,1]))),
		([0,0, 1], 'X4b', 'X1', sp3s_hoppingmatrix(mixed_bondpar5,angles(a1l,a2l,a3l,[1,1,0],X4bpos,X1pos,[0,0,1]))),
		
	)

	
	return lat
	
lattice = Hetero(0.385,1.6,0.3)

model = pb.Model(
    lattice,
    pb.translational_symmetry()
	#pb.primitive(a1=3, a2=3,a3=1)
)
print(model.hamiltonian.todense().shape)
solver = pb.solver.lapack(model)

a = 0.385
c = 2*1.594
gamma = [0, 0, 0]
k = [4*pi/(3*a),0,0]
m = [pi/a,-pi/(sqrt(3)*a),0]
apoint = [0, 0, pi/c]

plt.figure(figsize=(8, 2.3))

plt.subplot(141, title="GaSe 40-band structure")
bands = solver.calc_bands(k,gamma, m, gamma, apoint)
bands.plot(point_labels=["K",r"$\Gamma$", "M", r"$\Gamma$","A"])

plt.subplot(132, title="ldos Conduction")
ldos_map = solver.calc_spatial_ldos(energy=-9.86, broadening=0.1)  # [eV]
ldos_map.plot(axes="yz")

plt.subplot(133, title="ldos Valence")
ldos_map = solver.calc_spatial_ldos(energy=-10.61, broadening=0.1)  # [eV]
ldos_map.plot(axes="yz")

plt.show()
	
	
	
