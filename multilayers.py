import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, cos, sin
from scipy import integrate

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
	
def distance(a1,a2,a3,pos1,pos2,cellpos2):
	return  np.linalg.norm((cellpos2[0]*a1+cellpos2[1]*a2+cellpos2[2]*a3+pos2)-pos1)
	
def sp3s_hoppingmatrix(bondparameters,angle):
	return [[bondparameters[0],s_px(bondparameters[1],angle),s_py(bondparameters[1],angle),s_pz(bondparameters[1],angle),bondparameters[5]],
			[-s_px(bondparameters[1],angle),pxpx(bondparameters[2],bondparameters[3],angle),pxpy(bondparameters[2],bondparameters[3],angle),pxpz(bondparameters[2],bondparameters[3],angle),-s_px(bondparameters[4],angle)],
			[-s_py(bondparameters[1],angle),pypx(bondparameters[2],bondparameters[3],angle),pypy(bondparameters[2],bondparameters[3],angle),pypz(bondparameters[2],bondparameters[3],angle),-s_py(bondparameters[4],angle)],
			[-s_pz(bondparameters[1],angle),pzpx(bondparameters[2],bondparameters[3],angle),pzpy(bondparameters[2],bondparameters[3],angle),pzpz(bondparameters[2],bondparameters[3],angle),-s_pz(bondparameters[4],angle)],
			[bondparameters[5],s_px(bondparameters[4],angle),s_py(bondparameters[4],angle),s_pz(bondparameters[4],angle),bondparameters[6]]]

class PTMC:
	pass

class InSe(PTMC):
	#lattice
	a = 0.400
	c = 1.688/2
	z1 = 2*0.091
	z2 = 2*0.167

	#on-site therms
	M_Es = -12.97
	X_Es = -21.73
	M_Ep = -6.73
	X_Ep = -10.82
	M_Ese = -2.34
	X_Ese = -4.89

	#bond lenght
	d2=0.2802079999999998
	d1=0.2641801360385246
	d3=0.4
	d4=0.4
	d5=0.38433709681649686
	
	#bond sp3s* parameters
	bondpar_1=np.array([-1.001,2.105,3.290,-0.421,0.913,0,0])
	bondpar_2=np.array([-1.701,1.493,2.327,-0.968,1.896,0,0])
	bondpar_3=np.array([-0.133,0.293,1.119,0.204,0.597,0,0])
	bondpar_4=np.array([0.082,0.233,0.114,-0.109, -0.220,0,0])
	bondpar_5=np.array([-0.055,-0.147,0.582,-0.159,-0.051,0,0])

class GaSe(PTMC):
	#lattice
	a = 0.3755
	c = 1.594/2
	z1 = 2*0.1
	z2 = 2*0.173
	#on-site therms
	M_Es = -12.63
	X_Es = -22.60
	M_Ep = -6.44
	X_Ep = -12.30
	M_Ese = -2.60
	X_Ese = -4.89
	
	#bond lenght
	d2=0.2454759999999998
	d1=0.24604917877801047
	d3=0.37549999999999994
	d4=0.3755
	d5=0.38553018472401523
	
	#bond sp3s* parameters
	bondpar_1=np.array([-0.988,2.057,2.803,-0.533, 0.822,-0.333,2.253])
	bondpar_2=np.array([-2.241,1.881,2.462,-1.013,0.0,-0.279,-0.240])
	bondpar_3=np.array([-0.102,0.085,0.774,-0.115,0.561,0.007,0.415])
	bondpar_4=np.array([-0.133,0.242, 0.330,-0.075, 0.488,-0.386,1.110])
	bondpar_5=np.array([-0.05,0.051,0.483,-0.149,0.249,-0.010,-0.125])

class Layer:
	def __init__(self,material,a,parity,layerstart=0):
		self.a1l=np.array([a/2, (a/2) * sqrt(3),0])
		self.a2l=np.array([-a/2, (a/2) * sqrt(3),0])
		self.a3l=np.array([0,0,material.c])
		self.zero=np.array([0,0,layerstart])
		self.c=material.c

		if parity % 2 == 0:
			self.X1pos = (2*(self.a1l+self.a2l)/3+(material.z1)*self.a3l+self.zero)
			self.X2pos = (2*(self.a1l+self.a2l)/3+(1-material.z1)*self.a3l+self.zero)
			self.M1pos = (1*(self.a1l+self.a2l)/3+(material.z2)*self.a3l+self.zero)
			self.M2pos = ((self.a1l+self.a2l)/3+(1-material.z2)*self.a3l+self.zero)
		else:
			self.X1pos =((self.a1l+self.a2l)/3+(material.z1)*self.a3l+self.zero)
			self.X2pos = ((self.a1l+self.a2l)/3+(1-material.z1)*self.a3l+self.zero)
			self.M1pos = (2*(self.a1l+self.a2l)/3+(material.z2)*self.a3l+self.zero)
			self.M2pos = (2*(self.a1l+self.a2l)/3+(1-material.z2)*self.a3l+self.zero)
	
		#lattice
		self.a = a
		self.c = material.c
		self.z1 = material.z1
		self.z2 = material.z2
		#on-site therms
		self.M_Es = material.M_Es
		self.X_Es = material.X_Es
		self.M_Ep = material.M_Ep
		self.X_Ep = material.X_Ep 
		self.M_Ese = material.M_Ese
		self.X_Ese = material.X_Ese 
		
		#bond lenght
		self.d2 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M2pos,[0,0,0])
		self.d1 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.M1pos,[0,0,0])
		self.d3 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M1pos,[0,-1,0])
		self.d4 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.X1pos,[0,1,0])
		self.d5 = material.d5 #approximation

		print('d1='+str(self.d1-material.d1))
		print('d2='+str(self.d2-material.d2))
		print('d3='+str(self.d3-material.d3))
		print('d4='+str(self.d4-material.d4))
		print('d5='+str(self.d5-material.d5))

		
		#bond sp3s* parameters
		self.bondpar_2= material.bondpar_2*(1- 2*(self.d2-material.d2)/material.d2)
		self.bondpar_3= material.bondpar_3*(1- 2*(self.d3-material.d3)/material.d3)
		self.bondpar_1= material.bondpar_1*(1- 2*(self.d1-material.d1)/material.d1)
		self.bondpar_4= material.bondpar_4*(1- 2*(self.d4-material.d4)/material.d4)
		self.bondpar_5= material.bondpar_5


class Stack:
	def calc_c(self,layerlist):
		c=0
		for i in layerlist:
			c += i.c
		return c
	
	def __init__(self,materiallist, a):
		self.a = a
		self.materiallist = materiallist
		self.c = self.calc_c(materiallist)
		self.filled_c = 0
		self.layerlist = []
		self.a1l=np.array([a/2, (a/2) * sqrt(3),0])
		self.a2l=np.array([-a/2, (a/2) * sqrt(3),0]) 
		self.a3l=np.array([0,0,self.c])
		
		self.lat = pb.Lattice(a1=self.a1l.tolist(), a2=self.a2l.tolist(), a3=self.a3l.tolist())

		index = 0
		for material in self.materiallist:
			self.layerlist.append(Layer(material,self.a,index,self.filled_c))
			self.filled_c += material.c
			index += 1

		index = 0
		for layer in self.layerlist:
			self.add_layer(layer,index)
			index += 1

		
		for i in range(len(self.layerlist)):
			if i == len(self.layerlist)-1:
				if i % 2 != 0:
					self.lat.add_hoppings(
						([0, -1, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,-1,1]))),
						([-1, 0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[-1,0,1]))),
						([0,0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,0,1]))),
					)
				else: 
					self.lat.add_hoppings(
						([0, 1, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,1,1]))),
						([1, 0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[1,0,1]))),
						([0, 0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,0,1]))),
					)
			else:
				if i % 2 != 0:
					self.lat.add_hoppings(
						([0, -1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,-1,0]))),
						([-1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[-1,0,0]))),
						([0,0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,0,0]))),
					)
				else: 
					self.lat.add_hoppings(
						([0, 1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,1,0]))),
						([1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[1,0,0]))),
						([0, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,0,0]))),
					)
		
	def add_layer(self,layer,index):

		self.lat.add_sublattices(
			('X1-'+str(index), layer.X1pos.tolist(), [[layer.X_Es,0,0,0,0],
								[0,layer.X_Ep,0,0,0],
								[0,0,layer.X_Ep,0,0],
								[0,0,0,layer.X_Ep,0],
								[0,0,0,0,layer.X_Ese]]),
			('X2-'+str(index), layer.X2pos.tolist(), [[layer.X_Es,0,0,0,0],
								[0,layer.X_Ep,0,0,0],
								[0,0,layer.X_Ep,0,0],
								[0,0,0,layer.X_Ep,0],
								[0,0,0,0,layer.X_Ese]]),
			('M1-'+str(index), layer.M1pos.tolist(), [[layer.M_Es,0,0,0,0],
								[0,layer.M_Ep,0,0,0],
								[0,0,layer.M_Ep,0,0],
								[0,0,0,layer.M_Ep,0],
								[0,0,0,0,layer.M_Ese]]),
			('M2-'+str(index), layer.M2pos.tolist(), [[layer.M_Es,0,0,0,0],
								[0,layer.M_Ep,0,0,0],
								[0,0,layer.M_Ep,0,0],
								[0,0,0,layer.M_Ep,0],
								[0,0,0,0,layer.M_Ese]]),
		)

		if index % 2 == 0:
			self.lat.add_hoppings(
				#Ga-Ga bond 2
				([0, 0, 0], 'M1-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_2,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M2pos,[0,0,0]))),

				# Ga-Se Bond 1
				([0, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,0,0]))),
				([0, 1, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[1,0,0]))),
				([0, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,0,0]))),
				([1, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[1,0,0]))),
				
				# Ga-Ga Bond 3
				([0, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[0,-1,0]))),
				([1, 0, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,0,0]))),		
				([1, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,-1,0]))),
				([0, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[0,-1,0]))),
				([1, 0, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,0,0]))),		
				([1, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,-1,0]))),
				
				# Se-Se Bond 4
				([0, 1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,0,0]))),
				([1, -1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,-1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[0,1,0]))),
				([1, 0, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,0,0]))),
				([1, -1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,-1,0]))),
			)
		else:
			self.lat.add_hoppings(
				#Metal-Metal Bond 2
				([0, 0, 0], 'M1-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_2,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M2pos,[0,0,0]))),
				

				# Ga-Se Bond 1
				([0, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,0,0]))),
				([0, -1, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,-1,0]))),
				([-1, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[-1,0,0]))),
				([0, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,0,0]))),
				([-1, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,-1,0]))),
				([0, -1, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[-1,0,0]))),
				
										
				# Ga-Ga Bond 3
				([0, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[0,-1,0]))),
				([1, 0, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,0,0]))),		
				([1, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,-1,0]))),
				([0, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[0,-1,0]))),
				([1, 0, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,0,0]))),		
				([1, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,-1,0]))),
				
									
				# Se-Se Bond 4
				([0, 1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,0,0]))),
				([1, -1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,-1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[0,1,0]))),
				([1, 0, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,0,0]))),
				([1, -1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,-1,0]))),
			)

hetero = Stack([GaSe(),GaSe(),GaSe(),GaSe(),InSe(),InSe(),InSe(),InSe()],0.385)
	
lattice = hetero.lat

model = pb.Model(
    lattice,
    pb.translational_symmetry()
	#pb.primitive(a1=3, a2=3,a3=1)
)
print(model.hamiltonian.todense().shape)
solver = pb.solver.lapack(model)

a = 0.3755
c = 1.594
gamma = [0, 0, 0]
k = [4*pi/(3*a),0,0]
m = [pi/a,-pi/(sqrt(3)*a),0]
apoint = [0, 0, pi/c]

plt.figure(figsize=(8, 2.3))

ldos_map = solver.calc_spatial_ldos(energy=-10.517, broadening=0.1)  # [eV]

plt.subplot(141, title="GaSe 40-band structure")
bands = solver.calc_bands(k,gamma, m, gamma, apoint)
bands.plot(point_labels=["K",r"$\Gamma$", "M", r"$\Gamma$","A"])

plt.subplot(142, title="xz")
ldos_map.plot(axes="yz")

plt.subplot(143, title="yz")
model.plot(axes="yz")

plt.subplot(144, title="xy")
lattice.plot(axes="xy")

plt.show()
	
	
	
