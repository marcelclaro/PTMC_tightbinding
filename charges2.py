import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, cos, sin
from scipy import integrate
from scipy.optimize import minimize as minimize_scipy
import matplotlib.mlab as mlab
import matplotlib.gridspec as gs
import sys
from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit

pb.pltutils.use_style()


"""
TO DO:
1-strain in c
	a. new c in each layer
	b. Bond 5 normalization

"""

eps0 = 8.854187E-12
e_charge = 1.60217662E-19

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
	a = 0.40231918225
	c = 0.8469198513
	z1 = 3*0.060940054
	z2 = 3*0.111797495

	vbo = 0.0
	#on-site therms
	M_Es = -10.6448985-vbo
	X_Es = -20.9312344-vbo
	M_Ep = -7.60662203-vbo
	X_Ep = -13.4686215 -vbo
	M_Ese = -3.20562046-vbo
	X_Ese = -8.36132106-vbo

	#bond lenght
	d1=0.26580159178743173
	d2=0.27881874425332515
	d3=0.40231918225
	d4=0.40231918224999996

	charge=1.4632

	eps = 7.63

	#bond sp3s* parameters
	bondpar_1=np.array([-0.93087028,2.17662119,1.93915036,-0.57577716,-0.10196477,-0.14880588,0.08525471])
	bondpar_2=np.array([-2.02255816,2.92183879,3.37148706,0.37062229,-0.07333330,-0.01383834,3.24458523])
	bondpar_3=np.array([-0.20668040,0.00438908,1.03275835,0.24037372,0.21419123,0.05811275,-0.18338033])
	bondpar_4=np.array([-0.14675106,0.80847511,0.47375311,0.00714412, 0.63814236 ,0.67006494,0.00871295])
	bondpar_5=np.array([-0.04120019,-0.50741037,0.66732466 ,-0.16202840,0.25408178,-0.37491349,0.00592817])

class GaSe(PTMC):
	#lattice
	a = 0.37618898094
	c = 0.7876028792
	z1 = 3*0.068891933
	z2 = 3*0.117076586
	#on-site therms
	M_Es = -10.6934605
	X_Es = -22.0747470 
	M_Ep = -8.67313571
	X_Ep = -13.8103861
	M_Ese = -1.56830496
	X_Ese = -7.75586591
	
	#bond lenght
	d1=0.2452239643457275
	d2=0.23434374187696183
	d3=0.37618898094
	d4=0.3761889809399999

	#dipole charge
	charge=1.3711

	eps = 7.63
	
	#bond sp3s* parameters
	bondpar_1=np.array([-1.05908662,2.47549236 ,2.16608104,-0.66687273 , -0.11141026 ,-0.15410760,0.03118546 ])
	bondpar_2=np.array([-2.00684280 ,3.24371535,3.35629446,0.10853752,0.73472794,0.11758892,4.32088163])
	bondpar_3=np.array([-0.16813146,-0.10493689,0.70639068,0.26897119,0.23332979,0.14969084,-0.27095969])
	bondpar_4=np.array([-0.14626485,-0.55295508, 0.44265988,0.02210112,0.60012662 ,0.50896764,-0.03574015])
	bondpar_5=np.array([0.05793399,-0.47473129,  0.67716787 ,-0.20068697,0.16859448,-0.27186365,0.01796498])

class fitting(PTMC):
	#bond lenght
	d1=0
	d2=0
	d3=0
	d4=0

	def __init__(self,pars,a,c,z1,z2):
		self.a=a
		self.c=c
		self.z1=z1
		self.z2=z2
		self.M_Es = pars['M_Es']
		self.X_Es = pars['X_Es']
		self.M_Ep = pars['M_Ep']
		self.X_Ep = pars['X_Ep']
		self.M_Ese = pars['M_Ese']
		self.X_Ese = pars['X_Ese']
		self.bondpar_1=np.array([pars['one_ss'],pars['one_sp'],pars['one_pp_s'],pars['one_pp_p'],pars['one_sep'],pars['one_ses'],pars['one_sese']])
		self.bondpar_2=np.array([pars['two_ss'],pars['two_sp'],pars['two_pp_s'],pars['two_pp_p'],pars['two_sep'],pars['two_ses'],pars['two_sese']])
		self.bondpar_3=np.array([pars['three_ss'],pars['three_sp'],pars['three_pp_s'],pars['three_pp_p'],pars['three_sep'],pars['three_ses'],pars['three_sese']])
		self.bondpar_4=np.array([pars['four_ss'],pars['four_sp'],pars['four_pp_s'],pars['four_pp_p'],pars['four_sep'],pars['four_ses'],pars['four_sese']])
		self.bondpar_5=np.array([pars['five_ss'],pars['five_sp'],pars['five_pp_s'],pars['five_pp_p'],pars['five_sep'],pars['five_ses'],pars['five_sese']])

class Layer:
	def __init__(self,material,a,parity,layerstart=0,strained=True):
		self.a1l=np.array([a/2, (a/2) * sqrt(3),0])
		self.a2l=np.array([-a/2, (a/2) * sqrt(3),0])
		self.a3l=np.array([0,0,material.c])
		self.zero=np.array([0,0,layerstart])
		self.c=material.c

		if parity % 3 == 0:
			self.X1pos = (2*(self.a1l+self.a2l)/3+(material.z1)*self.a3l+self.zero)
			self.X2pos = (2*(self.a1l+self.a2l)/3+(1-material.z1)*self.a3l+self.zero)
			self.M1pos = (1*(self.a1l+self.a2l)+(material.z2)*self.a3l+self.zero)
			self.M2pos = (1*(self.a1l+self.a2l)+(1-material.z2)*self.a3l+self.zero)

			#bond lenght
			self.d2 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M2pos,[0,0,0])
			self.d1 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.M1pos,[0,0,0])
			self.d3 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M1pos,[0,-1,0])
			self.d4 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.X1pos,[0,1,0])

		elif parity % 3 == 1:
			self.X1pos =((material.z1)*self.a3l+self.zero)
			self.X2pos = ((1-material.z1)*self.a3l+self.zero)
			self.M1pos = ((self.a1l+self.a2l)/3+(material.z2)*self.a3l+self.zero)
			self.M2pos = ((self.a1l+self.a2l)/3+(1-material.z2)*self.a3l+self.zero)

			#bond lenght
			self.d2 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M2pos,[0,0,0])
			self.d1 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.M1pos,[0,-1,0])
			self.d3 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M1pos,[0,-1,0])
			self.d4 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.X1pos,[0,1,0])
		else:
			self.X1pos =((self.a1l+self.a2l)/3+(material.z1)*self.a3l+self.zero)
			self.X2pos = ((self.a1l+self.a2l)/3+(1-material.z1)*self.a3l+self.zero)
			self.M1pos = (2*(self.a1l+self.a2l)/3+(material.z2)*self.a3l+self.zero)
			self.M2pos = (2*(self.a1l+self.a2l)/3+(1-material.z2)*self.a3l+self.zero)

			#bond lenght
			self.d2 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M2pos,[0,0,0])
			self.d1 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.M1pos,[0,-1,0])
			self.d3 = distance(self.a1l,self.a2l,self.a3l,self.M1pos,self.M1pos,[0,-1,0])
			self.d4 = distance(self.a1l,self.a2l,self.a3l,self.X1pos,self.X1pos,[0,1,0])
	
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

		"""use it to get the d# values
		print('d1='+str(self.d1))
		print('d2='+str(self.d2))
		print('d3='+str(self.d3))
		print('d4='+str(self.d4))"""
		
		#bond sp3s* parameters
		if strained == True:
			self.bondpar_2= material.bondpar_2*(1- 2*(self.d2-material.d2)/material.d2)
			self.bondpar_3= material.bondpar_3*(1- 2*(self.d3-material.d3)/material.d3)
			self.bondpar_1= material.bondpar_1*(1- 2*(self.d1-material.d1)/material.d1)
			self.bondpar_4= material.bondpar_4*(1- 2*(self.d4-material.d4)/material.d4)
			self.bondpar_5= material.bondpar_5
		else:
			self.bondpar_2= material.bondpar_2
			self.bondpar_3= material.bondpar_3
			self.bondpar_1= material.bondpar_1
			self.bondpar_4= material.bondpar_4
			self.bondpar_5= material.bondpar_5


class Stack:
	def calc_c(self,layerlist):
		c=0
		for i in layerlist:
			c += i.c
		return c
	
	def __init__(self,materiallist, a,strained=True,potential=[]):
		self.a = a
		self.materiallist = materiallist
		if len(materiallist) % 3 != 0:
			raise ValueError("gamma polymorph must have multiple of 3 layers!")
		self.c = self.calc_c(materiallist)
		self.filled_c = 0
		self.layerlist = []
		self.potential = potential
		self.a1l=np.array([a/2, (a/2) * sqrt(3),0])
		self.a2l=np.array([-a/2, (a/2) * sqrt(3),0]) 
		self.a3l=np.array([0,0,self.c])
		
		self.lat = pb.Lattice(a1=self.a1l.tolist(), a2=self.a2l.tolist(), a3=self.a3l.tolist())

		self.planes = []
		self.defaultcharge = []



		for index,material in enumerate(self.materiallist):
			self.layerlist.append(Layer(material,self.a,index,self.filled_c,strained=strained))
			self.planes.extend([self.filled_c+material.z1,self.filled_c+material.z2,self.filled_c+material.c-material.z2,self.filled_c+material.c-material.z1])
			self.defaultcharge.extend([-material.charge,+material.charge,+material.charge,-material.charge])
			self.filled_c += material.c


		for index,layer in enumerate(self.layerlist):
			if len(self.potential) != 0:
				self.add_layer(layer,index,self.potential[index*4:index*4+4])
			else:
				self.add_layer(layer,index,[0,0,0,0])


		
		for i in range(len(self.layerlist)):
			if i == (len(self.layerlist)-1):
				if i % 3 == 2:
					self.lat.add_hoppings(
						([0, -1, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,-1,1]))),
						([-1, 0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[-1,0,1]))),
						([0, 0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,0,1]))),
					)
				else:
					print("You should have define a number of layer multiple of 3  - because it is gamma phase")

			else:
				if i % 3 == 0:
					self.lat.add_hoppings(
						([0, 1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,1,0]))),
						([1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[1,0,0]))),
						([1,1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[1,1,0]))),
					)
				elif i % 3 == 1: 
					self.lat.add_hoppings(
						([0, -1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,-1,0]))),
						([-1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[-1,0,0]))),
						([0,0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,0,0]))),
					)
				else:
					self.lat.add_hoppings(
						([0, -1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,-1,0]))),
						([-1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[-1,0,0]))),
						([0, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,0,0]))),
					)
		
		
		
	def add_layer(self,layer,index,potential):

		self.lat.add_sublattices(
			('X1-'+str(index), layer.X1pos.tolist(), [[layer.X_Es+potential[0],0,0,0,0],
								[0,layer.X_Ep+potential[0],0,0,0],
								[0,0,layer.X_Ep+potential[0],0,0],
								[0,0,0,layer.X_Ep+potential[0],0],
								[0,0,0,0,layer.X_Ese+potential[0]]]),
			('X2-'+str(index), layer.X2pos.tolist(), [[layer.X_Es+potential[1],0,0,0,0],
								[0,layer.X_Ep+potential[1],0,0,0],
								[0,0,layer.X_Ep+potential[1],0,0],
								[0,0,0,layer.X_Ep+potential[1],0],
								[0,0,0,0,layer.X_Ese+potential[1]]]),
			('M1-'+str(index), layer.M1pos.tolist(), [[layer.M_Es+potential[2],0,0,0,0],
								[0,layer.M_Ep+potential[2],0,0,0],
								[0,0,layer.M_Ep+potential[2],0,0],
								[0,0,0,layer.M_Ep+potential[2],0],
								[0,0,0,0,layer.M_Ese+potential[2]]]),
			('M2-'+str(index), layer.M2pos.tolist(), [[layer.M_Es+potential[3],0,0,0,0],
								[0,layer.M_Ep+potential[3],0,0,0],
								[0,0,layer.M_Ep+potential[3],0,0],
								[0,0,0,layer.M_Ep+potential[3],0],
								[0,0,0,0,layer.M_Ese+potential[3]]]),
		)

		if index % 3 == 0:
			self.lat.add_hoppings(
				#Ga-Ga bond 2
				([0, 0, 0], 'M1-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_2,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M2pos,[0,0,0]))),

				# Ga_se Bond 1
				([0, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,0,0]))),
				([0, -1, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,-1,0]))),
				([-1, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[-1,0,0]))),
				([0, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,0,0]))),
				([-1, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[-1,0,0]))),
				([0, -1, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,-1,0]))),
				
				# Ga-Ga Bond 3
				([0, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[0,-1,0]))),
				([1, 0, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,0,0]))),		
				([1, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,-1,0]))),
				([0, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[0,-1,0]))),
				([1, 0, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,0,0]))),		
				([1, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,-1,0]))),
				
				# Se_se Bond 4
				([0, 1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,0,0]))),
				([1, -1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,-1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[0,1,0]))),
				([1, 0, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,0,0]))),
				([1, -1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,-1,0]))),
			)
		elif index % 3 == 1:
			self.lat.add_hoppings(
				#Metal-Metal Bond 2
				([0, 0, 0], 'M1-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_2,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M2pos,[0,0,0]))),
				

				# Ga_se Bond 1
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
				
									
				# Se_se Bond 4
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
				

				# Ga_se Bond 1
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
				
									
				# Se_se Bond 4
				([0, 1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,0,0]))),
				([1, -1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,-1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[0,1,0]))),
				([1, 0, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,0,0]))),
				([1, -1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,-1,0]))),
			)

def apply_scissor(band,last_valence,exp_gap):
	top_valence=np.max(band[last_valence])
	bottom_conduction=np.min(band[last_valence+1])
	gap=bottom_conduction - top_valence
	deltagap=exp_gap-gap
	modified_band = np.array(band)
	for i in range(len(band)):
		if i > last_valence:
			modified_band[i,:] += deltagap
	return modified_band


def residuals(pars,dft,weight,a,c,z1,z2):
	material = fitting(pars,a,c,z1,z2)
	mat = Stack([material,material,material],a,strained=False)
	lattice = mat.lat

	kpoints= np.array([[0.0000000000,0.0000000000,0.5000000000],
			[0.0000000000,0.0185185185,0.5000000000],
			[0.0000000000,0.0370370370,0.5000000000],
			[0.0000000000,0.0555555556,0.5000000000],
			[0.0000000000,0.0740740741,0.5000000000],
			[0.0000000000,0.0925925926,0.5000000000],
			[0.0000000000,0.1111111111,0.5000000000],
			[0.0000000000,0.1296296296,0.5000000000],
			[0.0000000000,0.1481481481,0.5000000000],
			[0.0000000000,0.1666666667,0.5000000000],
			[0.0000000000,0.1851851852,0.5000000000],
			[0.0000000000,0.2037037037,0.5000000000],
			[0.0000000000,0.2222222222,0.5000000000],
			[0.0000000000,0.2407407407,0.5000000000],
			[0.0000000000,0.2592592593,0.5000000000],
			[0.0000000000,0.2777777778,0.5000000000],
			[0.0000000000,0.2962962963,0.5000000000],
			[0.0000000000,0.3148148148,0.5000000000],
			[0.0000000000,0.3333333333,0.5000000000],
			[0.0000000000,0.3518518519,0.5000000000],
			[0.0000000000,0.3703703704,0.5000000000],
			[0.0000000000,0.3888888889,0.5000000000],
			[0.0000000000,0.4074074074,0.5000000000],
			[0.0000000000,0.4259259259,0.5000000000],
			[0.0000000000,0.4444444444,0.5000000000],
			[0.0000000000,0.4629629630,0.5000000000],
			[0.0000000000,0.4814814815,0.5000000000],
			[0.0000000000,0.5000000000,0.5000000000],
			[0.0222220000,0.4888886667,0.5000000000],
			[0.0444440000,0.4777773333,0.5000000000],
			[0.0666660000,0.4666660000,0.5000000000],
			[0.0888880000,0.4555546667,0.5000000000],
			[0.1111100000,0.4444433333,0.5000000000],
			[0.1333320000,0.4333320000,0.5000000000],
			[0.1555540000,0.4222206667,0.5000000000],
			[0.1777760000,0.4111093333,0.5000000000],
			[0.1999980000,0.3999980000,0.5000000000],
			[0.2222200000,0.3888866667,0.5000000000],
			[0.2444420000,0.3777753333,0.5000000000],
			[0.2666640000,0.3666640000,0.5000000000],
			[0.2888860000,0.3555526667,0.5000000000],
			[0.3111080000,0.3444413333,0.5000000000],
			[0.3333300000,0.3333300000,0.5000000000],
			[0.3225774194,0.3225774194,0.5000000000],
			[0.3118248387,0.3118248387,0.5000000000],
			[0.3010722581,0.3010722581,0.5000000000],
			[0.2903196774,0.2903196774,0.5000000000],
			[0.2795670968,0.2795670968,0.5000000000],
			[0.2688145161,0.2688145161,0.5000000000],
			[0.2580619355,0.2580619355,0.5000000000],
			[0.2473093548,0.2473093548,0.5000000000],
			[0.2365567742,0.2365567742,0.5000000000],
			[0.2258041935,0.2258041935,0.5000000000],
			[0.2150516129,0.2150516129,0.5000000000],
			[0.2042990323,0.2042990323,0.5000000000],
			[0.1935464516,0.1935464516,0.5000000000],
			[0.1827938710,0.1827938710,0.5000000000],
			[0.1720412903,0.1720412903,0.5000000000],
			[0.1612887097,0.1612887097,0.5000000000],
			[0.1505361290,0.1505361290,0.5000000000],
			[0.1397835484,0.1397835484,0.5000000000],
			[0.1290309677,0.1290309677,0.5000000000],
			[0.1182783871,0.1182783871,0.5000000000],
			[0.1075258065,0.1075258065,0.5000000000],
			[0.0967732258,0.0967732258,0.5000000000],
			[0.0860206452,0.0860206452,0.5000000000],
			[0.0752680645,0.0752680645,0.5000000000],
			[0.0645154839,0.0645154839,0.5000000000],
			[0.0537629032,0.0537629032,0.5000000000],
			[0.0430103226,0.0430103226,0.5000000000],
			[0.0322577419,0.0322577419,0.5000000000],
			[0.0215051613,0.0215051613,0.5000000000],
			[0.0107525806,0.0107525806,0.5000000000],
			[0.0000000000,0.0000000000,0.5000000000],
			[0.0000000000,0.0000000000,0.3333333333],
			[0.0000000000,0.0000000000,0.1666666667],
			[0.0000000000,0.0000000000,0.0000000000],
			[0.0000000000,0.0185185185,0.0000000000],
			[0.0000000000,0.0370370370,0.0000000000],
			[0.0000000000,0.0555555556,0.0000000000],
			[0.0000000000,0.0740740741,0.0000000000],
			[0.0000000000,0.0925925926,0.0000000000],
			[0.0000000000,0.1111111111,0.0000000000],
			[0.0000000000,0.1296296296,0.0000000000],
			[0.0000000000,0.1481481481,0.0000000000],
			[0.0000000000,0.1666666667,0.0000000000],
			[0.0000000000,0.1851851852,0.0000000000],
			[0.0000000000,0.2037037037,0.0000000000],
			[0.0000000000,0.2222222222,0.0000000000],
			[0.0000000000,0.2407407407,0.0000000000],
			[0.0000000000,0.2592592593,0.0000000000],
			[0.0000000000,0.2777777778,0.0000000000],
			[0.0000000000,0.2962962963,0.0000000000],
			[0.0000000000,0.3148148148,0.0000000000],
			[0.0000000000,0.3333333333,0.0000000000],
			[0.0000000000,0.3518518519,0.0000000000],
			[0.0000000000,0.3703703704,0.0000000000],
			[0.0000000000,0.3888888889,0.0000000000],
			[0.0000000000,0.4074074074,0.0000000000],
			[0.0000000000,0.4259259259,0.0000000000],
			[0.0000000000,0.4444444444,0.0000000000],
			[0.0000000000,0.4629629630,0.0000000000],
			[0.0000000000,0.4814814815,0.0000000000],
			[0.0000000000,0.5000000000,0.0000000000],
			[0.0222220000,0.4888886667,0.0000000000],
			[0.0444440000,0.4777773333,0.0000000000],
			[0.0666660000,0.4666660000,0.0000000000],
			[0.0888880000,0.4555546667,0.0000000000],
			[0.1111100000,0.4444433333,0.0000000000],
			[0.1333320000,0.4333320000,0.0000000000],
			[0.1555540000,0.4222206667,0.0000000000],
			[0.1777760000,0.4111093333,0.0000000000],
			[0.1999980000,0.3999980000,0.0000000000],
			[0.2222200000,0.3888866667,0.0000000000],
			[0.2444420000,0.3777753333,0.0000000000],
			[0.2666640000,0.3666640000,0.0000000000],
			[0.2888860000,0.3555526667,0.0000000000],
			[0.3111080000,0.3444413333,0.0000000000],
			[0.3333300000,0.3333300000,0.0000000000],
			[0.3225774194,0.3225774194,0.0000000000],
			[0.3118248387,0.3118248387,0.0000000000],
			[0.3010722581,0.3010722581,0.0000000000],
			[0.2903196774,0.2903196774,0.0000000000],
			[0.2795670968,0.2795670968,0.0000000000],
			[0.2688145161,0.2688145161,0.0000000000],
			[0.2580619355,0.2580619355,0.0000000000],
			[0.2473093548,0.2473093548,0.0000000000],
			[0.2365567742,0.2365567742,0.0000000000],
			[0.2258041935,0.2258041935,0.0000000000],
			[0.2150516129,0.2150516129,0.0000000000],
			[0.2042990323,0.2042990323,0.0000000000],
			[0.1935464516,0.1935464516,0.0000000000],
			[0.1827938710,0.1827938710,0.0000000000],
			[0.1720412903,0.1720412903,0.0000000000],
			[0.1612887097,0.1612887097,0.0000000000],
			[0.1505361290,0.1505361290,0.0000000000],
			[0.1397835484,0.1397835484,0.0000000000],
			[0.1290309677,0.1290309677,0.0000000000],
			[0.1182783871,0.1182783871,0.0000000000],
			[0.1075258065,0.1075258065,0.0000000000],
			[0.0967732258,0.0967732258,0.0000000000],
			[0.0860206452,0.0860206452,0.0000000000],
			[0.0752680645,0.0752680645,0.0000000000],
			[0.0645154839,0.0645154839,0.0000000000],
			[0.0537629032,0.0537629032,0.0000000000],
			[0.0430103226,0.0430103226,0.0000000000],
			[0.0322577419,0.0322577419,0.0000000000],
			[0.0215051613,0.0215051613,0.0000000000],
			[0.0107525806,0.0107525806,0.0000000000],
			[0.0000000000,0.0000000000,0.0000000000],
			[0.0000000000,0.5000000000,0.5000000000],
			[0.0000000000,0.5000000000,0.3750000000],
			[0.0000000000,0.5000000000,0.2500000000],
			[0.0000000000,0.5000000000,0.1250000000],
			[0.0000000000,0.5000000000,0.0000000000],
			[0.3333300000,0.3333300000,0.5000000000],
			[0.3333300000,0.3333300000,0.3750000000],
			[0.3333300000,0.3333300000,0.2500000000],
			[0.3333300000,0.3333300000,0.1250000000],
			[0.3333300000,0.3333300000,0.0000000000]])
	points_index = [0,27,42,73,76,103,118,149,154,159]
	kpoints[:,0] *= pi*2/a
	kpoints[:,1] *= pi*2/a
	kpoints[:,2] *= pi*2/c

	#print(dft.shape)

	model = pb.Model(lattice,
		pb.translational_symmetry()
		#pb.primitive(a1=3, a2=3,a3=1)
	)
	#print(model.hamiltonian.todense().shape)
	solver = pb.solver.lapack(model)

	modelbands = solver.calc_bands_withpoints(kpoints,points_index)
	bandenergies = np.transpose(modelbands.energy)[:33,:]
	#print(bandenergies.shape)


	residuals = (bandenergies-dft)*weight
	print(np.sum(residuals**2))
	return residuals

def getCharges(model,stack,insidegap):
	charges = []
	kpm = pb.kpm(model,silent=True)
	for n,layer in enumerate(stack.layerlist):
		ldos = kpm.calc_ldos(energy=np.linspace(-30, insidegap, 4000), broadening=0.05,
			position=[0, 0,0], sublattice="X1-"+str(n))
		chargeX1=6-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable)
		ldos = kpm.calc_ldos(energy=np.linspace(-30, insidegap, 4000), broadening=0.05,
			position=[0, 0,0], sublattice="M1-"+str(n))
		chargeM1=3-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable)
		ldos = kpm.calc_ldos(energy=np.linspace(-30, insidegap, 4000), broadening=0.05,
			position=[0, 0,0], sublattice="X2-"+str(n))
		chargeX2=6-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable)
		ldos = kpm.calc_ldos(energy=np.linspace(-30, insidegap, 4000), broadening=0.05,
			position=[0, 0,0], sublattice="M2-"+str(n))
		chargeM2=3-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable)		
		charges.extend([chargeX1,chargeM1,chargeM2,chargeX2])
	
	return np.array(charges)

def getPotential(stack,excess_charges):
	V = [0,]
	for n in range(1,len(stack.layerlist)*4):
		distancelist = -(np.array(stack.planes)-stack.planes[n])*1E-9
		distancelist[n:] *= 0
		distancelist *= excess_charges*e_charge/(7.36*eps0*(stack.a*1E-9)**2*sin(pi*30/180))
		V.append(np.sum(distancelist))
	return V


def smooth(y):
	box = np.ones(4)/4
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

def residuals_pot(potential,materiallist,a):
	mat = Stack(materiallist,a,strained=False,potential=smooth(potential))
	lattice = mat.lat

	model = pb.Model(
	lattice,
	pb.translational_symmetry()
	#pb.primitive(a1=3, a2=3,a3=1)
	)

	chargescalc = getCharges(model,mat,-10.76)
	print(chargescalc)
	print(potential)
	excess_charge = np.abs(chargescalc)-np.abs(np.array(mat.defaultcharge))
	print("chi**2="+str(np.sum(excess_charge**2)))
	return np.sum(excess_charge**2)

materials = [InSe(),InSe(),InSe(),GaSe(),GaSe(),GaSe()]
potential = np.empty(len(materials)*4)
potential[:] = 0
potential[len(materials)*2:] = -0.01
"""potential = smooth([ 7.36514306e-02,5.32729842e-02,1.22326952e-02,-8.19263627e-08
,4.64836557e-02,4.39959444e-02,-2.50824516e-05,-1.44859592e-03
,3.86739278e-02,7.05021736e-02,-8.75314708e-03,6.38281240e-03
,-9.49207005e-02,-6.67744478e-02,-1.85111897e-02,-7.37129459e-03
,-6.43758167e-02,-6.90653995e-02,-6.73317692e-03,-1.40206266e-03
,-6.00071294e-02,-1.04571429e-01,2.03394645e-03,-1.36254221e-02])"""


out = minimize_scipy(residuals_pot, potential, args=(materials,0.388), method='Powell',options={'maxiter':100, 'maxfev':100000, "disp":True})
print(out.x)

potential = out.x
mat = Stack(materials,0.388,strained=False)
lattice = mat.lat
a=mat.a
c=mat.c
gamma = [0, 0, 0]
A = [0, 0, pi/c]
k = [4*pi/(3*a),0,0]
m = [pi/a,-pi/(sqrt(3)*a),0]
l = [4*pi/(3*a),0,pi/c]
h = [pi/a,-pi/(sqrt(3)*a),pi/c]

model = pb.Model(
lattice,
pb.translational_symmetry()
#pb.primitive(a1=3, a2=3,a3=1)
)
print(model.hamiltonian.todense().shape)
solver = pb.solver.lapack(model)

plt.scatter(mat.planes,smooth(potential))
plt.show()

"""
plt.figure(figsize=(8, 2.3))
plt.subplot(131, title="band structure")
bands = solver.calc_bands(A,l,h,A,gamma,k,m,gamma)
bands.plot(point_labels=["A","L","H","A",r"$\Gamma$", "K", "M", r"$\Gamma$"])

plt.subplot(132, title="Val.")
ldos_map = solver.calc_spatial_ldos(energy=-11.43, broadening=0.05)  # [eV]
ldos_map.plot(axes="yz")

plt.subplot(133, title="Cond.")
ldos_map = solver.calc_spatial_ldos(energy=-10.31, broadening=0.05)  # [eV]
ldos_map.plot(axes="yz")
plt.show()	
"""	
