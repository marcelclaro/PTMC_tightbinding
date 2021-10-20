import pybinding as pb
import numpy as np
from math import sqrt, pi, cos, sin
from .hopping import sp3s_hoppingmatrix
from .util import distance, angles

pb.pltutils.use_style()

"""
TO DO:
1-strain in c
	a. new c in each layer
	b. Bond 5 normalization

"""

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

		"""use it to get the d values
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
			"""
			self.bondpar_2= material.bondpar_2*(material.d2/self.d2)**2
			self.bondpar_3= material.bondpar_3*(material.d3/self.d3)**2
			self.bondpar_1= material.bondpar_1*(material.d1/self.d1)**2
			self.bondpar_4= material.bondpar_4*(material.d4/self.d4)**2
			self.bondpar_5= material.bondpar_5"""
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
	
	def __init__(self,materiallist, a,strained=True):
		self.a = a
		self.materiallist = materiallist
		if len(materiallist) % 3 != 0:
			raise ValueError("gamma polymorph must have multiple of 3 layers!")
		self.c = self.calc_c(materiallist)
		self.filled_c = 0
		self.layerlist = []
		self.a1l=np.array([a/2, (a/2) * sqrt(3),0])
		self.a2l=np.array([-a/2, (a/2) * sqrt(3),0]) 
		self.a3l=np.array([0,0,self.c])
		self.permittivity = 0
		
		self.lat = pb.Lattice(a1=self.a1l.tolist(), a2=self.a2l.tolist(), a3=self.a3l.tolist())

		self.planes = []
		self.defaultcharge = []

		for index,material in enumerate(self.materiallist):
			self.layerlist.append(Layer(material,self.a,index,self.filled_c,strained=strained))
			self.planes.extend([self.filled_c+material.z1,self.filled_c+material.z2,self.filled_c+material.c-material.z2,self.filled_c+material.c-material.z1])
			self.defaultcharge.extend([-material.charge,material.charge,material.charge,-material.charge])
			self.filled_c += material.c
			self.permittivity += material.permittivity

		self.permittivity /= len(materiallist)


		for index,layer in enumerate(self.layerlist):
			self.add_layer(layer,index)


		
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