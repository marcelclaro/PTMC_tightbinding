import pybinding as pb
import numpy as np
from math import sqrt, pi, cos, sin
from .hopping import sp3s_hoppingmatrix_SO
from .util import distance, angles

pb.pltutils.use_style()

"""
TO DO:
1-strain in c
	a. new c in each layer
	b. Bond 5 normalization

"""

"""
This class define a layer of PTMC (D3h symmetry) for a R3m crystal, 
it requires to be initialized with a material, the parity in the layer stack multiple of 1 to 3,
position where layer start and if the layer is strained using the Harrison scale law
"""
class Layer:
	def __init__(self,material,a,parity,layerstart=0,strained=False):
		#Here define the lattice parameters
		self.a1l=np.array([a/2, (a/2) * sqrt(3),0])
		self.a2l=np.array([-a/2, (a/2) * sqrt(3),0])
		self.a3l=np.array([0,0,material.c])
		self.a = a
		self.c = material.c
		self.z1 = material.z1
		self.z2 = material.z2
		self.material = material
		
		#here define the starting position of the layer
		self.zero=np.array([0,0,layerstart])

		#Depending on the parity the position and the distance between atoms are set. They are diferent due to symmetry reasons
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
	
		
		#here we define the on-site elements
		self.M_Es = material.M_Es
		self.X_Es = material.X_Es
		self.M_Ep = material.M_Ep
		self.X_Ep = material.X_Ep 
		self.M_Ese = material.M_Ese
		self.X_Ese = material.X_Ese
		self.M_SOcoupling = material.M_SOcoupling
		self.X_SOcoupling = material.X_SOcoupling

		"""use it to get the d values for new materials
		print('d1='+str(self.d1))
		print('d2='+str(self.d2))
		print('d3='+str(self.d3))
		print('d4='+str(self.d4))"""
		
		#here the bond/ off-diagonal sp3s* parameters are define, in case of strain it scales as Harrison law
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

"""
"""
class Stack:
	#calculate total c lattice parameter of the stack
	def calc_c(self,layerlist):
		c=0
		for i in layerlist:
			c += i.c
		return c
	
	def __init__(self,materiallist, a,strained=False,potential=[]):

		#read the list of materials, and check if is multiple of 3, because it is a R3m crystal
		self.materiallist = materiallist
		if len(materiallist) % 3 != 0:
			raise ValueError("gamma polymorph must have multiple of 3 layers!")
		
		#create the lattice parameters
		self.a = a
		self.c = self.calc_c(materiallist)
		self.filled_c = 0
		self.layerlist = []
		self.a1l=np.array([a/2, (a/2) * sqrt(3),0])
		self.a2l=np.array([-a/2, (a/2) * sqrt(3),0]) 
		self.a3l=np.array([0,0,self.c])
		self.lat = pb.Lattice(a1=self.a1l.tolist(), a2=self.a2l.tolist(), a3=self.a3l.tolist())
		self.unitcells = int(len(self.materiallist)/3)
		self.VBMindex = int(self.unitcells*27-1)
		self.CBMindex = int(self.unitcells*27)
		
		#dielectric permitivitty 
		self.permittivity = 0
		
		#read the position of the planes and the defaut charge (like if it is a bulk material)
		self.planes = []
		self.atompositions = []
		self.defaultcharge = []
		self.potential = np.copy(potential)

		#create the list of layers
		for index,material in enumerate(self.materiallist):
			self.layerlist.append(Layer(material,self.a,index,self.filled_c,strained=strained))
			self.planes.extend([self.filled_c+material.z1,self.filled_c+material.z2,self.filled_c+material.c-material.z2,self.filled_c+material.c-material.z1])
			self.filled_c += material.c
			self.permittivity += material.permittivity

		#average permittivity 
		self.permittivity /= len(materiallist)

		#create a layer and setup the hopping matrix (without interlayer coupling)
		for index,layer in enumerate(self.layerlist):
			if len(self.potential) != 0:
				self.add_layer(layer,index,self.potential[index*4:index*4+4])
			else:
				self.add_layer(layer,index,[0,0,0,0])


		#interlayer coupling hopping elements
		for i in range(len(self.layerlist)):
			if i == (len(self.layerlist)-1):
				#last to first layer coupling
				if i % 3 == 2:
					self.lat.add_hoppings(
						([0, -1, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,-1,1]))),
						([-1, 0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[-1,0,1]))),
						([0, 0, 1], 'X2-'+str(i), 'X1-'+str(0), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[0].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[0].X1pos,[0,0,1]))),
					)
				else:
					print("You should have define a number of layer multiple of 3  - because it is gamma phase")
			#the rest of layers interlayer coupling
			else:
				if i % 3 == 0:
					self.lat.add_hoppings(
						([0, 1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,1,0]))),
						([1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[1,0,0]))),
						([1,1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[1,1,0]))),
					)
				elif i % 3 == 1: 
					self.lat.add_hoppings(
						([0, -1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,-1,0]))),
						([-1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[-1,0,0]))),
						([0,0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,0,0]))),
					)
				else:
					self.lat.add_hoppings(
						([0, -1, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,-1,0]))),
						([-1, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[-1,0,0]))),
						([0, 0, 0], 'X2-'+str(i), 'X1-'+str(i+1), sp3s_hoppingmatrix_SO((self.layerlist[i].bondpar_5+self.layerlist[i+1].bondpar_5)/2,angles(self.a1l,self.a2l,self.a3l,[1,1,0],self.layerlist[i].X2pos,self.layerlist[i+1].X1pos,[0,0,0]))),
					)
		
		
		
	def add_layer(self,layer,index,potential):
		#on-site/diagonal hopping elements
		self.lat.add_sublattices(
			('X1-'+str(index), layer.X1pos.tolist(), [[layer.X_Es+potential[0],0,0,0,0,0,0,0,0,0],
								[0,layer.X_Ep+potential[0],complex(0,-layer.X_SOcoupling),0,0,0,0,0,layer.X_SOcoupling,0],
								[0,complex(0,layer.X_SOcoupling),layer.X_Ep+potential[0],0,0,0,0,0,complex(0,-layer.X_SOcoupling),0],
								[0,0,0,layer.X_Ep+potential[0],0,0,-layer.X_SOcoupling,complex(0,layer.X_SOcoupling),0,0],
								[0,0,0,0,layer.X_Ese+potential[0],0,0,0,0,0],
								[0,0,0,0,0,layer.X_Es+potential[0],0,0,0,0],
								[0,0,0,-layer.X_SOcoupling,0,0,layer.X_Ep+potential[0],complex(0,layer.X_SOcoupling),0,0],
								[0,0,0,complex(0,-layer.X_SOcoupling),0,0,complex(0,-layer.X_SOcoupling),layer.X_Ep+potential[0],0,0],
								[0,layer.X_SOcoupling,complex(0,layer.X_SOcoupling),0,0,0,0,0,layer.X_Ep+potential[0],0],
								[0,0,0,0,0,0,0,0,0,layer.X_Ese+potential[0]]]),
			('M1-'+str(index), layer.M1pos.tolist(), [[layer.M_Es+potential[1],0,0,0,0,0,0,0,0,0],
								[0,layer.M_Ep+potential[1],complex(0,-layer.M_SOcoupling),0,0,0,0,0,layer.M_SOcoupling,0],
								[0,complex(0,layer.M_SOcoupling),layer.M_Ep+potential[1],0,0,0,0,0,complex(0,-layer.M_SOcoupling),0],
								[0,0,0,layer.M_Ep+potential[1],0,0,-layer.M_SOcoupling,complex(0,layer.M_SOcoupling),0,0],
								[0,0,0,0,layer.M_Ese+potential[1],0,0,0,0,0],
								[0,0,0,0,0,layer.M_Es+potential[1],0,0,0,0],
								[0,0,0,-layer.M_SOcoupling,0,0,layer.M_Ep+potential[1],complex(0,layer.M_SOcoupling),0,0],
								[0,0,0,complex(0,-layer.M_SOcoupling),0,0,complex(0,-layer.M_SOcoupling),layer.M_Ep+potential[1],0,0],
								[0,layer.M_SOcoupling,complex(0,layer.M_SOcoupling),0,0,0,0,0,layer.M_Ep+potential[1],0],
								[0,0,0,0,0,0,0,0,0,layer.M_Ese+potential[1]]]),
			('M2-'+str(index), layer.M2pos.tolist(), [[layer.M_Es+potential[2],0,0,0,0,0,0,0,0,0],
								[0,layer.M_Ep+potential[2],complex(0,-layer.M_SOcoupling),0,0,0,0,0,layer.M_SOcoupling,0],
								[0,complex(0,layer.M_SOcoupling),layer.M_Ep+potential[2],0,0,0,0,0,complex(0,-layer.M_SOcoupling),0],
								[0,0,0,layer.M_Ep+potential[2],0,0,-layer.M_SOcoupling,complex(0,layer.M_SOcoupling),0,0],
								[0,0,0,0,layer.M_Ese+potential[2],0,0,0,0,0],
								[0,0,0,0,0,layer.M_Es+potential[2],0,0,0,0],
								[0,0,0,-layer.M_SOcoupling,0,0,layer.M_Ep+potential[2],complex(0,layer.M_SOcoupling),0,0],
								[0,0,0,complex(0,-layer.M_SOcoupling),0,0,complex(0,-layer.M_SOcoupling),layer.M_Ep+potential[2],0,0],
								[0,layer.M_SOcoupling,complex(0,layer.M_SOcoupling),0,0,0,0,0,layer.M_Ep+potential[2],0],
								[0,0,0,0,0,0,0,0,0,layer.M_Ese+potential[2]]]),
			('X2-'+str(index), layer.X2pos.tolist(), [[layer.X_Es+potential[3],0,0,0,0,0,0,0,0,0],
								[0,layer.X_Ep+potential[3],complex(0,-layer.X_SOcoupling),0,0,0,0,0,layer.X_SOcoupling,0],
								[0,complex(0,layer.X_SOcoupling),layer.X_Ep+potential[3],0,0,0,0,0,complex(0,-layer.X_SOcoupling),0],
								[0,0,0,layer.X_Ep+potential[3],0,0,-layer.X_SOcoupling,complex(0,layer.X_SOcoupling),0,0],
								[0,0,0,0,layer.X_Ese+potential[3],0,0,0,0,0],
								[0,0,0,0,0,layer.X_Es+potential[3],0,0,0,0],
								[0,0,0,-layer.X_SOcoupling,0,0,layer.X_Ep+potential[3],complex(0,layer.X_SOcoupling),0,0],
								[0,0,0,complex(0,-layer.X_SOcoupling),0,0,complex(0,-layer.X_SOcoupling),layer.X_Ep+potential[3],0,0],
								[0,layer.X_SOcoupling,complex(0,layer.X_SOcoupling),0,0,0,0,0,layer.X_Ep+potential[3],0],
								[0,0,0,0,0,0,0,0,0,layer.X_Ese+potential[3]]]),
		)

		#list with position of each atom
		self.atompositions.extend([layer.X1pos.tolist(),layer.M1pos.tolist(),layer.M2pos.tolist(),layer.X2pos.tolist()])
		self.defaultcharge.extend([-layer.material.charge,layer.material.charge,layer.material.charge,-layer.material.charge])

		#off-diagonal elements of the layer, due to the symmetry it deppends on parity
		if index % 3 == 0:
			self.lat.add_hoppings(
				#M-M bond 2
				([0, 0, 0], 'M1-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_2,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M2pos,[0,0,0]))),

				# M-X Bond 1
				([0, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,0,0]))),
				([0, -1, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,-1,0]))),
				([-1, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[-1,0,0]))),
				([0, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,0,0]))),
				([-1, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[-1,0,0]))),
				([0, -1, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,-1,0]))),
				
				# M-M Bond 3
				([0, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[0,-1,0]))),
				([1, 0, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,0,0]))),		
				([1, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,-1,0]))),
				([0, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[0,-1,0]))),
				([1, 0, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,0,0]))),		
				([1, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,-1,0]))),
				
				# X_X Bond 4
				([0, 1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,0,0]))),
				([1, -1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,-1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[0,1,0]))),
				([1, 0, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,0,0]))),
				([1, -1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,-1,0]))),
			)
		elif index % 3 == 1:
			self.lat.add_hoppings(
				#Bond 2
				([0, 0, 0], 'M1-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_2,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M2pos,[0,0,0]))),
				

				#Bond 1
				([0, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,0,0]))),
				([0, -1, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,-1,0]))),
				([-1, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[-1,0,0]))),
				([0, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,0,0]))),
				([-1, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[-1,0,0]))),
				([0, -1, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,-1,0]))),
				
										
				#Bond 3
				([0, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[0,-1,0]))),
				([1, 0, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,0,0]))),		
				([1, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,-1,0]))),
				([0, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[0,-1,0]))),
				([1, 0, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,0,0]))),		
				([1, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,-1,0]))),
				
									
				#Bond 4
				([0, 1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,0,0]))),
				([1, -1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,-1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[0,1,0]))),
				([1, 0, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,0,0]))),
				([1, -1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,-1,0]))),
			)
		else:
			self.lat.add_hoppings(
				#Bond 2
				([0, 0, 0], 'M1-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_2,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M2pos,[0,0,0]))),
				

				#Bond 1
				([0, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,0,0]))),
				([0, -1, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[0,-1,0]))),
				([-1, 0, 0], 'X1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.M1pos,[-1,0,0]))),
				([0, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,0,0]))),
				([-1, 0, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[0,-1,0]))),
				([0, -1, 0], 'X2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_1,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.M2pos,[-1,0,0]))),
				
										
				#Bond 3
				([0, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[0,-1,0]))),
				([1, 0, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,0,0]))),		
				([1, -1, 0], 'M1-'+str(index), 'M1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M1pos,layer.M1pos,[1,-1,0]))),
				([0, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[0,-1,0]))),
				([1, 0, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,0,0]))),		
				([1, -1, 0], 'M2-'+str(index), 'M2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_3,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.M2pos,layer.M2pos,[1,-1,0]))),
				
									
				#Bond 4
				([0, 1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[0,1,0]))),
				([1, 0, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,0,0]))),
				([1, -1, 0], 'X1-'+str(index), 'X1-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X1pos,layer.X1pos,[1,-1,0]))),
				([0, 1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[0,1,0]))),
				([1, 0, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,0,0]))),
				([1, -1, 0], 'X2-'+str(index), 'X2-'+str(index), sp3s_hoppingmatrix_SO(layer.bondpar_4,angles(layer.a1l,layer.a2l,layer.a3l,[1,1,0],layer.X2pos,layer.X2pos,[1,-1,0]))),
			)