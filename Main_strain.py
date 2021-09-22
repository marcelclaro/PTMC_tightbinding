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
	
def PTMC_layer(Metal,Chalcogenide,strain_xy=0,strain_z=0):

	if Metal == "In" and Chalcogenide == "Se":
		a = 0.400
		c = 1.688
		z1 = 0.091
		z2 = 0.167
		
		M_Es = -12.97
		X_Es = -21.73
		M_Ep = -6.73
		X_Ep = -10.82
		M_Ese = -2.34
		X_Ese = -4.89
		
		d2=0.2802079999999998
		d1=0.2641801360385246
		d3=0.4
		d4=0.4
		d5=0.38433709681649686
		
		bondpar_1=(-1.001,2.105,3.290,-0.421,0.913,0,0)
		bondpar_2=(-1.701,1.493,2.327,-0.968,1.896,0,0)
		bondpar_3=(-0.133,0.293,1.119,0.204,0.597,0,0)
		bondpar_4=(0.082,0.233,0.114,-0.109, -0.220,0,0)
		bondpar_5=(-0.055,-0.147,0.582,-0.159,-0.051,0,0)
		
	if Metal == "Ga" and Chalcogenide == "Se":
		a = 0.3755
		c = 1.594
		z1 = 0.1
		z2 = 0.173
		
		M_Es = -12.63
		X_Es = -22.60
		M_Ep = -6.44
		X_Ep = -12.30
		M_Ese = -2.60
		X_Ese = -4.89
		
		d2=0.2454759999999998
		d1=0.24604917877801047
		d3=0.37549999999999994
		d4=0.3755
		d5=0.38553018472401523
		
		bondpar_1=(-0.988,2.057,2.803,-0.533, 0.822,-0.333,2.253)
		bondpar_2=(-2.241,1.881,2.462,-1.013,0.0,-0.279,-0.240)
		bondpar_3=(-0.102,0.085,0.774,-0.115,0.561,0.007,0.415)
		bondpar_4=(-0.133,0.242, 0.330,-0.075, 0.488,-0.386,1.110)
		bondpar_5=(-0.05,0.051,0.483,-0.149,0.249,-0.010,-0.125)
	
	a1l=(1+strain_xy)*np.array([a/2, (a/2) * sqrt(3),0])
	a2l=(1+strain_xy)*np.array([-a/2, (a/2) * sqrt(3),0]) 
	a3l=(1+strain_z)*np.array([0,0,c])	
	X1pos = (2*(a1l+a2l)/3+z1*a3l)
	X2pos = (2*(a1l+a2l)/3+(1/2-z1)*a3l)
	X3pos =((a1l+a2l)/3+(1/2+z1)*a3l)
	X4pos = ((a1l+a2l)/3+(1-z1)*a3l)
	M1pos = (1*(a1l+a2l)/3+z2*a3l)
	M2pos = ((a1l+a2l)/3+(1/2-z2)*a3l)
	M3pos = (2*(a1l+a2l)/3+(1/2+z2)*a3l)
	M4pos = (2*(a1l+a2l)/3+(1-z2)*a3l)
	
	lat = pb.Lattice(a1=a1l.tolist(), a2=a2l.tolist(), a3=a3l.tolist())

	lat.add_sublattices(
		('X1', X1pos.tolist(), [[X_Es,0,0,0,0],
								[0,X_Ep,0,0,0],
								[0,0,X_Ep,0,0],
								[0,0,0,X_Ep,0],
								[0,0,0,0,X_Ese]]),
		('X2', X2pos.tolist(), [[X_Es,0,0,0,0],
								[0,X_Ep,0,0,0],
								[0,0,X_Ep,0,0],
								[0,0,0,X_Ep,0],
								[0,0,0,0,X_Ese]]),
		('M1', M1pos.tolist(), [[M_Es,0,0,0,0],
								[0,M_Ep,0,0,0],
								[0,0,M_Ep,0,0],
								[0,0,0,M_Ep,0],
								[0,0,0,0,M_Ese]]),
		('M2', M2pos.tolist(), [[M_Es,0,0,0,0],
								[0,M_Ep,0,0,0],
								[0,0,M_Ep,0,0],
								[0,0,0,M_Ep,0],
								[0,0,0,0,M_Ese]]),
		('X3', X3pos.tolist(), [[X_Es,0,0,0,0],
								[0,X_Ep,0,0,0],
								[0,0,X_Ep,0,0],
								[0,0,0,X_Ep,0],
								[0,0,0,0,X_Ese]]),
		('X4', X4pos.tolist(), [[X_Es,0,0,0,0],
								[0,X_Ep,0,0,0],
								[0,0,X_Ep,0,0],
								[0,0,0,X_Ep,0],
								[0,0,0,0,X_Ese]]),
		('M3', M3pos.tolist(), [[M_Es,0,0,0,0],
								[0,M_Ep,0,0,0],
								[0,0,M_Ep,0,0],
								[0,0,0,M_Ep,0],
								[0,0,0,0,M_Ese]]),
		('M4', M4pos.tolist(), [[M_Es,0,0,0,0],
								[0,M_Ep,0,0,0],
								[0,0,M_Ep,0,0],
								[0,0,0,M_Ep,0],
								[0,0,0,0,M_Ese]])
	)
	
	print(2*(distance(a1l,a2l,a3l,M3pos,M4pos,[0,0,0])-d2)/d2)
	print(2*(distance(a1l,a2l,a3l,M3pos,M3pos,[0,-1,0])-d3)/d3)
	print(2*(distance(a1l,a2l,a3l,X3pos,M3pos,[0,-1,0])-d1)/d1)
	print(2*(distance(a1l,a2l,a3l,X3pos,X3pos,[0,1,0])-d4)/d4)
	print(2*(distance(a1l,a2l,a3l,X2pos,X3pos,[0,1,0])-d5)/d5)
	"""print(bondpar_1)
	print(bondpar_2)
	print(bondpar_3)
	print(bondpar_4)
	print(bondpar_5)
	bondpar_1= np.array(bondpar_1) *1
	bondpar_2= np.array(bondpar_2)*1
	bondpar_3=np.array( bondpar_3)*1
	bondpar_4= np.array(bondpar_4) *1
	bondpar_5= np.array(bondpar_5)*1
	"""
	bondpar_1= np.array(bondpar_1) *(1- 2*(distance(a1l,a2l,a3l,M3pos,M4pos,[0,0,0])-d2)/d2)
	bondpar_2= np.array(bondpar_2) *(1- 2*(distance(a1l,a2l,a3l,M3pos,M3pos,[0,-1,0])-d3)/d3)
	bondpar_3=np.array( bondpar_3)*(1- 2*(distance(a1l,a2l,a3l,X3pos,M3pos,[0,-1,0])-d1)/d1)
	bondpar_4= np.array(bondpar_4) *(1- 2*(distance(a1l,a2l,a3l,X3pos,X3pos,[0,1,0])-d4)/d4)
	bondpar_5= np.array(bondpar_5) *(1- 2*(distance(a1l,a2l,a3l,X2pos,X3pos,[0,1,0])-d5)/d5)
	"""print(bondpar_1)
	print(bondpar_2)
	print(bondpar_3)
	print(bondpar_4)
	print(bondpar_5)"""
	
	lat.add_hoppings(
		#Metal-Metal Bond 2
		([0, 0, 0], 'M3', 'M4', sp3s_hoppingmatrix(bondpar_2,angles(a1l,a2l,a3l,[1,1,0],M3pos,M4pos,[0,0,0]))),
		

		# Ga-Se Bond 1
		([0, 0, 0], 'X3', 'M3', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X3pos,M3pos,[0,0,0]))),
		([0, -1, 0], 'X3', 'M3', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X3pos,M3pos,[0,-1,0]))),
		([-1, 0, 0], 'X3', 'M3', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X3pos,M3pos,[-1,0,0]))),
		([0, 0, 0], 'X4', 'M4', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X4pos,M4pos,[0,0,0]))),
		([-1, 0, 0], 'X4', 'M4', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X4pos,M4pos,[0,-1,0]))),
		([0, -1, 0], 'X4', 'M4', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X4pos,M4pos,[-1,0,0]))),
		
								
		# Ga-Ga Bond 3
		([0, -1, 0], 'M3', 'M3', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M3pos,M3pos,[0,-1,0]))),
		([1, 0, 0], 'M3', 'M3', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M3pos,M3pos,[1,0,0]))),		
		([1, -1, 0], 'M3', 'M3', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M3pos,M3pos,[1,-1,0]))),
		([0, -1, 0], 'M4', 'M4', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M4pos,M4pos,[0,-1,0]))),
		([1, 0, 0], 'M4', 'M4', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M4pos,M4pos,[1,0,0]))),		
		([1, -1, 0], 'M4', 'M4', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M4pos,M4pos,[1,-1,0]))),
		
							
		# Se-Se Bond 4
		([0, 1, 0], 'X3', 'X3', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X3pos,X3pos,[0,1,0]))),
		([1, 0, 0], 'X3', 'X3', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X3pos,X3pos,[1,0,0]))),
		([1, -1, 0], 'X3', 'X3', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X3pos,X3pos,[1,-1,0]))),
		([0, 1, 0], 'X4', 'X4', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X4pos,X4pos,[0,1,0]))),
		([1, 0, 0], 'X4', 'X4', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X4pos,X4pos,[1,0,0]))),
		([1, -1, 0], 'X4', 'X4', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X4pos,X4pos,[1,-1,0]))),
		
								
		#Ga-Ga Bond 2
		([0, 0, 0], 'M1', 'M2', sp3s_hoppingmatrix(bondpar_2,angles(a1l,a2l,a3l,[1,1,0],M1pos,M2pos,[0,0,0]))),

		# Ga-Se Bond 1
		([0, 0, 0], 'X1', 'M1', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X1pos,M1pos,[0,0,0]))),
		([0, 1, 0], 'X1', 'M1', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X1pos,M1pos,[0,1,0]))),
		([1, 0, 0], 'X1', 'M1', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X1pos,M1pos,[1,0,0]))),
		([0, 0, 0], 'X2', 'M2', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X2pos,M2pos,[0,0,0]))),
		([1, 0, 0], 'X2', 'M2', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X2pos,M2pos,[0,1,0]))),
		([0, 1, 0], 'X2', 'M2', sp3s_hoppingmatrix(bondpar_1,angles(a1l,a2l,a3l,[1,1,0],X2pos,M2pos,[1,0,0]))),
		
		# Ga-Ga Bond 3
		([0, -1, 0], 'M1', 'M1', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M1pos,M1pos,[0,-1,0]))),
		([1, 0, 0], 'M1', 'M1', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M1pos,M1pos,[1,0,0]))),		
		([1, -1, 0], 'M1', 'M1', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M1pos,M1pos,[1,-1,0]))),
		([0, -1, 0], 'M2', 'M2', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M2pos,M2pos,[0,-1,0]))),
		([1, 0, 0], 'M2', 'M2', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M2pos,M2pos,[1,0,0]))),		
		([1, -1, 0], 'M2', 'M2', sp3s_hoppingmatrix(bondpar_3,angles(a1l,a2l,a3l,[1,1,0],M2pos,M2pos,[1,-1,0]))),
		
		# Se-Se Bond 4
		([0, 1, 0], 'X1', 'X1', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X1pos,X1pos,[0,1,0]))),
		([1, 0, 0], 'X1', 'X1', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X1pos,X1pos,[1,0,0]))),
		([1, -1, 0], 'X1', 'X1', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X1pos,X1pos,[1,-1,0]))),
		([0, 1, 0], 'X2', 'X2', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X2pos,X2pos,[0,1,0]))),
		([1, 0, 0], 'X2', 'X2', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X2pos,X2pos,[1,0,0]))),
		([1, -1, 0], 'X2', 'X2', sp3s_hoppingmatrix(bondpar_4,angles(a1l,a2l,a3l,[1,1,0],X2pos,X2pos,[1,-1,0]))),

		# Se-Se Bond 5
		([0, 1, 0], 'X2', 'X3', sp3s_hoppingmatrix(bondpar_5,angles(a1l,a2l,a3l,[1,1,0],X2pos,X3pos,[0,1,0]))),
		([1, 0, 0], 'X2', 'X3', sp3s_hoppingmatrix(bondpar_5,angles(a1l,a2l,a3l,[1,1,0],X2pos,X3pos,[1,0,0]))),
		([0, 0, 0], 'X2', 'X3', sp3s_hoppingmatrix(bondpar_5,angles(a1l,a2l,a3l,[1,1,0],X2pos,X3pos,[0,0,0]))),
		([0, -1, 1], 'X4', 'X1', sp3s_hoppingmatrix(bondpar_5,angles(a1l,a2l,a3l,[1,1,0],X4pos,X1pos,[0,-1,1]))),
		([-1, 0, 1], 'X4', 'X1', sp3s_hoppingmatrix(bondpar_5,angles(a1l,a2l,a3l,[1,1,0],X4pos,X1pos,[-1,0,1]))),
		([0,0, 1], 'X4', 'X1', sp3s_hoppingmatrix(bondpar_5,angles(a1l,a2l,a3l,[1,1,0],X4pos,X1pos,[0,0,1]))),
	)

	return lat
	
lattice = PTMC_layer("In","Se",-0.03,0.0)

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

kpm = pb.kpm(model)

dos = kpm.calc_dos(energy=np.linspace(-30, -10.0, 2000), broadening=0.2, num_random=16)
dos.plot()
plt.show()

for sub_name in ["X4"]:
	ldos = kpm.calc_ldos(energy=np.linspace(-30, -10, 2000), broadening=0.2,
		position=[0, 0,0], sublattice=sub_name)
	#ldos.plot(label=sub_name)
	print(6-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable))
for sub_name in [ "M4"]:
	ldos = kpm.calc_ldos(energy=np.linspace(-30, -10, 2000), broadening=0.2,
		position=[0, 0,0], sublattice=sub_name)
	#ldos.plot(label=sub_name)
	print(3-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable))
#pb.pltutils.legend()

plt.figure(figsize=(8, 2.3))

ldos_map = solver.calc_spatial_ldos(energy=-9.08, broadening=0.05)  # [eV]

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
	
	
	
