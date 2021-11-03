import numpy as np
from numpy.lib.function_base import _delete_dispatcher
from scipy import integrate
import pybinding as pb
from math import ceil,cos,exp,sin,sqrt, pi
from scipy import special
from PTMC_TB.gamma import Stack

#calculate charge using green function, insidegap parameter is a energy inside the band gap
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

def sign(value):
    if (value) == 0.0:
        return 0.0
    elif value < 0.0:
        return -1.0
    else:
        return +1.0

#calculate electric field based o excess charge, we use here parallel plane model (each atom form a plane in c direction)
def electricfield(pos,stack,excesscharges):
    area = (stack.a*1e-9)**2*sin(pi*30/180)
    permittivity = stack.permittivity
    field = np.zeros(len(pos))
    for i,value in enumerate(pos):
        for p,zprime in enumerate(stack.planes):
            field[i]+=-0.5*sign(value-zprime)*excesscharges[p]*1.60217662e-19/(permittivity*area)
    return field

"""
H. Fehske, R. Schneider and A. Weiße (Eds.), Computational Many-Particle Physics,
Lect. Notes Phys. 739 (Springer, Berlin Heidelberg 2008), DOI 10.1007/ 978-3-540-
74686-7
"""
def getpotential_ewald(stack: Stack,charges,alpha,rcut,kcut):
	positions = np.array(stack.atompositions)
	if len(positions) != len(charges):
		raise ValueError("charges and atomic sites are not in the same size")
	vs = np.zeros(len(charges))
	al1 = stack.a1l
	al2 = stack.a2l
	al3 = stack.a3l
	cell_vol = np.dot(al1*1e-9,np.cross(al2*1e-9,al3*1e-9))
	bk1 = np.cross(al2*1e-9,al3*1e-9) * 2*pi/cell_vol
	bk2 = np.cross(al3*1e-9,al1*1e-9) * 2*pi/cell_vol
	bk3 = np.cross(al1*1e-9,al2*1e-9) * 2*pi/cell_vol
	nx = 1+ceil(rcut / np.linalg.norm(al1))
	ny = 1+ceil(rcut / np.linalg.norm(al2))
	nz = 1+ceil(rcut / np.linalg.norm(al3))
	factor = 1/(4*stack.permittivity)
	for atom,currentpos in enumerate(positions):
		for ir1 in range(-nx, nx):
			for ir2 in range(-ny, ny):
				for ir3 in range(-nz, nz):
					relpos = np.copy(positions)+ir1*al1+ir2*al2+ir3*al3
					distance = np.linalg.norm(relpos - currentpos, axis=1)
					for j,dist in enumerate(distance):
						if dist != 0:
							vs[atom] += factor*charges[j]*1.60217662e-19*special.erfc(alpha*(dist*1e-9))/(dist*1e-9)

	for atom,currentpos in enumerate(positions):
		relpos = np.copy(positions)
		distance = relpos - currentpos
		for k1 in range(-kcut, kcut):
			for k2 in range(-kcut, kcut):
				for k3 in range(-kcut, kcut):
					if k1 ==0 and k2 == 0 and k3 ==0:
						continue
					k = k1*bk1+k2*bk2+k3*bk3
					for j,dist in enumerate(distance):
						if j != atom:
							vs[atom] += 4*pi*factor*charges[atom]*1.60217662e-19*exp(-np.dot(k,k)/(4*alpha**2))*cos(np.dot(k,dist*1e-9))/cell_vol
	
	vs -= factor*charges*1.60217662e-19*2*alpha/sqrt(pi)
	
	return vs

def getpotential_direct(stack: Stack,charges,rcut):
	positions = np.array(stack.atompositions)
	if len(positions) != len(charges):
		raise ValueError("charges and atomic sites are not in the same size")
	vs = np.zeros(len(charges))
	al1 = stack.a1l
	al2 = stack.a2l
	al3 = stack.a3l
	nx = 1+ceil(rcut / np.linalg.norm(al1))
	ny = 1+ceil(rcut / np.linalg.norm(al2))
	nz = 1+ceil(rcut / np.linalg.norm(al3))
	print("ns:")
	print([nx,ny,nz])
	factor = 1/(4*stack.permittivity)
	for atom,currentpos in enumerate(positions):
		for ir1 in range(-nx, nx):
			for ir2 in range(-ny, ny):
				for ir3 in range(-nz, nz):
					relpos = np.copy(positions)+ir1*al1+ir2*al2+ir3*al3
					distance = np.linalg.norm(relpos - currentpos, axis=1)
					for j,dist in enumerate(distance):
						if dist != 0:
							vs[atom] += factor*charges[j]*1.60217662e-19*1.0/(dist*1e-9)

	return vs

"""
def dipolecharge(a,material):
	if material == 'GaSe':
		return 0.3624+a*2.68
	elif material == 'InSe':
		return 0.1678+a*3.24623
	elif material == 'GaS':
		return -0.07383+a*4.0372
		"""