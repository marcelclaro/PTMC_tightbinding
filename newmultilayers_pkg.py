import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, cos, sin
from scipy import integrate
import matplotlib.mlab as mlab
import matplotlib.gridspec as gs
import sys
from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit

from PTMC_TB.gamma import Stack
from PTMC_TB.materials import *
import PTMC_TB.fitting as fitting
import PTMC_TB.QE_util as QE

pb.pltutils.use_style()

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
	material = fitting.material(pars,a,c,z1,z2)
	mat = Stack([material,material,material],a,strained=False)
	lattice = mat.lat

	kpoints = QE.get_fullband_kpoints()
	points_index = QE.fullband_index
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


kpoints = QE.get_fullband_kpoints()
points_index = QE.fullband_index


mat = Stack([InSe(),InSe(),InSe()],0.4,strained=False)
lattice = mat.lat
kpoints[:,0] *= pi*2/mat.a
kpoints[:,1] *= pi*2/mat.a
kpoints[:,2] *= pi*2/mat.c
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

"""plt.subplot(132, title="Val.")
model.plot(axes="yx")

plt.subplot(133, title="Cond.")
model.plot(axes="yz")"""


kpm = pb.kpm(model,silent=True)
print("charges:")
for n in range(len(mat.materiallist)):
	ldos = kpm.calc_ldos(energy=np.linspace(-30, -10.7, 4000), broadening=0.05,
		position=[0, 0,0], sublattice="X2-"+str(n))
	chargeX=6-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable)
	ldos = kpm.calc_ldos(energy=np.linspace(-30, -10.7, 4000), broadening=0.05,
		position=[0, 0,0], sublattice="M2-"+str(n))
	chargeM=3-2*integrate.simps(np.nan_to_num(ldos.data),ldos.variable)
	print("layer "+str(n)+" -X1="+str(chargeX)+"  M1="+str(chargeM))
	

plt.show()	
	
