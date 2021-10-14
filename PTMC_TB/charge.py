import numpy as np
from scipy import integrate
import pybinding as pb
from math import sin, pi

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

def electricfield(pos,stack,excesscharges):
    area = (stack.a*1e-9)**2*sin(pi*30/180)
    permittivity = stack.permittivity
    field = np.zeros(len(pos))
    for i,value in enumerate(pos):
        for p,zprime in enumerate(stack.planes):
            field[i]+=-0.5*sign(value-zprime)*excesscharges[p]*1.60217662e-19/(permittivity*area)
    return field
