import numpy as np
from .materials import PTMC

#A material for fitting
class material(PTMC):
	#bond lenght
	d1=0
	d2=0
	d3=0
	d4=0

	def __init__(self,pars,a,c,z1,z2):
		self.charge = 0
		self.permittivity = 0
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