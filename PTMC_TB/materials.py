
import numpy as np

class PTMC:
	pass

class InSe(PTMC):
	#lattice
	a = 0.40231918225
	c = 0.8469198513
	z1 = 3*0.060940054
	z2 = 3*0.111797495


	vbo = -0.10
	align = -0.389
	#on-site therms
	M_Es = -8.25051791+align+vbo
	X_Es = -21.7774272+align+vbo
	M_Ep = -5.78718728+align+vbo
	X_Ep = -13.2424346+align+vbo
	M_Ese = -0.00177680+align+vbo
	X_Ese = -7.82566784+align+vbo

	#bond lenght
	d1=0.26580159178743173
	d2=0.27881874425332515
	d3=0.40231918225
	d4=0.40231918224999996

	permittivity = 7.63 * 8.854187e-12


	#bond sp3s* parameters
	bondpar_1=np.array([-0.80047138,2.00104922,2.47287788,-0.63729041,-0.15896652,-0.28206764,0.15050463])
	bondpar_2=np.array([-2.81932754,2.42888575,5.65046613,1.40414532,-0.24829224,0.79917298,-0.86389002])
	bondpar_3=np.array([-0.61550504,-0.62086996,1.22027438,0.46091637,0.45802753,-0.25189653,-1.18996135])
	bondpar_4=np.array([-0.04216172,-0.30000933,0.47875544,-0.07658634, 0.82644271 ,0.47723517,0.18445480])
	bondpar_5=np.array([0.14318449,-1.02159242,0.92629092 ,-0.19926596,0.32488433,-0.19256169,0.32491191])

	def __init__(self, vbo = -0.10, charge = 1.473955):
		self.vbo = vbo
		self.charge = charge
		self.M_Es = -8.25051791+self.align+vbo
		self.X_Es = -21.7774272+self.align+vbo
		self.M_Ep = -5.78718728+self.align+vbo
		self.X_Ep = -13.2424346+self.align+vbo
		self.M_Ese = -0.00177680+self.align+vbo
		self.X_Ese = -7.82566784+self.align+vbo

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

	permittivity = 7.63 * 8.854187e-12
	
	#bond sp3s* parameters
	bondpar_1=np.array([-1.05908662,2.47549236 ,2.16608104,-0.66687273 , -0.11141026 ,-0.15410760,0.03118546 ])
	bondpar_2=np.array([-2.00684280 ,3.24371535,3.35629446,0.10853752,0.73472794,0.11758892,4.32088163])
	bondpar_3=np.array([-0.16813146,-0.10493689,0.70639068,0.26897119,0.23332979,0.14969084,-0.27095969])
	bondpar_4=np.array([-0.14626485,-0.55295508, 0.44265988,0.02210112,0.60012662 ,0.50896764,-0.03574015])
	bondpar_5=np.array([0.05793399,-0.47473129,  0.67716787 ,-0.20068697,0.16859448,-0.27186365,0.01796498])

	def __init__(self, vbo = 0, charge = 1.37109):
		self.vbo = vbo
		self.charge = charge

class GaS(PTMC):
	#lattice
	a = 0.35787065881
	c = 0.735930452
	z1 = 3*0.070283744
	z2 = 3*0.115642587
	
	#on-site therms
	align = -0.02
	vbo = -0.47
	M_Es = -8.27485297+align+vbo
	X_Es = -21.2216967+align +vbo
	M_Ep = -6.91032591+align +vbo
	X_Ep = -13.6205020+align+vbo
	M_Ese = -0.37864800+align +vbo
	X_Ese = -2.0840e-04+align+vbo
	
	#bond lenght
	d1=0.22960632000699166
	d2=0.23434374187696183
	d3=0.35787065881
	d4=0.35787065881

	permittivity = 7.63 * 8.854187e-12
	
	#bond sp3s* parameters
	bondpar_1=np.array([-1.02436390,2.63710397 ,2.72339356,-0.63204644 , 0.46394986 ,-0.56904087,-0.38369435 ])
	bondpar_2=np.array([-3.82367937 ,2.25737862,5.13232850,1.63559673,-1.09700106,2.59754309 ,-2.41098533])
	bondpar_3=np.array([-0.47643468,-0.53400382,1.10929477,0.52673442,-0.41914212,-0.32337050,-0.84513423])
	bondpar_4=np.array([-0.19278761,0.41032797, 0.50189261,-0.06950092,1.10037916 ,0.37129232,0.88703060])
	bondpar_5=np.array([0.08160404,-0.89388693,  0.78971076 ,-0.13090671,-0.66583978,-1.17377742,-3.91834164])

	def __init__(self, vbo = -0.47, charge = 1.3778):
		self.vbo = vbo
		self.charge = charge
		self.M_Es = -8.27485297+self.align+vbo
		self.X_Es = -21.2216967+self.align +vbo
		self.M_Ep = -6.91032591+self.align +vbo
		self.X_Ep = -13.6205020+self.align+vbo
		self.M_Ese = -0.37864800+self.align +vbo
		self.X_Ese = -2.0840e-04+self.align+vbo

class GaTe(PTMC):
	#lattice
	a = 0.40676785939
	c = 0.877720556
	z1 = 3*0.068052439
	z2 = 3*0.118158034
	
	#on-site therms
	align = -1.7
	vbo = 0.15
	M_Es = -0.29459962+align+vbo
	X_Es = -17.6615209+align +vbo
	M_Ep = -4.16384156+align +vbo
	X_Ep = -11.8576929+align+vbo
	M_Ese = -3.4284e-05+align +vbo
	X_Ese = -6.47597587+align+vbo
	
	#bond lenght
	d1=0.2693705751675323
	d2=0.25546214420991864
	d3=0.40676785939
	d4=0.40676785939

	permittivity = 7.63 * 8.854187e-12
	
	#bond sp3s* parameters
	bondpar_1=np.array([-0.93130667,2.22314914 ,2.61254626,-0.15331669 , -0.54954328 ,-0.98540657,0.05314677 ])
	bondpar_2=np.array([-5.54770615 ,3.99950805,5.69052071,0.86627316,-2.86642097,1.64593929 ,-1.58835950])
	bondpar_3=np.array([ 0.39061482,-1.48804478,1.14395731,0.37779626,-0.78345347,0.03654274,-1.33815481])
	bondpar_4=np.array([0.11543500,0.09023345, 0.80616394,-0.12486609,0.99023165 ,0.12371133,0.35250573])
	bondpar_5=np.array([-0.48549629,1.01168223,  0.79508838 ,-0.13346246,-1.31549488,-1.94071508,0.94326900])

	def __init__(self, vbo = 0.15, charge = 1.46861879):
		self.vbo = vbo
		self.charge = charge
		self.M_Es = -0.29459962+self.align+vbo
		self.X_Es = -17.6615209+self.align +vbo
		self.M_Ep = -4.16384156+self.align +vbo
		self.X_Ep = -11.8576929+self.align+vbo
		self.M_Ese = -3.4284e-05+self.align +vbo
		self.X_Ese = -6.47597587+self.align+vbo