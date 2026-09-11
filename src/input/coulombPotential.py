import math
def CoulombPotential(params,x1,y1,z1,x2,y2,z2):
     d = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
     dsoft = d**8 * params["sigmaFit"][0] + d**7 * params["sigmaFit"][1] + d**6 * params["sigmaFit"][2] + d**5 * params["sigmaFit"][3] + d**4 * params["sigmaFit"][4] + d**3 * params["sigmaFit"][5] + d**2 * params["sigmaFit"][6] + d * params["sigmaFit"][7] + params["sigmaFit"][8]
     VVal = (18.8950673 / params["RelPerm"]) * dsoft**(-1) #//use Coulomb softening with cap at 100, 18.89... is conversion factor to natural units and nm
     return VVal