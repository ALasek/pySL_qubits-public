import math
def electrostaticPotential(params,p,x,y=0,z=0):
    Vy = 0.5 * params["Mass"][p] * params["w2Y"] * y**2 #harmonic pot in y
    Vp = 0.5 * params["Mass"][p] * params["w2X"] * x**2   # .. in x
    Vb = params["ATB1"] * params["Mass"][p] * math.exp(-params["s2TB1"]*(x**2) / 2) #gaussian pot
    Vb2 = params["ATB2"] * params["Mass"][p] * math.exp(-params["s2TB2"]*(x ** 2) / 2)
    VVal = Vy + Vp + Vb + Vb2
    return VVal
