def get_input_attrib(name):
    from .gradient import gradient 
    from .inputgrad import input_gradient
    from .smoothgrad import smoothgrad
    from .integratedgrad import ig    
    
    if name =="grad":
        return gradient
    elif name == 'inputgrad':
        return input_gradient
    elif name == 'smoothgrad':
        return smoothgrad
    elif name == "ig":
        return ig
    
def get_input_attrib_energy(name):
    from .gradient import gradient_energy
    from .inputgrad import input_gradient
    from .smoothgrad import smoothgrad_energy
    from .integratedgrad import ig    
    
    if name =="grad":
        return gradient_energy
    elif name == 'inputgrad':
        return None
    elif name == 'smoothgrad':
        return smoothgrad_energy
    elif name == "ig":
        return None