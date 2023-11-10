from itertools import cycle
import numpy as np

def gauss_newton(model, data_loader, max_step=100000, threshold=1e-5):
    track_position = [model.get_position()]
    track_res = []
    for i, data in enumerate(cycle(data_loader)):
        x, y, z, target = data
        output = model.forward(x, y, z)
        residual = output - target
        track_res.append(residual)
        jacobian = model.jacobian(x, y, z)
        pk = np.linalg.lstsq(a=np.expand_dims(jacobian, axis=0), b=np.array([-residual]))[0]
        model.set_position(model.get_position() + 0.001 * pk)
        track_position.append(model.get_position())
        if i >= max_step:
            break
        
    return track_position, track_res