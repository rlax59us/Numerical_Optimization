from itertools import cycle
import numpy as np


def levenberg_marquardt(model, data_loader, max_step=100000, threshold=1e-15):
    track_position = [model.get_position()]
    track_res = []
    lambda_k = 1
    lambda_min = 1e-5
    lambda_max = 10**5
    for i, data in enumerate(cycle(data_loader)):
        x, y, z, target = data
        output = model.forward(x, y, z)
        residual = output - target
        track_res.append(residual)
        jacobian = model.jacobian(x, y, z)

        a = np.dot(np.transpose(np.expand_dims(jacobian, axis=0)), np.expand_dims(jacobian, axis=0))
        a = a + lambda_k * np.ones(a.shape)
        b = np.dot(-np.transpose(np.expand_dims(jacobian, axis=0)), np.array([residual]))

        pk = np.linalg.lstsq(a=a, b=b)[0]
        model.set_position(model.get_position() + 0.001 * pk)
        track_position.append(model.get_position())
        if i >= max_step:
            break
        if i >= 1:
            if np.abs(track_res[-2]) > np.abs(track_res[-1]) and lambda_k > lambda_min: #decrease
                lambda_k = lambda_k / 10
            elif np.abs(track_res[-2]) < np.abs(track_res[-1]) and lambda_k < lambda_max:
                lambda_k = lambda_k * 10
    
    return track_position, track_res