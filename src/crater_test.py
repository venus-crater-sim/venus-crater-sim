from venus import VENUS_RADIUS

import math

import numpy as np

def crater_contour(x, y, z, r, num_points=20, debug=False):
    if debug:
        print(f"Event center: {[x, y, z]}")
        print(f"Event radius: {r}")

    alpha = r / VENUS_RADIUS

    points = np.array([[VENUS_RADIUS, longitude, alpha]
                       for longitude in np.linspace(0, 2 * math.pi, num_points)])

    points_rect = np.array([[VENUS_RADIUS * math.cos(longitude) * math.sin(alpha),
                             VENUS_RADIUS * math.sin(longitude) * math.sin(alpha),
                             VENUS_RADIUS * math.cos(alpha)]
                            for [rho, longitude, alpha] in points])

    if debug:
        print(points_rect)

    longitude = theta = np.arctan2(y, x)
    latitude = np.arcsin(z / VENUS_RADIUS)
    colatitude = phi = math.pi / 2 - latitude

    if debug:
        print(theta, phi)

    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)

    rotation_matrix = np.array([[ct, -st * cp, st * sp],
                                [st, ct * cp,  -ct * sp],
                                [0,  sp,       cp]])

    if debug:
        print(rotation_matrix)

    rim_rect = np.transpose(np.matmul(rotation_matrix, np.transpose(points_rect)))
    rim_lon = np.arctan2(rim_rect[:, 1], rim_rect[:, 0])
    rim_lat = np.arcsin(rim_rect[:, 2] / VENUS_RADIUS)

    if debug:
        print(np.column_stack([rim_lon, rim_lat]))

    return np.column_stack([rim_lon, rim_lat])


# crater_contour(255, 926, 5975, 3827, debug=True)
crater_contour(206, -25, 6048, 1012, debug=True)
