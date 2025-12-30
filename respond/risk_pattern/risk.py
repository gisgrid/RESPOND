import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F


def Gaussian_3d_torus_a(arc_len, par1, dla):
    """
    Calculates the parameter 'a' for the Gaussian 3D torus.

    Args:
        arc_len (float): Arc length.
        par1 (float): First parameter for the Gaussian function.
        dla (float): Look ahead distance.

    Returns:
        float: Calculated parameter 'a'.
    """


    if isinstance(arc_len, torch.Tensor):
        arc_len = arc_len.cpu().numpy()
    if isinstance(par1, torch.Tensor):
        par1 = par1.cpu().numpy()
    if isinstance(dla, torch.Tensor):
        dla = dla.cpu().numpy()


    par2 = dla
    # arc_len:(mesh_grid_len,) -> (frame_count,mesh_grid_len)
    # par2: (1,) -> (frame_count,1)
    a_par = par1 * (arc_len - par2) ** 2
    # if arc_len > dla --> 0
    a_par_sign1 = (np.sign(dla - arc_len) + 1) / 2
    # if value is negative
    a_par_sign2 = (np.sign(a_par) + 1) / 2
    # if arc_len is negative
    a_par_sign3 = (np.sign(arc_len) + 1) / 2
    # final a
    a = a_par_sign1 * a_par_sign2 * a_par_sign3 * a_par
    return a

def Gaussian_3d_torus_arclen(x, y, xv, yv, delta, xc, yc, R):
    """
    Calculates the arc length for the Gaussian 3D torus.

    Args:
        x, y (float): Coordinates of the point on the curve.
        xv, yv (float): Current vehicle position.
        delta (float): Steering angle.
        xc, yc (float): Center of the vehicle's turning circle.
        R (float): Turning radius.

    Returns:
        float: Calculated arc length.
    """


    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(xv, torch.Tensor):
        xv = xv.cpu().numpy()
    if isinstance(yv, torch.Tensor):
        yv = yv.cpu().numpy()
    if isinstance(delta, torch.Tensor): 
        delta = delta.cpu().numpy()
    if isinstance(xc, torch.Tensor):
        xc = xc.cpu().numpy()
    if isinstance(yc, torch.Tensor):
        yc = yc.cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()

    mag_u = np.sqrt((xv - xc) ** 2 + (yv - yc) ** 2)
    mag_v = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    dot_pro = (xv - xc) * (x - xc) + (yv - yc) * (y - yc)
    costheta = dot_pro / (mag_u * mag_v)

    costheta_clipped = np.clip(costheta, -1, 1)
    theta_abs = np.arccos(costheta_clipped)  # will always be positive
    sign_theta = np.sign((xv - xc) * (y - yc) - (x - xc) * (yv - yc))
    theta_pos_neg = np.sign(delta) * sign_theta * theta_abs
    theta = np.remainder(2 * np.pi + theta_pos_neg, 2 * np.pi)
    arc_len = R * theta
    return arc_len


def Gaussian_3d_torus_delta(delta_a):
    """
    Processes the delta value for the Gaussian 3D torus.

    Args:
        delta_a (float): Input delta value.

    Returns:
        float: Processed delta value.
    """
    if abs(delta_a) < 1e-8:
        delta = 1e-8
    else:
        delta = delta_a
    return delta

def Gaussian_3d_torus_dla(tla, V):

    if isinstance(tla, torch.Tensor):
        tla = tla.cpu().numpy()
    if isinstance(V, torch.Tensor):
        V = V.cpu().numpy()


    dla = tla * V
    if dla < 1:
        dla = 1
    return dla
def Gaussian_3d_torus_mexp(kexp, mcexp, delta, v=0, delta1=0, dt=0):
    """
    Calculates the mexp value for the Gaussian 3D torus.

    Args:
        kexp (float): Exponential factor.
        mcexp (float): Base mexp value.
        delta (float): Steering angle.
        v (float): Vehicle speed.
        delta1 (float): Previous steering angle (not used in this implementation).
        dt (float): Time step.

    Returns:
        float: Calculated mexp value.
    """
    mexp = mcexp + kexp * abs(delta)
    return mexp

def Gaussian_3d_torus_phiv(phiv_a):
    """
    Calculates the phiv value for the Gaussian 3D torus.

    Args:
        phiv_a (float): Input phiv value.

    Returns:
        float: Processed phiv value.
    """
    pi2temp = np.ceil(np.abs(phiv_a / (2 * np.pi)))  # how many rotations (e.g. 6*pi/2*pi = 3)
    phiv = np.abs(np.remainder(2 * np.pi * pi2temp + phiv_a, 2 * np.pi))  # phiv in terms of 0->2*pi radians
    return phiv

def Gaussian_3d_torus_R(L, delta):
    """
    Calculates the turning radius for the Gaussian 3D torus.

    Args:
        L (float): Wheel base of the car.
        delta (float): Steering angle.

    Returns:
        float: Calculated turning radius.
    """
    R = np.abs(L / np.tan(delta))
    return R

def Gaussian_3d_torus_sigma(arc_len, prb1, prb2):
    """
    Calculates the sigma value for the Gaussian 3D torus.

    Args:
        arc_len (float): Arc length.
        prb1 (float): First parameter for the Gaussian function.
        prb2 (float): Second parameter for the Gaussian function.

    Returns:
        float: Calculated sigma value.
    """

    if isinstance(arc_len, torch.Tensor):
        arc_len = arc_len.cpu().numpy()
    if isinstance(prb1, torch.Tensor):
        prb1 = prb1.cpu().numpy()
    if isinstance(prb2, torch.Tensor):
        prb2 = prb2.cpu().numpy()

    sigma = prb1 * arc_len + prb2
    return sigma

def Gaussian_3d_torus_xcyc(xv, yv, phiv, delta, R):
    """
    Calculates the center coordinates for the Gaussian 3D torus.

    Args:
        xv, yv (float): Current vehicle positions.
        phiv (float): Vehicle orientation.
        delta (float): Steering angle.
        R (float): Turning radius.

    Returns:
        tuple: Center coordinates (xc, yc).
    """
    if delta > 0:
        phil = phiv + np.pi / 2
    else:
        phil = phiv - np.pi / 2
    xc = R * np.cos(phil) + xv
    yc = R * np.sin(phil) + yv
    return xc, yc

def Gaussian_3d_torus_z(x, y, xc, yc, R, a, sigma1, sigma2):
    """
    Calculates the z value for the Gaussian 3D torus.

    Args:
        x, y (float): Coordinates of the point under consideration.
        xc, yc (float): Center coordinates of the circle.
        R (float): Turning radius.
        a (float): Height of the Gaussian.
        sigma1, sigma2 (float): Widths of the Gaussian.

    Returns:
        float: Calculated z value.
    """


    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(xc, torch.Tensor):
        xc = xc.cpu().numpy()
    if isinstance(yc, torch.Tensor):
        yc = yc.cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(sigma1, torch.Tensor):
        sigma1 = sigma1.cpu().numpy()
    if isinstance(sigma2, torch.Tensor):
        sigma2 = sigma2.cpu().numpy()

    dist_R = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    dist_R_R = dist_R - R
    dist_R_R_sign = np.sign(dist_R_R)
    a_inside = (1 - dist_R_R_sign) / 2
    a_outside = (1 + dist_R_R_sign) / 2

    num = -(dist_R_R ** 2)

    den1 = 2 * sigma1 ** 2
    zpure1 = a * a_inside * np.exp(num / den1)

    den2 = 2 * sigma2 ** 2
    zpure2 = a * a_outside * np.exp(num / den2)

    zpure = zpure1 + zpure2
    return zpure


def Gaussian_3d_torus_meshgrid(xv, yv, dla, res, Car_Nrp_Idx):
    """
    Generates a meshgrid for the Gaussian 3D torus.

    Args:
        xv, yv (float): Current vehicle positions.
        dla (float): Look ahead distance.
        res (float): Resolution of the grid.
        Car_Nrp_Idx (int): Current nearest road point index (not used in this implementation).

    Returns:
        tuple: Meshgrid arrays (X, Y) and boundary limits (xbl, xbu, ybl, ybu).
    """
    # --- START : very safe way ---
    n = 2
    xbl = xv - n * dla
    xbu = xv + n * dla
    ybl = yv - n * dla
    ybu = yv + n * dla
    x = np.arange(xbl, xbu + res, res)
    y = np.arange(ybl, ybu + res, res)
    X, Y = np.meshgrid(x, y)
    # --- END : very safe way ---

    # --- START : for my circuit specifically ---
    # Car_Nrp_Idx (not used in this implementation)

    return X, Y, xbl, xbu, ybl, ybu


def field_distribution(x=0, y=0, speed=10, heading_angle=0, turning_angle=0.1, vehicle_length=5, common_grid=None):
    v = speed  # vehicle speed
    xv = x  # vehicle position
    yv = y
    L = vehicle_length
    phiv = heading_angle

    Sr = 54
    res = 1  # meshgrid resolution
    tla = 2.75
    par1 = 2 * 0.0064
    kexp1 = 1 * 0.5
    kexp2 = 5 * 0.5
    mcexp = 0.26
    cexp = 2.55

    delta_fut_h = (np.pi / 180) * turning_angle / Sr
    phiv_a = (np.pi / 180) * phiv

    delta = Gaussian_3d_torus_delta(delta_fut_h)
    phiv = Gaussian_3d_torus_phiv(phiv_a)
    dla = Gaussian_3d_torus_dla(tla, v)
    R = Gaussian_3d_torus_R(L, delta)
    xc, yc = Gaussian_3d_torus_xcyc(xv, yv, phiv, delta, R)
    X, Y, xbl, xbu, ybl, ybu = Gaussian_3d_torus_meshgrid(xv, yv, dla, res, None)
    mexp1 = Gaussian_3d_torus_mexp(kexp1, mcexp, delta, v, None, None)
    mexp2 = Gaussian_3d_torus_mexp(kexp2, mcexp, delta, v, None, None)
    arc_len = Gaussian_3d_torus_arclen(X, Y, xv, yv, delta, xc, yc, R)
    a = Gaussian_3d_torus_a(arc_len, par1, dla)
    sigma1 = Gaussian_3d_torus_sigma(arc_len, mexp1, cexp)
    sigma2 = Gaussian_3d_torus_sigma(arc_len, mexp2, cexp)
    Z_cur = Gaussian_3d_torus_z(X, Y, xc, yc, R, a, sigma1, sigma2)
    qpr = np.sum(Z_cur)
    if common_grid is not None:
        X_common, Y_common = common_grid
        Z_cur_interp = griddata((X.flatten(), Y.flatten()), Z_cur.flatten(), (X_common, Y_common), method='linear',
                                fill_value=0)
        return Z_cur_interp

    # return X, Y, Z_cur
    return qpr

def field_distribution2D(x=0, y=0, speed=10, heading_angle=0, turning_angle=0.1, vehicle_length=5, common_grid=None):
    v = speed  # vehicle speed
    xv = x  # vehicle position
    yv = y
    L = vehicle_length
    phiv = heading_angle

    Sr = 54
    res = 1  # meshgrid resolution
    tla = 2.75
    par1 = 2 * 0.0064
    kexp1 = 1 * 0.5
    kexp2 = 5 * 0.5
    mcexp = 0.26
    cexp = 2.55

    delta_fut_h = (np.pi / 180) * turning_angle / Sr
    phiv_a = (np.pi / 180) * phiv

    delta = Gaussian_3d_torus_delta(delta_fut_h)
    phiv = Gaussian_3d_torus_phiv(phiv_a)
    dla = Gaussian_3d_torus_dla(tla, v)
    R = Gaussian_3d_torus_R(L, delta)
    xc, yc = Gaussian_3d_torus_xcyc(xv, yv, phiv, delta, R)
    X, Y, xbl, xbu, ybl, ybu = Gaussian_3d_torus_meshgrid(xv, yv, dla, res, None)
    mexp1 = Gaussian_3d_torus_mexp(kexp1, mcexp, delta, v, None, None)
    mexp2 = Gaussian_3d_torus_mexp(kexp2, mcexp, delta, v, None, None)
    arc_len = Gaussian_3d_torus_arclen(X, Y, xv, yv, delta, xc, yc, R)
    a = Gaussian_3d_torus_a(arc_len, par1, dla)
    sigma1 = Gaussian_3d_torus_sigma(arc_len, mexp1, cexp)
    sigma2 = Gaussian_3d_torus_sigma(arc_len, mexp2, cexp)
    Z_cur = Gaussian_3d_torus_z(X, Y, xc, yc, R, a, sigma1, sigma2)
    qpr = np.sum(Z_cur)
    if common_grid is not None:
        X_common, Y_common = common_grid
        Z_cur_interp = griddata((X.flatten(), Y.flatten()), Z_cur.flatten(), (X_common, Y_common), method='linear',
                                fill_value=0)
        return Z_cur_interp

    # return X, Y, Z_cur
    return Z_cur

def field_distributionNew(x=0, y=0, speed=10, heading_angle=0, turning_angle=0.1, vehicle_length=5, common_grid=None):
    v = speed  # vehicle speed
    xv = x  # vehicle position
    yv = y
    L = vehicle_length
    phiv = heading_angle

    Sr = 54
    res = 1  # meshgrid resolution
    # tla = 2.75
    tla = 4.0   # updated by Respond
    par1 = 2 * 0.0064
    kexp1 = 1 * 0.5
    kexp2 = 5 * 0.5
    mcexp = 0.07 # updated by Respond
    # mcexp = 0.26
    # cexp = 2.55
    cexp = 1.0  # updated by Respond


    delta_fut_h = (np.pi / 180) * turning_angle / Sr
    phiv_a = (np.pi / 180) * phiv

    delta = Gaussian_3d_torus_delta(delta_fut_h)
    phiv = Gaussian_3d_torus_phiv(phiv_a)
    dla = Gaussian_3d_torus_dla(tla, v)
    R = Gaussian_3d_torus_R(L, delta)
    xc, yc = Gaussian_3d_torus_xcyc(xv, yv, phiv, delta, R)
    X, Y, xbl, xbu, ybl, ybu = Gaussian_3d_torus_meshgrid(xv, yv, dla, res, None)
    mexp1 = Gaussian_3d_torus_mexp(kexp1, mcexp, delta, v, None, None)
    mexp2 = Gaussian_3d_torus_mexp(kexp2, mcexp, delta, v, None, None)
    arc_len = Gaussian_3d_torus_arclen(X, Y, xv, yv, delta, xc, yc, R)
    a = Gaussian_3d_torus_a(arc_len, par1, dla)
    sigma1 = Gaussian_3d_torus_sigma(arc_len, mexp1, cexp)
    sigma2 = Gaussian_3d_torus_sigma(arc_len, mexp2, cexp)
    Z_cur = Gaussian_3d_torus_z(X, Y, xc, yc, R, a, sigma1, sigma2)

    # Create a common grid for interpolation (150x600)
    x_min, x_max = xv - 300, xv + 300  # Centered around ego car
    y_min, y_max = yv - 75, yv + 75
    X_common, Y_common = create_common_grid(x_min, x_max, y_min, y_max, resolution=1)

    # Interpolate Z_cur onto the common grid
    Z_cur_interp = griddata((X.flatten(), Y.flatten()), Z_cur.flatten(), (X_common, Y_common), method='linear', fill_value=0)

    return X_common, Y_common, Z_cur_interp

def create_common_grid(x_min, x_max, y_min, y_max, resolution=0.5):
    X_common, Y_common = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    return X_common, Y_common



def field_distribution_Optimized(
    x=0, y=0, speed=10, heading_angle=0, turning_angle=0.1,
    vehicle_length=5, common_grid=None, device=None
):

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    v = speed
    xv, yv = x, y
    L = vehicle_length
    phiv = heading_angle

    Sr, res = 54, 1
    tla, par1 = 4.0, 2 * 0.0064
    kexp1, kexp2 = 0.5, 2.5
    mcexp, cexp = 0.07, 1.0

    delta_fut_h = (np.pi / 180) * turning_angle / Sr
    phiv_a = (np.pi / 180) * phiv

    delta = Gaussian_3d_torus_delta(delta_fut_h)
    phiv = Gaussian_3d_torus_phiv(phiv_a)
    dla  = Gaussian_3d_torus_dla(tla, v)
    R    = Gaussian_3d_torus_R(L, delta)
    xc, yc = Gaussian_3d_torus_xcyc(xv, yv, phiv, delta, R)

    X, Y, *_ = Gaussian_3d_torus_meshgrid(xv, yv, dla, res, None)
    mexp1 = Gaussian_3d_torus_mexp(kexp1, mcexp, delta, v)
    mexp2 = Gaussian_3d_torus_mexp(kexp2, mcexp, delta, v)
    arc_len = Gaussian_3d_torus_arclen(X, Y, xv, yv, delta, xc, yc, R)
    a       = Gaussian_3d_torus_a(arc_len, par1, dla)
    sigma1  = Gaussian_3d_torus_sigma(arc_len, mexp1, cexp)
    sigma2  = Gaussian_3d_torus_sigma(arc_len, mexp2, cexp)
    Z_cur   = Gaussian_3d_torus_z(X, Y, xc, yc, R, a, sigma1, sigma2)  # numpy (H1,W1)


    if common_grid is None:
        x_min, x_max = xv - 300, xv + 300   # 600 m
        y_min, y_max = yv -  75, yv +  75   # 150 m
        X_common, Y_common = np.meshgrid(
            np.arange(x_min, x_max, 1.0),
            np.arange(y_min, y_max, 1.0)
        )                                   # → shape (150,600)
    else:
        X_common, Y_common = common_grid

    H, W   = X_common.shape
    H1, W1 = Z_cur.shape


    src = torch.as_tensor(Z_cur, dtype=torch.float32, device=device)\
              .unsqueeze(0).unsqueeze(0)      # [1,1,H1,W1]


    x_min1, x_max1 = X.min(), X.max()
    y_min1, y_max1 = Y.min(), Y.max()

    grid_x = 2.0 * (torch.as_tensor(X_common, dtype=torch.float32, device=device) - x_min1) / (x_max1 - x_min1) - 1.0
    grid_y = 2.0 * (torch.as_tensor(Y_common, dtype=torch.float32, device=device) - y_min1) / (y_max1 - y_min1) - 1.0
    grid   = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1,H,W,2]


    tgt = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    Z_cur_interp = tgt.squeeze().cpu().numpy()   

    return X_common, Y_common, Z_cur_interp
