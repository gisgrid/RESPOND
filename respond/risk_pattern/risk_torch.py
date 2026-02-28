import torch
from torch import Tensor as T

def cal_speed(vx:T, vy:T):
    speed = torch.sqrt(vx ** 2 + vy ** 2)
    return speed

def cal_phiv_a(heading:T):
    phiv_a = (torch.pi / 180) * heading
    return phiv_a

def Gaussian_3d_torus_a(arc_len:T, par1:T, dla:T):
    """
    Calculates the parameter 'a' for the Gaussian 3D torus.

    Args:
        arc_len and par1 must be immutable
        arc_len:[s,k] (torch.Tensor): Arc length.
        par1:scalar (torch.Tensor): First parameter for the Gaussian function.
        dla:[s,] (torch.Tensor): Look ahead distance.

    Returns:
        torch.Tensor: Calculated parameter 'a'.
    """
    ext_shape = [-1]
    for _ in range(0,len(arc_len.shape)-1):
        ext_shape.append(1)
    # arc_len: (frame_count,mesh_grid_len)
    # par2=dla: (1,) -> (frame_count,1)
    dla = dla.reshape(ext_shape)
    
    # allocate tmp0
    a_par = arc_len - dla  # arc_len - dla
    a_par.square_()  # (arc_len - dla) ** 2
    a_par.mul_(par1)
    # a_par = par1 * (arc_len - dla) ** 2

    # allocate tmp1
    a_par_sign1 = (dla - arc_len).sign_().add_(1).div_(2)
    # a_par_sign1 = (torch.sign(dla - arc_len) + 1) / 2
    
    # reuse tmp1 as final result
    a = a_par_sign1.mul_(a_par)
    # a_par will use once in the following

    # allocate tmp2 and delete a_par -> fuse to reuse a_par
    a_par_sign2 = torch.sign(a_par, out=a_par).add_(1).div_(2)
    # a_par_sign2 = (torch.sign(a_par) + 1) / 2
    
    a = a_par_sign1.mul_(a_par_sign2)

    # reuse a_par_sign2
    a_par_sign3 = torch.sign(arc_len,out=a_par_sign2).add_(1).div_(2)
    # a_par_sign3 = (torch.sign(arc_len) + 1) / 2

    # final a
    a = a.mul_(a_par_sign3)
    # a = a_par_sign1 * a_par_sign2 * a_par_sign3 * a_par
    return a

def Gaussian_3d_torus_arclen(x:T, y:T, xv:T, yv:T, delta:T, xc:T, yc:T, R:T):
    """
    Calculates the arc length for the Gaussian 3D torus.

    Args:
        all input must be immutable!
        x, y (torch.Tensor): Coordinates of the point on the curve.
        xv, yv (torch.Tensor): Current vehicle position.
        delta (torch.Tensor): Steering angle.
        xc, yc (torch.Tensor): Center of the vehicle's turning circle.
        R (torch.Tensor): Turning radius.

    Returns:
        torch.Tensor: Calculated arc length.
    """
    ext_shape = [-1]
    for _ in range(0,len(x.shape)):
        ext_shape.append(1)
    xv = xv.reshape(ext_shape)
    yv = yv.reshape(ext_shape)
    xc = xc.reshape(ext_shape)
    yc = yc.reshape(ext_shape)
    
    # xv_xc = xv.sub_(xc)             # (s,1,1)
    xv_xc = xv - xc
    # yv_yc = yv.sub(yc)              # (s,1,1)
    yv_yc = yv - yc
    x_xc = (x - xc)                 # alloc (s,n,m)
    y_yc = (y - yc)                 # alloc (s,n,m)
    
    xv_xc_2 = xv_xc.square()
    # xv_xc_2 = (xv - xc) ** 2
    yv_yc_2 = yv_yc.square()
    # xv_xc_2 = (yv - yc) ** 2
    mag_u = xv_xc_2.add_(yv_yc_2).sqrt_() # del yv_yc_2 # (s,1,1)
    # mag_u = sqrt( (xv - xc) ** 2 + (yv - yc) ** 2 )
    
    b = y_yc.square()          # alloc (s,n,m)
    # b = (y - yc) ** 2
    mag_v = x_xc.square().add_(b).sqrt_()   # alloc (s,n,m)
    # mag_v = sqrt( (x - xc) ** 2 + (y - yc) ** 2 )
    mag_v_u = mag_v.mul_(mag_u.broadcast_to(mag_v.shape)) # del b # del mag_u # (s,1,1)
    # mag_v_u = mag_u * mag_v
    
    b.copy_(x_xc)                           # reuse b, by merging del:(s,n,m),alloc:(s,n,m) pair
    t = y_yc * yv_yc                        # alloc (s,n,m)
    dot_pro = b.mul_(xv_xc.broadcast_to(x_xc.shape)).add_(t) # del t # (s,n,m)
    # dot_pro = (x -xc) * (xv - xc) + (y - yc) * (yv - yc)
    costheta = dot_pro.divide_(mag_v_u) # del mag_v_u # (s,n,m)
    # costheta = dot_pro / (mag_u * mag_v)
    x_xc.mul_(yv_yc.broadcast_to(x_xc.shape))
    # x_xc = (x - xc) * (yv - yc)
    sign_theta = y_yc.mul_(xv_xc.broadcast_to(y_yc.shape)).sub_(x_xc).sign_() # del x_xc
    # sign_theta = sign( (y - yc)*(xv - xc) - (x - xc)*(yv - yc) )
    
    # 确保 costheta 在 [-1, 1] 范围内
    costheta_clipped = costheta.clip_(-1,1)
    theta_abs = costheta_clipped.acos_()  # will always be positive
    theta_pos_neg = theta_abs.mul_(torch.sign(delta).broadcast_to(theta_abs.shape)).mul_(sign_theta) # del sign_theta
    # theta_pos_neg = theta_abs * torch.sign(delta) * sign_theta
    two_pi = 2 * torch.pi
    theta = theta_pos_neg.add_(two_pi).remainder_(two_pi)
    # theta = remainder(theta_pos_neg + 2pi, 2pi)
    arc_len = theta.mul_(R)
    return arc_len

def Gaussian_3d_torus_delta(delta_a):
    """
    Processes the delta value for the Gaussian 3D torus.

    Args:
        delta_a (torch.Tensor): Input delta value.

    Returns:
        torch.Tensor: Processed delta value.
    """
    if torch.abs(delta_a) < 1e-8:
        delta = torch.tensor(1e-8, dtype=delta_a.dtype, device=delta_a.device)
    else:
        delta = delta_a
    return delta

def Gaussian_3d_torus_dla(tla, V:T):
    """
    Processes the dla value for the Gaussian 3D torus.

    Args:
        tla (torch.Tensor): Look ahead time.
        V (torch.Tensor): Vehicle speed.

    Returns:
        torch.Tensor: Processed dla value.
    """
    dla = tla * V
    dla = torch.where(dla < 1, 1, dla)
    return dla

def Gaussian_3d_torus_mexp(kexp, mcexp, delta):
    """
    Calculates the mexp value for the Gaussian 3D torus.

    Args:
        kexp (torch.Tensor): Exponential factor.
        mcexp (torch.Tensor): Base mexp value.
        delta (torch.Tensor): Steering angle.
        v (torch.Tensor): Vehicle speed.
        delta1 (torch.Tensor): Previous steering angle (not used in this implementation).
        dt (torch.Tensor): Time step.

    Returns:
        torch.Tensor: Calculated mexp value.
    """
    mexp = mcexp + kexp * torch.abs(delta)
    return mexp

def Gaussian_3d_torus_phiv(phiv_a: torch.Tensor):
    """
    Calculates the phiv value for the Gaussian 3D torus.

    Args:
        phiv_a (torch.Tensor): Input phiv value.

    Returns:
        torch.Tensor: Processed phiv value.
    """

    pi2temp = torch.abs(phiv_a)  # |phiv_a|
    pi2temp.div_(2 * torch.pi)  # |phiv_a| / (2 * pi)
    pi2temp.ceil_()  # ceil(|phiv_a| / (2 * pi))
    # pi2temp = torch.ceil(torch.abs(phiv_a / (2 * torch.pi)))  # how many rotations (e.g. 6*pi/2*pi = 3)

    # 计算 remainder
    phiv = pi2temp.mul_(2 * torch.pi)  # 2 * pi * pi2temp
    phiv.add_(phiv_a)  # 2 * pi * pi2temp + phiv_a
    # phiv.remainder_(phiv, 2 * torch.pi)  # remainder(2 * pi * pi2temp + phiv_a, 2 * pi)
    phiv = torch.remainder(phiv, 2 * torch.pi, out=phiv)
    # 取绝对值
    phiv.abs_()  # |remainder(...)|
    # phiv = torch.abs(torch.remainder(2 * torch.pi * pi2temp + phiv_a, 2 * torch.pi))  # phiv in terms of 0->2*pi radians

    return phiv

def Gaussian_3d_torus_R(L, delta):
    """
    Calculates the turning radius for the Gaussian 3D torus.

    Args:
        L (torch.Tensor): Wheel base of the car.
        delta (torch.Tensor): Steering angle.

    Returns:
        torch.Tensor: Calculated turning radius.
    """
    R = torch.abs(L / torch.tan(delta))
    return R

def Gaussian_3d_torus_sigma(arc_len:T, prb1, prb2):
    """
    Calculates the sigma value for the Gaussian 3D torus.

    Args:
        arc_len (torch.Tensor): Arc length.
        prb1 (torch.Tensor): First parameter for the Gaussian function.
        prb2 (torch.Tensor): Second parameter for the Gaussian function.

    Returns:
        torch.Tensor: Calculated sigma value.
    """
    sigma = (arc_len.mul(prb1)).add_(prb2)
    return sigma

def Gaussian_3d_torus_xcyc(xv:T, yv:T, phiv:T, delta:T, R:T):
    """
    Calculates the center coordinates for the Gaussian 3D torus.

    Args:
        xv, yv (torch.Tensor): Current vehicle positions.
        phiv (torch.Tensor): Vehicle orientation.
        delta (torch.Tensor): Steering angle.
        R (torch.Tensor): Turning radius.

    Returns:
        tuple: Center coordinates (xc, yc).
    """

    if delta > 0:
        phil = phiv + torch.pi / 2
    else:
        phil = phiv - torch.pi / 2

    # 计算 xc 和 yc
    xc = phil.cos()  # cos(phil)，不能重用 phil 的内存
    xc.mul_(R).add_(xv)  # xc = R * cos(phil) + xv
    # xc = R * torch.cos(phil) + xv

    yc = phil.sin_()  # sin(phil),重用 phil 的内存
    yc.mul_(R).add_(yv)  # yc = R * sin(phil) + yv
    # yc = R * torch.sin(phil) + yv

    return xc, yc

def Gaussian_3d_torus_z(x:T, y:T, xc:T, yc:T, R, a:T, sigma1:T, sigma2:T):
    """
    Calculates the z value for the Gaussian 3D torus.

    Args:
        x,y,xc,yc,R must be immutable
        a, sigma1,sigma2 can be modify
        x, y (torch.Tensor): Coordinates of the point under consideration.
        xc, yc (torch.Tensor): Center coordinates of the circle.
        R (torch.Tensor): Turning radius.
        a (torch.Tensor): Height of the Gaussian.
        sigma1, sigma2 (torch.Tensor): Widths of the Gaussian.

    Returns:
        float: Calculated z value.
    """
    # x, y:     (mesh_grid_len,)
    # xc, yc:   (frame_count,)
    # a, sigma: (frame_count,mesh_grid_len)
    ext_shape = [-1]
    for _ in range(0,len(x.shape)):
        ext_shape.append(1)
    xc = xc.reshape(ext_shape)
    yc = yc.reshape(ext_shape)
    # xc,yc: (frame_count,1)

    # big tmp0:(frame_count,mesh_grid_len)
    y_yc_square = (y - yc).square_()
    dist_R_R = (x - xc).square_().add_(y_yc_square).sqrt_().sub_(R)

    dist_R_R_sign = torch.sign(dist_R_R)
    # a_inside = (1 - dist_R_R_sign).div_(2)
    # a_outside = (1 + dist_R_R_sign).div_(2)
    # (a_inside+ a_outside) = 1

    # reuse dist_R_R
    num = dist_R_R.square_().neg_()

    # den1 = 2 * sigma1 ** 2
    # zpure1 = a * a_inside * np.exp(num / den1)

    # den2 = 2 * sigma2 ** 2
    # zpure2 = a * a_outside * np.exp(num / den2)


    # reuse sigma1
    den1 = sigma1.square_().mul_(2)
    # reuse den1: exp(num/den1)
    num_den1 = torch.div(num,den1,out=den1).exp_()
    # reuse sigma1
    den2 = sigma2.square_().mul_(2)
    # reuse den2: exp(num/den2)
    num_den2 = torch.div(num,den2,out=den2).exp_()
    
    # reuse num: exp(num/den2) - exp(num/den1)
    num_den1_num_den2 = torch.subtract(num_den1, num_den2, out= num)
    # reuse dist_R_R_sign: dist_R_R_sign / 2 * (exp(num/den2) - exp(num/den1))
    right_part = dist_R_R_sign.div_(2).mul_(num_den1_num_den2)
    # reuse right_part: exp(num/den1) + exp(num/den2) + dist_R_R_sign / 2 * (exp(num/den2) - exp(num/den1))
    whole_part = right_part.add_(num_den1).add_(num_den2)

    # zpure = zpure1 + zpure2
    # = a * np.exp(num / den2) + a * np.exp(num / den2) * dist_R_R_sign / 2
    # + a * np.exp(num / den1) - a * np.exp(num / den1) * dist_R_R_sign / 2
    # = a * (exp(num/den1) + exp(num/den2))
    # + a * dist_R_R_sign / 2 * (exp(num/den2) - exp(num/den1))
    # = a * ( exp(num/den1) + exp(num/den2)
    #        + dist_R_R_sign / 2 * (exp(num/den2) - exp(num/den1)) )
    zpure = whole_part.mul_(a)
    return zpure

def Gaussian_3d_torus_z_simplify(x:T, y:T, xc:T, yc:T, R, a:T, sigma:T):
    """
    Calculates the z value for the Gaussian 3D torus.

    Args:
        x,y,xc,yc,R must be immutable
        a, sigma can be modify
        x, y (torch.Tensor): Coordinates of the point under consideration.
        xc, yc (torch.Tensor): Center coordinates of the circle.
        R (torch.Tensor): Turning radius.
        a (torch.Tensor): Height of the Gaussian.
        sigma = sigma1 = sigma2 (torch.Tensor): Widths of the Gaussian.

    Returns:
        torch.Tensor: Calculated z value.
    """

    # x, y:     (mesh_grid_len,)
    # xc, yc:   (frame_count,)
    # a, sigma: (frame_count,mesh_grid_len)
    ext_shape = [-1]
    for _ in range(0,len(x.shape)):
        ext_shape.append(1)
    xc = xc.reshape(ext_shape)
    yc = yc.reshape(ext_shape)
    # xc,yc: (frame_count,1)

    # big tmp0:(frame_count,mesh_grid_len)
    y_yc_square = (y - yc).square_()
    dist_R_R = (x - xc).square_().add_(y_yc_square).sqrt_().sub_(R)
    # dist_R_R_sign = torch.sign(dist_R_R)
    # a_inside = (1 - dist_R_R_sign).div_(2)
    # a_outside = (1 + dist_R_R_sign).div_(2)
    # (a_inside+ a_outside) = 1

    # reuse dist_R_R
    num = dist_R_R.square_().neg_()
    # reuse sigma
    den = sigma.square_().mul_(2)
    # reuse num
    a_exp_num_den = num.div_(den).exp_().mul_(a)
    # zpure = a_exp_num_den *(a_inside+ a_outside)
    zpure = a_exp_num_den
    return zpure

# def Gaussian_3d_torus_meshgrid(xv, yv, dla, res, Car_Nrp_Idx):
#     """
#     Generates a meshgrid for the Gaussian 3D torus.

#     Args:
#         xv, yv (torch.Tensor): Current vehicle positions.
#         dla (torch.Tensor): Look ahead distance.
#         res (torch.Tensor): Resolution of the grid.
#         Car_Nrp_Idx (int): Current nearest road point index (not used in this implementation).

#     Returns:
#         tuple: Meshgrid arrays (X, Y) and boundary limits (xbl, xbu, ybl, ybu).
#     """
#     # --- START : very safe way ---
#     n = 2
#     xbl = xv - n * dla
#     xbu = xv + n * dla
#     ybl = yv - n * dla
#     ybu = yv + n * dla
#     x = torch.arange(xbl, xbu + res, res)
#     y = torch.arange(ybl, ybu + res, res)
#     X, Y = torch.meshgrid(x, y)
#     # --- END : very safe way ---

#     # --- START : for my circuit specifically ---
#     # Car_Nrp_Idx (not used in this implementation)

#     return X, Y, xbl, xbu, ybl, ybu