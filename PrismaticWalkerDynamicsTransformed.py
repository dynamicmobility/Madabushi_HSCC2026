import jax
import jax.numpy as np

g = 9.81
r0 = 0.5

def inv_phi(x):
    r, y, rd, yd = np.ravel(x)
    th = np.atan(y)
    thd = yd/(1+y**2)
    return np.array([r, th, rd, thd])

def phi(y):
    r, th, rd, thd = np.ravel(y)
    y = np.tan(th)
    yd = thd/(np.cos(th)**2)
    return np.array([r, y, rd, yd])


class PrismaticWalkerTransformed():
    nu = 1
    nx = 4

    @staticmethod
    def f(t, x, u):
        # r, th, rd, thd = x[0, :] # Hmmmm
        # r, th, rd, thd = np.ravel(x)

        r, y, rd, yd = np.ravel(x)
        th = np.atan(y)
        thd = yd/(1+y**2)

        rdd = -g*np.cos(th) + r*thd**2 + u[0]
        thdd = (g*np.sin(th) - 2*rd*thd)/r
        ydd = thdd*(1+y**2) + thd*(2*y*yd)
        return np.array ([
            rd,
            yd,
            rdd,
            ydd
        ])
    @staticmethod
    def reset(t: float, x: jax.Array, theta_H: float, Jds: float) -> jax.Array:
        r, y, rd, yd = np.ravel(x)
        th = np.atan(y)
        thd = yd/(1+y**2)

        thp = -(theta_H - th)
        vm = rd*np.array([[np.sin(th)], [np.cos(th)]]) + r*thd*np.array([[np.cos(th)], [-np.sin(th)]]) # Preimpact velocity in cartesian coordinates
        vhp = np.array([[np.cos(thp)], [-np.sin(thp)]]) # Unit vector in perpendicular direction to new stance leg
        rhm = np.array([[np.sin(th)], [np.cos(th)]]) # Unit vector parallel to old stance leg
        vp = np.dot(vm.T, vhp)*vhp + Jds*rhm # Postimpact velocity in cartesian coordinates

        Rcwp_inv = np.array([[np.cos(-thp), np.sin(-thp)], [-np.sin(-thp), np.cos(-thp)]])

        vp_sl = Rcwp_inv@vp
        thdp = (vp_sl[0]/r0).flatten()
        rdp = (vp_sl[1]).flatten()

        yp = np.tan(-(theta_H - th))
        ydp = thdp*(1+yp**2)

        return np.hstack ([
            r0,
            yp,
            rdp,
            ydp
        ])

    @staticmethod
    def foot_height(t: float, y: jax.Array, theta_H: float) -> jax.Array:
        # guard surface in the transformed coordinates
        return np.array([-1/(r0*np.sin(theta_H)), 1, 0, 0])@y + 1/np.tan(theta_H)

    @staticmethod
    def foot_vel(t: float, x: jax.Array, theta_H: float) -> jax.Array:
        grad_h = np.array([-1/(r0*np.sin(theta_H)), 1, 0, 0])
        return np.dot(grad_h, PrismaticWalkerTransformed.f(t, x, np.array([0]))) # note that u doesn't matter here
