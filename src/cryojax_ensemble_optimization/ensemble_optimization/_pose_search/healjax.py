"""
Copyright (c) 2023 ghcollin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
These functions were ported from
https://github.com/ghcollin/healjax/tree/master

Please see Copyright notice above for more information.
"""

import jax
import jax.numpy as jnp


def hp_to_zphi_polar(nside, bighp, x, y):
    zfactor, x, y = jax.lax.cond(
        jnp.logical_and(issouthpolar(bighp), x + y < nside),
        lambda: (-1.0, nside - y, nside - x),
        lambda: (1.0, x, y),
    )

    phi_t_flag = jnp.logical_or(y != nside, x != nside)
    phi_t = phi_t_flag * (jnp.pi * (nside - y) / (2.0 * ((nside - x) + (nside - y))))

    vv = jnp.where(
        phi_t < jnp.pi / 4,
        jnp.fabs(jnp.pi * (nside - x) / ((2.0 * phi_t - jnp.pi) * nside) / jnp.sqrt(3.0)),
        jnp.fabs(jnp.pi * (nside - y) / (2.0 * phi_t * nside) / jnp.sqrt(3.0)),
    )

    z = (1 - vv) * (1 + vv)
    rad = jnp.sqrt(1.0 + z) * vv

    z = z * zfactor

    # // The big healpix determines the phi offset
    phi = jnp.where(
        issouthpolar(bighp),
        jnp.pi / 2.0 * (bighp - 8) + phi_t,
        jnp.pi / 2.0 * bighp + phi_t,
    )
    phi = jnp.mod(phi, 2 * jnp.pi)
    return z, phi, rad


def hp_to_zphi_equator(nside, bighp, x, y):
    x = x / nside
    y = y / nside

    bighp, zoff, phioff = jax.lax.cond(
        bighp <= 3,
        lambda: (bighp, 0.0, 1.0),  # // north
        lambda: jax.lax.cond(
            bighp <= 7,
            lambda: (bighp - 4, -1.0, 0.0),  # // equator
            lambda: (bighp - 8, -2.0, 1.0),  # // south
        ),
    )

    z = 2.0 / 3.0 * (x + y + zoff)
    phi = jnp.pi / 4 * (x - y + phioff + 2 * bighp)
    phi = jnp.mod(phi, 2 * jnp.pi)
    rad = jnp.sqrt(
        jnp.maximum(0.0, 1 - jnp.square(z))
    )  # This sqrt can cause spirious NaN errors in debug mode, so we clip the input
    return z, phi, rad


def hp_to_zphi(nside, bighp, xp, yp, dx, dy):
    # // this is x,y position in the healpix reference frame
    x = xp + dx
    y = yp + dy

    polar_routine = jnp.logical_or(
        jnp.logical_and(isnorthpolar(bighp), x + y > nside),
        jnp.logical_and(issouthpolar(bighp), x + y < nside),
    )

    return jax.lax.cond(
        polar_routine, hp_to_zphi_polar, hp_to_zphi_equator, nside, bighp, x, y
    )



def zphi2xyz(z, phi, rad):
    x = rad * jnp.cos(phi)
    y = rad * jnp.sin(phi)
    return x, y, z


def xyz2radec(x, y, z):
    ra = jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi)
    dec = jnp.where(
        jnp.fabs(z) > 0.9, jnp.pi / 2 - jnp.arctan2(jnp.hypot(x, y), z), jnp.arcsin(z)
    )
    return ra, dec

def healpixl_nested_to_xy(nside, hp):
    ns2 = nside * nside
    bighp = (hp / ns2).astype(hp.dtype)

    def loop_body(i, carry):
        index0, xc, yc = carry
        new_x = jnp.bitwise_or(xc, jnp.left_shift(jnp.bitwise_and(index0, 0x1), i))
        index1 = jnp.right_shift(index0, 1)
        new_y = jnp.bitwise_or(yc, jnp.left_shift(jnp.bitwise_and(index1, 0x1), i))
        index2 = jnp.right_shift(index1, 1)
        return index2, new_x, new_y

    index = jnp.fmod(hp, ns2)
    x_dtype = hp.dtype
    _, x, y = jax.lax.fori_loop(
        0,
        8 * x_dtype.itemsize // 2,
        loop_body,
        (index, jnp.array(0).astype(x_dtype), jnp.array(0).astype(x_dtype)),
    )

    return bighp, x, y



def isnorthpolar(bighealpix):
    return bighealpix <= 3


def issouthpolar(bighealpix):
    return bighealpix >= 8

##############
# API funcs
##############

def vec2ang_radec(x, y, z):
    return xyz2radec(x, y, z)


def vec2ang(x, y, z):
    ra, dec = vec2ang_radec(x, y, z)
    phi = ra
    theta = jnp.pi / 2 - dec
    return theta, phi

def _pix2ang(nside, hp, dx=None, dy=None):
    dx = 0.5 if dx is None else dx
    dy = 0.5 if dy is None else dy
    return vec2ang(
        *zphi2xyz(*hp_to_zphi(nside, *healpixl_nested_to_xy(nside, hp), dx, dy))
    )

@jax.jit
def pix2ang(nside, hp_indices):
    return jax.vmap(lambda x: _pix2ang(nside, x), in_axes=0)(hp_indices)