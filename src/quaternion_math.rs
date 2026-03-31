use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "align")] {
        //use core::simd::simd_swizzle;
    }
}

use crate::Quaternion;

pub trait QuaternionMath: Sized {
    fn neg(q: Quaternion<Self>) -> Quaternion<Self>;
    fn conjugate(q: Quaternion<Self>) -> Quaternion<Self>;
    fn mul(a: Quaternion<Self>, b: Quaternion<Self>) -> Quaternion<Self>;
}

impl QuaternionMath for f64 {
    #[inline(always)]
    fn neg(q: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: -q.w, x: -q.x, y: -q.y, z: -q.z }
    }

    #[inline(always)]
    fn conjugate(q: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion { w: q.w, x: -q.x, y: -q.y, z: -q.z }
    }

    #[inline(always)]
    fn mul(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        Quaternion {
            w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
            x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
            y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
            z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
        }
    }
}

// SIMD-accelerated implementation for f32
impl QuaternionMath for f32 {
    #[inline(always)]
    fn neg(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;
            // Transmute the 16-byte aligned struct to a SIMD register
            let q_simd: f32x4 = unsafe { core::mem::transmute(q) };

            // Negate all 4 lanes (x, y, z, w) simultaneously
            let res_simd = -q_simd;

            unsafe { core::mem::transmute(res_simd) }
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: -q.w, x: -q.x, y: -q.y, z: -q.z }
        }
    }

    #[inline(always)]
    fn conjugate(q: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;
            let q_simd: f32x4 = unsafe { core::mem::transmute(q) };

            // Negate x, y, z but keep w positive
            // Mask: [1.0, -1.0, -1.0, -1.0]
            let mask = f32x4::from_array([1.0, -1.0, -1.0, -1.0]);
            let res_simd = q_simd * mask;

            unsafe { core::mem::transmute(res_simd) }
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion { w: q.w, x: -q.x, y: -q.y, z: -q.z }
        }
    }
    #[inline(always)]
    fn mul(lhs: Quaternion<Self>, rhs: Quaternion<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            Quaternion {
                w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
                x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
                y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
                z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion {
                w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
                x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
                y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
                z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
            }
        }
    }
}

// **** Mul ****
/*
impl core::ops::Mul for Quaternion {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(feature = "simd")]
        {
            use core::simd::{f32x4, simd_swizzle};

            let a: f32x4 = unsafe { core::mem::transmute_copy(&self) };
            let b: f32x4 = unsafe { core::mem::transmute_copy(&rhs) };

            // 1. Initial product: [w1*x2, w1*y2, w1*z2, w1*w2]
            // We swizzle 'a' to broadcast 'w' into all lanes
            let a_wwww = simd_swizzle!(a, [3, 3, 3, 3]);
            let mut res = a_wwww * b;

            // 2. Add/Sub subsequent terms using swizzles and FMA
            // [x1*w2, y1*w2, z1*w2, -x1*x2]
            let a_xyzx = simd_swizzle!(a, [0, 1, 2, 0]);
            let b_wwxx = simd_swizzle!(b, [3, 3, 3, 0]);
            // Logic: res = res + (a_xyzx * b_wwxx) with sign flips..

            // Note: For brevity, most SIMD libs use a specific
            // set of 4 vector FMAs to complete the Hamilton product.

            // For now, let's look at the robust Scalar version that
            // the compiler can still auto-vectorize:
            self.scalar_mul(rhs)
        }
        #[cfg(not(feature = "simd"))]
        {
            self.scalar_mul(rhs)
        }
    }
}

    impl Quaternion {
    #[inline(always)]
    fn scalar_mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}
*/
