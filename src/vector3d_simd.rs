use cfg_if::cfg_if;
//use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
//Add<Output = T> + Sub<Output = T> + Mul<R, Output = T>

use crate::Vector3d;
use num_traits::MulAdd;

cfg_if! {
    if #[cfg(feature = "align")] {
        //use core::ops::Mul;
        use core::simd::f32x4;
        use core::mem::transmute;
        use core::simd::simd_swizzle;
        use crate::Vector3df32;
    }
}

impl<T> Vector3d<T>
where
    T: VectorMath,
{
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> T {
        T::dot(self, other)
    }
}
/*
/// Vector dot product
/// ```
/// # use vector_quaternion_matrix::Vector3df32;
/// let v = Vector3df32::new(2.0, 3.0, 5.0);
/// let w = Vector3df32::new(7.0, 11.0, 13.0);
///
/// let x = v.dot(w);
///
/// assert_eq!(x, 112.0);
/// ```
pub fn dot(&self, rhs: Self) -> T {
    self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
}
*/

pub trait VectorMath: Sized {
    fn dot(a: &Vector3d<Self>, b: &Vector3d<Self>) -> Self;
    fn cross(a: &Vector3d<Self>, b: &Vector3d<Self>) -> Vector3d<Self>;
    //fn normalize(v: &Vector3d<Self>) -> Vector3d<Self>;
}

// Default/Scalar implementation for f64
impl VectorMath for f64 {
    #[inline(always)]
    fn dot(a: &Vector3d<Self>, b: &Vector3d<Self>) -> Self {
        (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
    }
    fn cross(a: &Vector3d<Self>, b: &Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }
    }
    /*fn normalize(v: &Vector3d<Self>) -> Vector3d<Self> {
        let mag_sq = v.x * v.x + v.y * v.y + v.z * v.z;
        if mag_sq > 0.0 {
            let inv_mag = 1.0 / mag_sq.sqrt();
            Vector3d { x: v.x * inv_mag, y: v.y * inv_mag, z: v.z * inv_mag }
        } else {
            Vector3d::default()
        }
    }*/
}

// SIMD-accelerated implementation for f32
impl VectorMath for f32 {
    #[inline(always)]
    fn dot(a: &Vector3d<Self>, b: &Vector3d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;
            use core::simd::num::SimdFloat;

            let va: f32x4 = unsafe { core::mem::transmute_copy(a) };
            let vb: f32x4 = unsafe { core::mem::transmute_copy(b) };

            // Multiply the vectors
            let prod = va * vb;

            // Create a vector that acts as a "filter"
            let mask_vec = f32x4::from_array([1.0, 1.0, 1.0, 0.0]);

            // Zero out the 4th lane (padding)
            let filtered = prod * mask_vec;

            filtered.reduce_sum()
        }
        #[cfg(not(feature = "simd"))]
        {
            (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
        }
    }

    #[inline(always)]
    fn cross(a: &Vector3d<Self>, b: &Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;

            let va: f32x4 = unsafe { core::mem::transmute_copy(a) };
            let vb: f32x4 = unsafe { core::mem::transmute_copy(b) };

            // Swizzle A: [y, z, x, w]
            let a_yzx = simd_swizzle!(va, [1, 2, 0, 3]);
            // Swizzle B: [z, x, y, w]
            let b_zxy = simd_swizzle!(vb, [2, 0, 1, 3]);

            // Swizzle A2: [z, x, y, w]
            let a_zxy = simd_swizzle!(va, [2, 0, 1, 3]);
            // Swizzle B2: [y, z, x, w]
            let b_yzx = simd_swizzle!(vb, [1, 2, 0, 3]);

            // Result = (a_yzx * b_zxy) - (a_zxy * b_yzx)
            let res_simd = (a_yzx * b_zxy) - (a_zxy * b_yzx);

            // Transmute back to our Vector3d struct
            unsafe { core::mem::transmute_copy(&res_simd) }
        }
        #[cfg(not(feature = "simd"))]
        {
            Vector3d { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }
        }
    }

    /*#[inline(always)]
    fn normalize(v: &Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;

            // 1. Calculate magnitude squared using our SIMD Dot Product
            let mag_sq = Self::dot(v, v);

            // 2. Guard against division by zero (Important for sensor glitches!)
            if mag_sq > 0.0 {
                use crate::SqrtMethods;

                let inv_mag = mag_sq.reciprocal_sqrt(); // Uses hardware vrsqrt

                // 3. Load vector into SIMD and "Splat" the inverse magnitude
                let mut v_simd: f32x4 = unsafe { core::mem::transmute_copy(v) };
                let scale = f32x4::splat(inv_mag);

                // 4. Multiply all lanes at once
                v_simd *= scale;

                unsafe { core::mem::transmute_copy(&v_simd) }
            } else {
                Vector3d::default() // Return zero vector if magnitude is 0
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            let mag_sq = v.x * v.x + v.y * v.y + v.z * v.z;
            if mag_sq > 0.0 {
                let inv_mag = 1.0 / mag_sq.sqrt();
                Vector3d { x: v.x * inv_mag, y: v.y * inv_mag, z: v.z * inv_mag }
            } else {
                Vector3d::default()
            }
        }
    }*/
}

#[cfg(feature = "simd")]
impl MulAdd<f32, Vector3d<f32>> for Vector3d<f32> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: f32, b: Self) -> Self {
        use core::simd::f32x4;
        let v_self: f32x4 = self.into();
        let v_b: f32x4 = b.into();
        let v_a = f32x4::splat(a);

        // This maps to the Vector Fused Multiply-Add instruction
        let res = (v_self * v_a) + v_b;
        res.into()
    }
}

#[cfg(not(feature = "simd"))]
impl MulAdd<f32, Vector3d<f32>> for Vector3d<f32> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: f32, b: Self) -> Self {
        Vector3d { x: self.x * a + b.x, y: self.y * a + b.y, z: self.z * a + b.z }
    }
}

impl MulAdd<f64, Vector3d<f64>> for Vector3d<f64> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: f64, b: Self) -> Self {
        Vector3d { x: self.x * a + b.x, y: self.y * a + b.y, z: self.z * a + b.z }
    }
}

#[cfg(feature = "simd")]
impl From<Vector3df32> for f32x4 {
    #[inline(always)]
    fn from(v: Vector3df32) -> Self {
        // SAFETY: Both types are 16 bytes and aligned to 16 bytes.
        // The 'dummy' 4th float in the SIMD lane will be whatever
        // was in the padding (usually 0.0 if you use Default).
        unsafe { transmute(v) }
    }
}

#[cfg(feature = "simd")]
impl From<f32x4> for Vector3df32 {
    #[inline(always)]
    fn from(simd: f32x4) -> Self {
        // SAFETY: Same size and alignment.
        unsafe { transmute(simd) }
    }
}

/*impl<T> Mul<f32> for Vector3d<T>
where
    T: Mul<f32, Output = T>
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}*/
/*

// Inside your VectorMath trait or a specialized impl for f32
#[cfg(feature = "simd")]
{
    use core::simd::f32x4;

    // 1. Transmute to SIMD
    let v_simd: f32x4 = unsafe { core::mem::transmute_copy(&self) };

    // 2. "Splat" the scalar: [s, s, s, s]
    let scalar_simd = f32x4::splat(rhs);

    // 3. Multiply all 4 lanes in 1 cycle (x*s, y*s, z*s, padding*s)
    let res_simd = v_simd * scalar_simd;

    unsafe { core::mem::transmute_copy(&res_simd) }
}
 */
/*
impl Mul<Vector3d<f32>> for f32 {
    type Output = Vector3d<f32>;

    #[inline(always)]
    fn mul(self, rhs: Vector3d<f32>) -> Self::Output {
        rhs * self // Just reuse the implementation we already wrote!
    }
}
*/
#[cfg(test)]
mod tests {
    use super::*;
    //use core::mem::{align_of, size_of};
    use crate::Vector3df32;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Vector3d<f32>>();
    }
    #[test]
    fn dot() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let b = Vector3df32 { x: 7.0, y: 11.0, z: 13.0 };
        assert_eq!(a.dot(&a), 38.0);
        assert_eq!(a.dot(&b), 112.0);
        assert_eq!(b.dot(&a), 112.0);
        assert_eq!(b.dot(&b), 339.0);
        let v1 = Vector3df32::new(1.0, 2.0, 3.0);
        let v2 = Vector3df32::new(4.0, 5.0, 6.0);
        // (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        assert_eq!(v1.dot(&v2), 32.0);
    }
    #[test]
    fn normalize() {
        let a = Vector3df32 { x: 2.0, y: 3.0, z: 5.0 };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        //b.normalize();
        assert_eq!(b, a_normalized);
        let z = Vector3df32 { x: 0.0, y: 0.0, z: 0.0 };
        let mut y = z;
        y.normalize();
        assert_eq!(z, y);
    }
}
