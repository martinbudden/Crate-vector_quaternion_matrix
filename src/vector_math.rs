use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "align")] {
        use core::simd::simd_swizzle;
    }
}
use crate::{SqrtMethods, Vector3d};

pub trait VectorMath: Sized {
    fn dot(a: Vector3d<Self>, b: Vector3d<Self>) -> Self;
    fn cross(a: Vector3d<Self>, b: Vector3d<Self>) -> Vector3d<Self>;
    fn normalize(v: Vector3d<Self>) -> Vector3d<Self>;
}

// Default/Scalar implementation for f64
impl VectorMath for f64 {
    #[inline(always)]
    fn dot(a: Vector3d<Self>, b: Vector3d<Self>) -> Self {
        (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
    }
    fn cross(a: Vector3d<Self>, b: Vector3d<Self>) -> Vector3d<Self> {
        Vector3d { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }
    }
    fn normalize(v: Vector3d<Self>) -> Vector3d<Self> {
        let norm_squared = v.x * v.x + v.y * v.y + v.z * v.z;
        if norm_squared == 0.0 {
            return Vector3d::default();
        }
        let norm_reciprocal = norm_squared.reciprocal_sqrt();
        Vector3d { x: v.x * norm_reciprocal, y: v.y * norm_reciprocal, z: v.z * norm_reciprocal }
    }
}

// SIMD-accelerated implementation for f32
impl VectorMath for f32 {
    #[inline(always)]
    fn dot(a: Vector3d<Self>, b: Vector3d<Self>) -> Self {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;
            use core::simd::num::SimdFloat;

            let va: f32x4 = unsafe { core::mem::transmute(a) };
            let vb: f32x4 = unsafe { core::mem::transmute(b) };

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
    fn cross(a: Vector3d<Self>, b: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;

            let va: f32x4 = unsafe { core::mem::transmute(a) };
            let vb: f32x4 = unsafe { core::mem::transmute(b) };

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

    #[inline(always)]
    fn normalize(v: Vector3d<Self>) -> Vector3d<Self> {
        #[cfg(feature = "simd")]
        {
            use core::simd::f32x4;

            // 1. Calculate magnitude squared using our SIMD Dot Product
            let norm_squared = Self::dot(v, v);

            // 2. Guard against division by zero (Important for sensor glitches!)
            if norm_squared == 0.0 {
                return Vector3d::default(); // Return zero vector if magnitude is 0
            }
            use crate::SqrtMethods;

            let norm_reciprocal = norm_squared.reciprocal_sqrt(); // Uses hardware vrsqrt

            // 3. Load vector into SIMD and "Splat" the inverse magnitude
            let mut v_simd: f32x4 = unsafe { core::mem::transmute(v) };
            let scale = f32x4::splat(norm_reciprocal);

            // 4. Multiply all lanes at once
            v_simd *= scale;

            unsafe { core::mem::transmute(v_simd) }
        }
        #[cfg(not(feature = "simd"))]
        {
            let norm_squared = v.x * v.x + v.y * v.y + v.z * v.z;
            if norm_squared == 0.0 {
                return Vector3d::default();
            }
            let norm_reciprocal = norm_squared.reciprocal_sqrt();
            Vector3d { x: v.x * norm_reciprocal, y: v.y * norm_reciprocal, z: v.z * norm_reciprocal }
        }
    }
}
