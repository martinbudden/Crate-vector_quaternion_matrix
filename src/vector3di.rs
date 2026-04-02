use core::ops::Mul;

use crate::Vector3d;

/// 3-dimensional `{x, y, z}` vector of `i8` values
pub type Vector3di8 = Vector3d<i8>;
/// 3-dimensional `{x, y, z}` vector of `i16` values
pub type Vector3di16 = Vector3d<i16>;
/// 3-dimensional `{x, y, z}` vector of `i32` values
pub type Vector3di32 = Vector3d<i32>;

impl Mul<f32> for Vector3d<i8> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i8, y: (self.y as f32 * k) as i8, z: (self.z as f32 * k) as i8 }
    }
}

impl Mul<f32> for Vector3d<i16> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i16, y: (self.y as f32 * k) as i16, z: (self.z as f32 * k) as i16 }
    }
}

impl Mul<f32> for Vector3d<i32> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i32, y: (self.y as f32 * k) as i32, z: (self.z as f32 * k) as i32 }
    }
}

/// `Vector3d<f32>` from `Vector3d<i16>`
/// ```
/// # use vector_quaternion_matrix::{Vector3df32,Vector3di16,Vector3di32};
/// let v_i16 = Vector3di16{x: 2, y: 3, z: 5};
/// let v_f32 = Vector3df32::from(v_i16);
///
/// let w_f32 = Vector3df32{x: 7.0, y: 11.0, z: 13.0};
/// let w_i16 : Vector3di16 = w_f32.into();
///
/// let u_i32 = Vector3di32{x: 17, y: 19, z: 23};
/// let u_f32 : Vector3df32 = u_i32.into();
///
/// assert_eq!(v_f32, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w_i16, Vector3di16 { x: 7, y: 11, z: 13 });
/// assert_eq!(u_f32, Vector3df32 { x: 17.0, y: 19.0, z: 23.0 });
/// ```
impl From<Vector3d<i16>> for Vector3d<f32> {
    fn from(v: Vector3d<i16>) -> Self {
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i16> {
    fn from(v: Vector3d<f32>) -> Self {
        Self { x: v.x as i16, y: v.y as i16, z: v.z as i16 }
    }
}

impl From<Vector3d<i32>> for Vector3d<f32> {
    fn from(v: Vector3d<i32>) -> Self {
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i32> {
    fn from(v: Vector3d<f32>) -> Self {
        Self { x: v.x as i32, y: v.y as i32, z: v.z as i32 }
    }
}
