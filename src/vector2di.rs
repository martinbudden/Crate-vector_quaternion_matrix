use crate::Vector2d;
use core::ops::Mul;

/// 2-dimensional `{x, y}` vector of `i8` values
pub type Vector2di8 = Vector2d<i8>;
/// 2-dimensional `{x, y}` vector of `i16` values
pub type Vector2di16 = Vector2d<i16>;
/// 2-dimensional `{x, y}` vector of `i32` values
pub type Vector2di32 = Vector2d<i32>;

impl Mul<f32> for Vector2d<i8> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i8, y: (self.y as f32 * k) as i8 }
    }
}

impl Mul<f32> for Vector2d<i16> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i16, y: (self.y as f32 * k) as i16 }
    }
}

impl Mul<f32> for Vector2d<i32> {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self { x: (self.x as f32 * k) as i32, y: (self.y as f32 * k) as i32 }
    }
}

/// `Vector2d<f32>` from `Vector2d<i16>`
/// ```
/// # use vector_quaternion_matrix::{Vector2df32,Vector2di16,Vector2di32};
/// let v_i16 = Vector2di16{x: 2, y: 3};
/// let v_f32 = Vector2df32::from(v_i16);
///
/// let w_f32 = Vector2df32{x: 7.0, y: 11.0};
/// let w_i16 : Vector2di16 = w_f32.into();
///
/// let u_i32 = Vector2di32{x: 17, y: 19};
/// let u_f32 : Vector2df32 = u_i32.into();
///
/// assert_eq!(v_f32, Vector2df32 { x: 2.0, y: 3.0 });
/// assert_eq!(w_i16, Vector2di16 { x: 7, y: 11 });
/// assert_eq!(u_f32, Vector2df32 { x: 17.0, y: 19.0 });
/// ```
impl From<Vector2d<i16>> for Vector2d<f32> {
    fn from(v: Vector2d<i16>) -> Self {
        Self { x: v.x as f32, y: v.y as f32 }
    }
}

impl From<Vector2d<f32>> for Vector2d<i16> {
    fn from(v: Vector2d<f32>) -> Self {
        Self { x: v.x as i16, y: v.y as i16 }
    }
}
impl From<Vector2d<i32>> for Vector2d<f32> {
    fn from(v: Vector2d<i32>) -> Self {
        Self { x: v.x as f32, y: v.y as f32 }
    }
}

impl From<Vector2d<f32>> for Vector2d<i32> {
    fn from(v: Vector2d<f32>) -> Self {
        Self { x: v.x as i32, y: v.y as i32 }
    }
}
