use core::ops::{Add, Mul, Sub};
use num_traits::Zero;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3dI16 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}

/// Vector from tuple
/// ```
/// # use vector_quaternion_matrix::Vector3dI16;
///
/// let v = Vector3dI16::from((2, 3, 5));
/// let w: Vector3dI16 = (7, 11, 13).into();
///
/// assert_eq!(v, Vector3dI16{ x: 2, y: 3, z: 5 });
/// assert_eq!(w, Vector3dI16{ x: 7, y: 11, z: 13 });
/// ```
impl From<(i16, i16, i16)> for Vector3dI16 {
    fn from(v: (i16, i16, i16)) -> Self {
        Self {
            x: v.0,
            y: v.1,
            z: v.2,
        }
    }
}

/// Vector from array
/// ```
/// # use vector_quaternion_matrix::Vector3dI16;
///
/// let v = Vector3dI16::from([2, 3, 5]);
/// let w: Vector3dI16 = [7, 11, 13].into();
///
/// assert_eq!(v, Vector3dI16{ x: 2, y: 3, z: 5 });
/// assert_eq!(w, Vector3dI16{ x: 7, y: 11, z: 13 });
/// ```
impl From<[i16; 3]> for Vector3dI16 {
    fn from(v: [i16; 3]) -> Self {
        Self {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }
}

/// Zero Vector3dI16
/// ```
/// # use vector_quaternion_matrix::Vector3dI16;
/// # use num_traits::zero;
///
/// let z: Vector3dI16 = zero();
///
/// assert_eq!(z, Vector3dI16 { x: 0, y: 0, z: 0 });
/// ```
impl Zero for Vector3dI16 {
    fn zero() -> Self {
        Self { x: 0, y: 0, z: 0 }
    }

    fn is_zero(&self) -> bool {
        self.x == 0 && self.y == 0 && self.z == 0
    }
}

/// Add two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3dI16;
///
/// let u = Vector3dI16 { x: 2, y: 3, z: 5 };
/// let v = Vector3dI16 { x: 7, y: 11, z: 13 };
/// let r = u + v;
///
/// assert_eq!(r, Vector3dI16{ x: 9, y: 14, z: 18 });
/// ```   
impl Add for Vector3dI16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

/// Subtract two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3dI16;
///
/// let u = Vector3dI16 { x: 2, y: 3, z: 5 };
/// let v = Vector3dI16 { x: 7, y: 11, z: 17 };
/// let r = u - v;
///
/// assert_eq!(r, Vector3dI16{ x: -5, y: -8, z: -12 });
/// ```   
impl Sub for Vector3dI16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

/// Multiply vector by an i16 constant
/// ```
/// # use vector_quaternion_matrix::Vector3dI16;
///
/// let v = Vector3dI16 { x: 2, y: 3, z: 5 };
/// let r = v * 2;
///
/// assert_eq!(r, Vector3dI16 { x: 4, y: 6, z: 10 });
/// ```
impl Mul<i16> for Vector3dI16 {
    type Output = Self;
    fn mul(self, k: i16) -> Self {
        Self {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }
}

/// Multiply vector by an f32 constant
/// ```
/// # use vector_quaternion_matrix::Vector3dI16;
///
/// let v = Vector3dI16 { x: 2, y: 3, z: 5 };
/// let r = v * 2.0_f32;
///
/// assert_eq!(r, Vector3dI16 { x: 4, y: 6, z: 10 });
/// ```
impl Mul<f32> for Vector3dI16 {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            x: (self.x as f32 * k) as i16,
            y: (self.y as f32 * k) as i16,
            z: (self.z as f32 * k) as i16,
        }
    }
}
