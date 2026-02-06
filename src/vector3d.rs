use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use num_traits::Zero;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3d {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Vector from tuple
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::from((2.0, 3.0, 5.0));
/// let w: Vector3d = (7.0, 11.0, 13.0).into();
///
/// assert_eq!(v, Vector3d{ x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w, Vector3d{ x: 7.0, y: 11.0, z: 13.0 });
/// ```
impl From<(f32, f32, f32)> for Vector3d {
    fn from(v: (f32, f32, f32)) -> Self {
        Self {
            x: v.0,
            y: v.1,
            z: v.2,
        }
    }
}

/// Vector from array
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::from([2.0, 3.0, 5.0]);
/// let w: Vector3d = [7.0, 11.0, 13.0].into();
///
/// assert_eq!(v, Vector3d{ x: 2.0, y: 3.0, z: 5.0 });
/// assert_eq!(w, Vector3d{ x: 7.0, y: 11.0, z: 13.0 });
/// ```
impl From<[f32; 3]> for Vector3d {
    fn from(v: [f32; 3]) -> Self {
        Self {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }
}

/// Array from vector
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d{ x: 2.0, y: 3.0, z: 5.0 };
///
/// let a = <[f32; 3]>::from(v);
/// let b: [f32; 3] = v.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0]);
/// ```
impl From<Vector3d> for [f32; 3] {
    fn from(v: Vector3d) -> Self {
        [v.x, v.y, v.z]
    }
}

/// Zero vector
/// ```
/// # use vector_quaternion_matrix::Vector3d;
/// # use num_traits::zero;
///
/// let z: Vector3d = zero();
///
/// assert_eq!(z, Vector3d{ x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl Zero for Vector3d {
    fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }
}

/// Negate vector
/// ```
/// # use vector_quaternion_matrix::Vector3d;
/// let mut v = Vector3d{ x: 2.0, y: 3.0, z: 5.0 };
/// v = -v;
///
/// assert_eq!(v, Vector3d { x: -2.0, y: -3.0, z: -5.0 });
/// ```   
impl Neg for Vector3d {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Add two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3d;
/// let u = Vector3d::new(2.0, 3.0, 5.0);
/// let v = Vector3d::new(7.0, 11.0, 13.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector3d { x: 9.0, y: 14.0, z: 18.0 });
/// ```   
impl Add for Vector3d {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

// Add one vector to another
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut r = Vector3d::new(2.0, 3.0, 5.0);
/// let u = Vector3d::new(7.0, 11.0, 13.0);
/// r += u;
///
/// assert_eq!(r, Vector3d { x: 9.0, y: 14.0, z: 18.0 });
///
/// # use num_traits::zero;
///
/// let z: Vector3d = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```   
impl AddAssign for Vector3d {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// Subtract two vectors
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let u = Vector3d::new(2.0, 3.0, 5.0);
/// let v = Vector3d::new(7.0, 11.0, 13.0);
/// let r = u - v;
///
/// assert_eq!(r, Vector3d { x: -5.0, y: -8.0, z: -8.0 });
/// ```   
impl Sub for Vector3d {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

// Subtract one vector from another
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut r = Vector3d::new(2.0, 3.0, 5.0);
/// let v = Vector3d::new(7.0, 11.0, 17.0);
/// r -= v;
///
/// assert_eq!(r, Vector3d { x: -5.0, y: -8.0, z: -12.0 });
/// ```
impl SubAssign for Vector3d {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

/// Pre-multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::new(2.0, 3.0, 5.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl Mul<Vector3d> for f32 {
    type Output = Vector3d;
    fn mul(self, rhs: Vector3d) -> Vector3d {
        Vector3d {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

/// Multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::new(2.0, 3.0, 5.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl Mul<f32> for Vector3d {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }
}

/// In-place multiply a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut v = Vector3d::new(2.0, 3.0, 5.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector3d { x: 4.0, y: 6.0, z: 10.0 });
/// ```
impl MulAssign<f32> for Vector3d {
    fn mul_assign(&mut self, k: f32) {
        *self = *self * k;
    }
}

/// Divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::new(2.0, 3.0, 5.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl Div<f32> for Vector3d {
    type Output = Self;
    fn div(self, k: f32) -> Self {
        let r: f32 = 1.0 / k;
        Self {
            x: self.x * r,
            y: self.y * r,
            z: self.z * r,
        }
    }
}

/// In-place divide a vector by a constant
///    v /= k;
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut v = Vector3d::new(2.0, 3.0, 5.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector3d { x: 1.0, y: 1.5, z: 2.5 });
/// ```
impl DivAssign<f32> for Vector3d {
    fn div_assign(&mut self, k: f32) {
        *self = *self / k;
    }
}

/// Access vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let v = Vector3d::new(2.0, 3.0, 5.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// assert_eq!(v[2], 5.0);
/// ```
impl Index<usize> for Vector3d {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.z, // default to z component if index out of range
        }
    }
}

// Set vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector3d;
///
/// let mut v = Vector3d::new(2.0, 3.0, 5.0);
/// v[0] = 7.0;
/// v[1] = 11.0;
/// v[2] = 13.0;
///
/// assert_eq!(v, Vector3d { x:7.0, y:11.0, z:13.0 });
/// ```
impl IndexMut<usize> for Vector3d {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.z, // default to z component if index out of range
        }
    }
}

impl Vector3d {
    /// Create a vector
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    /// Vector dot product
    pub fn dot(&self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    /// Vector cross product
    pub fn cross(&self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: -self.z * rhs.z + self.z * rhs.x,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
    /// Return distance between two points, squared
    pub fn distance_squared(&self, rhs: Self) -> f32 {
        (*self - rhs).squared_norm()
    }
    // Return distance between two points
    pub fn distance(&self, rhs: Self) -> f32 {
        self.distance_squared(rhs).sqrt()
    }
    /// Return square of Euclidean norm
    pub fn squared_norm(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    /// Return Euclidean norm
    pub fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    /// Return normalized form of vector
    pub fn normalized(&self) -> Self {
        let norm: f32 = self.norm();
        // If norm == 0.0 then the vector is already normalized
        if norm == 0.0 {
            return *self;
        }
        *self / norm
    }
    /// Normalize the vector in place
    pub fn normalize(&mut self) {
        let norm: f32 = self.norm();
        // If norm == 0.0 then the vector is already normalized
        if norm != 0.0 {
            *self /= self.norm();
        }
    }
    /// Return a copy of the vector with all components set to their absolute values
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }
    /// Set all components of the vector to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
    /// Return a copy of the vector with all components clamped to the specified range
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }
    /// Clamp all components of the vector to the specified range
    pub fn clamp_in_place(&mut self, min: f32, max: f32) {
        *self = self.clamp(min, max);
    }
    /// Return the sum of all components of the vector
    pub fn sum(&self) -> f32 {
        self.x + self.y + self.z
    }
    /// Return the mean of all components of the vector
    pub fn mean(&self) -> f32 {
        (self.x + self.y + self.z) / 3.0
    }
    /// Return the product of all components of the vector
    pub fn product(&self) -> f32 {
        self.x * self.y * self.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Vector3d>();
    }
    #[test]
    fn default() {
        let a: Vector3d = Vector3d::default();
        assert_eq!(
            a,
            Vector3d {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        );
        let z: Vector3d = Vector3d::zero();
        //let z: Vector3d = zero();
        assert_eq!(a, z);
        assert!(z.is_zero());
    }
    #[test]
    fn neg() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(
            -a,
            Vector3d {
                x: -2.0,
                y: -3.0,
                z: -5.0
            }
        );

        let b = -a;
        assert_eq!(
            b,
            Vector3d {
                x: -2.0,
                y: -3.0,
                z: -5.0
            }
        );
    }
    #[test]
    fn add() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let b = Vector3d {
            x: 7.0,
            y: 11.0,
            z: 13.0,
        };
        assert_eq!(
            a + b,
            Vector3d {
                x: 9.0,
                y: 14.0,
                z: 18.0
            }
        );
    }
    #[test]
    fn add_assign() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let b = Vector3d {
            x: 7.0,
            y: 11.0,
            z: 13.0,
        };
        let mut c = a;
        c += b;
        assert_eq!(
            c,
            Vector3d {
                x: 9.0,
                y: 14.0,
                z: 18.0
            }
        );
    }
    #[test]
    fn sub() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let b = Vector3d {
            x: 7.0,
            y: 11.0,
            z: 17.0,
        };
        let c = a - b;
        assert_eq!(
            c,
            Vector3d {
                x: -5.0,
                y: -8.0,
                z: -12.0
            }
        );
    }
    #[test]
    fn sub_assign() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let b = Vector3d {
            x: 7.0,
            y: 11.0,
            z: 17.0,
        };
        let mut c = a;
        c -= b;
        assert_eq!(
            c,
            Vector3d {
                x: -5.0,
                y: -8.0,
                z: -12.0
            }
        );
    }
    #[test]
    fn mul() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(
            a * 2.0,
            Vector3d {
                x: 4.0,
                y: 6.0,
                z: 10.0
            }
        );
        assert_eq!(
            2.0 * a,
            Vector3d {
                x: 4.0,
                y: 6.0,
                z: 10.0
            }
        );
    }
    #[test]
    fn mul_assign() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let mut b = a;
        b *= 2.0;
        assert_eq!(
            b,
            Vector3d {
                x: 4.0,
                y: 6.0,
                z: 10.0
            }
        );
    }
    #[test]
    fn div() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(
            a / 2.0,
            Vector3d {
                x: 1.0,
                y: 1.5,
                z: 2.5
            }
        );
    }
    #[test]
    fn div_assign() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let mut b = a;
        b /= 2.0;
        assert_eq!(
            b,
            Vector3d {
                x: 1.0,
                y: 1.5,
                z: 2.5
            }
        );
    }
    #[test]
    fn new() {
        let a = Vector3d::new(2.0, 3.0, 5.0);
        assert_eq!(
            a,
            Vector3d {
                x: 2.0,
                y: 3.0,
                z: 5.0
            }
        );
        let b = Vector3d::from((2.0, 3.0, 5.0));
        assert_eq!(a, b);

        use num_traits::zero;
        let z: Vector3d = zero();
        assert!(z.is_zero());

        let c: Vector3d = (2.0, 3.0, 5.0).into();
        assert_eq!(a, c);
        let d = Vector3d::from((2.0, 3.0, 5.0));
        assert_eq!(a, d);
        let e: Vector3d = [2.0, 3.0, 5.0].into();
        assert_eq!(a, e);
        let f = Vector3d::from([2.0, 3.0, 5.0]);
        assert_eq!(a, f);

        let h = <[f32; 3]>::from(a);
        assert_eq!([2.0, 3.0, 5.0], h);
        let i: [f32; 3] = a.into();
        assert_eq!([2.0, 3.0, 5.0], i);
    }
    #[test]
    fn dot() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let b = Vector3d {
            x: 7.0,
            y: 11.0,
            z: 13.0,
        };
        assert_eq!(a.dot(a), 38.0);
        assert_eq!(a.dot(b), 112.0);
        assert_eq!(b.dot(a), 112.0);
        assert_eq!(b.dot(b), 339.0);
    }
    #[test]
    fn squared_norm() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(a.squared_norm(), 38.0);
    }
    #[test]
    fn norm() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(a.norm(), 38.0_f32.sqrt());
        let z = Vector3d {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let b = a / 38.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Vector3d {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(z.normalized(), z);
    }
    #[test]
    fn normalize() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        assert_eq!(b, a_normalized);
        let z = Vector3d {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let mut y = z;
        y.normalize();
        assert_eq!(z, y);
    }
    #[test]
    fn abs() {
        let a = Vector3d {
            x: -2.0,
            y: -3.0,
            z: -5.0,
        };
        assert_eq!(
            a.abs(),
            Vector3d {
                x: 2.0,
                y: 3.0,
                z: 5.0
            }
        );
    }
    #[test]
    fn abs_in_place() {
        let a = Vector3d {
            x: -2.0,
            y: -3.0,
            z: -5.0,
        };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamp() {
        let a = Vector3d {
            x: -2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(
            a.clamp(-1.0, 4.0),
            Vector3d {
                x: -1.0,
                y: 3.0,
                z: 4.0
            }
        );
    }
    #[test]
    fn clamp_in_place() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
    #[test]
    fn sum() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(a.sum(), 10.0);
    }
    #[test]
    fn mean() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(a.mean(), 10.0 / 3.0);
    }
    #[test]
    fn product() {
        let a = Vector3d {
            x: 2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(a.product(), 30.0);
    }
}
