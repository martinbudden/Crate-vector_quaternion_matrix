use core::convert::From;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use num_traits::{One, Zero};

use crate::Vector3d;
use crate::math_methods::MathMethods;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RollPitchYaw {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RollPitch {
    pub roll: f32,
    pub pitch: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl From<(f32, f32)> for Quaternion {
    fn from(angles: (f32, f32)) -> Self {
        Quaternion::from_roll_pitch_angles_radians(angles.0, angles.1)
    }
}

impl From<(f32, f32, f32)> for Quaternion {
    fn from(angles: (f32, f32, f32)) -> Self {
        Quaternion::from_roll_pitch_yaw_angles_radians(angles.0, angles.1, angles.2)
    }
}

impl From<RollPitchYaw> for Quaternion {
    fn from(angles: RollPitchYaw) -> Self {
        Quaternion::from_roll_pitch_yaw_angles_radians(angles.roll, angles.pitch, angles.yaw)
    }
}

impl From<RollPitch> for Quaternion {
    fn from(angles: RollPitch) -> Self {
        Quaternion::from_roll_pitch_angles_radians(angles.roll, angles.pitch)
    }
}
/// Quaternion from array
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let v = Quaternion::from([2.0, 3.0, 5.0, 6.0]);
/// let w: Quaternion = [7.0, 11.0, 13.0, 17.0].into();
///
/// assert_eq!(v, Quaternion{ w: 2.0, x: 3.0, y: 5.0, z: 6.0 });
/// assert_eq!(w, Quaternion{ w: 7.0, x: 11.0, y: 13.0, z: 17.0 });
/// ```
impl From<[f32; 4]> for Quaternion {
    fn from(q: [f32; 4]) -> Self {
        Self {
            w: q[0],
            x: q[1],
            y: q[2],
            z: q[3],
        }
    }
}

/// Array from quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let q = Quaternion{ w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
///
/// let a = <[f32; 4]>::from(q);
/// let b: [f32; 4] = q.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0, 7.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0, 7.0]);
/// ```
impl From<Quaternion> for [f32; 4] {
    fn from(q: Quaternion) -> Self {
        [q.w, q.x, q.y, q.z]
    }
}

/// Zero quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// # use num_traits::zero;
///
/// let z: Quaternion = zero();
///
/// assert_eq!(z, Quaternion { w:0.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl Zero for Quaternion {
    fn zero() -> Self {
        Self {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.w == 0.0 && self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }
}

/// Unit quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// # use num_traits::one;
///
/// let i: Quaternion = one();
///
/// assert_eq!(i, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
/// ```
impl One for Quaternion {
    fn one() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    fn is_one(&self) -> bool {
        self.w == 1.0 && self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }
}

/// Negate quaternion
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// let mut q = Quaternion{ w: 2.0, x: 3.0, y: 5.0, z: 7.0 };
/// q = -q;
///
/// assert_eq!(q, Quaternion { w: -2.0, x: -3.0, y: -5.0, z: -7.0 });
/// ```
impl Neg for Quaternion {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Add two quaternions
/// ```
/// # use vector_quaternion_matrix::Quaternion;
/// let u = Quaternion::new(2.0, 3.0, 5.0, 7.0);
/// let v = Quaternion::new(11.0, 13.0, 17.0, 19.0);
/// let r = u + v;
///
/// assert_eq!(r, Quaternion { w: 13.0, x: 16.0, y: 22.0, z: 26.0 });
/// ```
impl Add for Quaternion {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

/// Add one quaternion to another
impl AddAssign for Quaternion {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Subtract two quaternions
impl Sub for Quaternion {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

/// Subtract one quaternion from another
impl SubAssign for Quaternion {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Pre-multiply quaternion by a constant
impl Mul<Quaternion> for f32 {
    type Output = Quaternion;
    fn mul(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            w: self * rhs.w,
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

/// Multiply quaternion by a constant
impl Mul<f32> for Quaternion {
    type Output = Self;
    fn mul(self, k: f32) -> Self {
        Self {
            w: self.w * k,
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }
}

/// In-place multiply a quaternion by a constant
impl MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, k: f32) {
        *self = *self * k;
    }
}

/// Multiply two quaternions
impl Mul<Quaternion> for Quaternion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

/// Multiply one quaternion by another
impl MulAssign<Quaternion> for Quaternion {
    fn mul_assign(&mut self, rhs: Quaternion) {
        *self = *self * rhs;
    }
}

/// Divide a quaternion by a constant
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let q = Quaternion::new(2.0, 3.0, 5.0, 7.0);
/// let r = q / 2.0;
///
/// assert_eq!(r, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
/// ```
impl Div<f32> for Quaternion {
    type Output = Self;
    fn div(self, k: f32) -> Self {
        let r: f32 = 1.0 / k;
        Self {
            w: self.w * r,
            x: self.x * r,
            y: self.y * r,
            z: self.z * r,
        }
    }
}

/// In-place divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let mut q = Quaternion::new(2.0, 3.0, 5.0, 7.0);
/// q /= 2.0;
///
/// assert_eq!(q, Quaternion { w: 1.0, x: 1.5, y: 2.5, z: 3.5 });
/// ```
impl DivAssign<f32> for Quaternion {
    fn div_assign(&mut self, k: f32) {
        *self = *self / k;
    }
}

/// Access quaternion component by index
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let mut q = Quaternion::new(2.0, 3.0, 5.0, 7.0);
///
/// assert_eq!(q[0], 2.0);
/// assert_eq!(q[1], 3.0);
/// assert_eq!(q[2], 5.0);
/// assert_eq!(q[3], 7.0);
/// ```
impl Index<usize> for Quaternion {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.w,
            1 => &self.x,
            2 => &self.y,
            3 => &self.z,
            _ => &self.z, // default to z component if index out of range
        }
    }
}

// Set quaternion component by index
/// ```
/// # use vector_quaternion_matrix::Quaternion;
///
/// let mut q = Quaternion::new(2.0, 3.0, 5.0, 6.0);
/// q[0] = 7.0;
/// q[1] = 11.0;
/// q[2] = 13.0;
/// q[3] = 17.0;
///
/// assert_eq!(q, Quaternion { w:7.0, x:11.0, y:13.0, z: 17.0 });
/// ```
impl IndexMut<usize> for Quaternion {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.w,
            1 => &mut self.x,
            2 => &mut self.y,
            3 => &mut self.z,
            _ => &mut self.z, // default to z component if index out of range
        }
    }
}
impl Quaternion {
    /// Create a quaternion
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Create a Quaternion from roll, pitch, and yaw Euler angles (in radians).
    /// See: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion
    pub fn from_roll_pitch_yaw_angles_radians(
        roll_radians: f32,
        pitch_radians: f32,
        yaw_radians: f32,
    ) -> Self {
        let (sin_half_roll, cos_half_roll) = (0.5 * roll_radians).sin_cos();
        let (sin_half_pitch, cos_half_pitch) = (0.5 * pitch_radians).sin_cos();
        let (sin_half_yaw, cos_half_yaw) = (0.5 * yaw_radians).sin_cos();
        Self {
            w: cos_half_roll * cos_half_pitch * cos_half_yaw
                + sin_half_roll * sin_half_pitch * sin_half_yaw,
            x: sin_half_roll * cos_half_pitch * cos_half_yaw
                - cos_half_roll * sin_half_pitch * sin_half_yaw,
            y: cos_half_roll * sin_half_pitch * cos_half_yaw
                + sin_half_roll * cos_half_pitch * sin_half_yaw,
            z: cos_half_roll * cos_half_pitch * sin_half_yaw
                - sin_half_roll * sin_half_pitch * cos_half_yaw,
        }
    }

    /// Create a Quaternion from roll and pitch Euler angles (in radians), assumes yaw angle is zero.
    pub fn from_roll_pitch_angles_radians(roll_radians: f32, pitch_radians: f32) -> Self {
        let (sin_half_roll, cos_half_roll) = (0.5 * roll_radians).sin_cos();
        let (sin_half_pitch, cos_half_pitch) = (0.5 * pitch_radians).sin_cos();

        Self {
            w: cos_half_roll * cos_half_pitch,
            x: sin_half_roll * cos_half_pitch,
            y: cos_half_roll * sin_half_pitch,
            z: -sin_half_roll * sin_half_pitch,
        }
    }

    /// Create a Quaternion from roll, pitch, and yaw Euler angles (in degrees).
    pub fn from_roll_pitch_yaw_angles_degrees(
        roll_degrees: f32,
        pitch_degrees: f32,
        yaw_degrees: f32,
    ) -> Self {
        Self::from_roll_pitch_yaw_angles_radians(
            roll_degrees.to_radians(),
            pitch_degrees.to_radians(),
            yaw_degrees.to_radians(),
        )
    }

    /// Create a Quaternion from roll and pitch Euler angles (in degrees), assumes yaw angle is zero.
    pub fn from_roll_pitch_angles_degrees(roll_degrees: f32, pitch_degrees: f32) -> Self {
        Self::from_roll_pitch_angles_radians(roll_degrees.to_radians(), pitch_degrees.to_radians())
    }

    // Return the conjugate of the quaternion
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Return the imaginary part of the quaternion
    pub fn imaginary(self) -> Vector3d<f32> {
        Vector3d::<f32> {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
    /// Return the last column of the equivalent rotation matrix, but calculated more efficiently than a full conversion
    pub fn direction_cosine_matrix_z(self) -> Vector3d<f32> {
        Vector3d::<f32> {
            x: 2.0 * (self.w * self.y + self.x * self.z),
            y: 2.0 * (self.y * self.z - self.w * self.x),
            z: self.w * self.w,
        }
    }

    /// Return square of Euclidean norm
    pub fn squared_norm(&self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Return Euclidean norm
    pub fn norm(&self) -> f32 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Return normalized form of the quaternion
    pub fn normalized(&self) -> Self {
        let norm: f32 = self.norm();
        // If norm == 0.0 then the quaternion is already normalized
        if norm == 0.0 {
            return *self;
        }
        *self / norm
    }

    /// Normalize the quaternion in place
    pub fn normalize(&mut self) {
        let norm: f32 = self.norm();
        // If norm == 0.0 then the quaternion is already normalized
        if norm != 0.0 {
            *self /= self.norm();
        }
    }

    pub fn half_gravity(&self) -> Vector3d<f32> {
        Vector3d::<f32> {
            x: self.x * self.z - self.w * self.y,
            y: self.w * self.x + self.y * self.z,
            z: -0.5 + self.w * self.w + self.z * self.z,
        }
    }

    pub fn gravity(&self) -> Vector3d<f32> {
        self.half_gravity() * 2.0
    }

    /// Return a copy of the quaternion with all components set to their absolute values
    pub fn abs(&self) -> Self {
        Self {
            w: self.w.abs(),
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Set all components of the quaternion to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }

    /// Return a copy of the quaternion with all components clamped to the specified range
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            w: self.w.clamp(min, max),
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    /// Clamp all components of the quaternion to the specified range
    pub fn clamp_in_place(&mut self, min: f32, max: f32) {
        *self = self.clamp(min, max);
    }

    /// Rotate about the x-axis,
    /// equivalent to *= Quaternion(cos(theta/2), sin(theta/2), 0, 0)
    pub fn rotate_x(&mut self, theta: f32) {
        let (sin, cos) = (theta / 2.0).sin_cos();
        let wt: f32 = self.w * cos - self.x * sin;
        self.x = self.w * sin + self.x * cos;
        let yt: f32 = self.y * cos + self.z * sin;
        self.z = -self.y * sin + self.z * cos;
        self.w = wt;
        self.y = yt;
    }

    /// Rotate about the y-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, sin(theta/2), 0)
    pub fn rotate_y(&mut self, theta: f32) {
        let (sin, cos) = (theta / 2.0).sin_cos();
        let wt: f32 = self.w * cos - self.y * sin;
        let xt: f32 = self.x * cos - self.z * sin;
        self.y = self.w * sin + self.y * cos;
        self.z = self.x * sin - self.z * cos;
        self.w = wt;
        self.x = xt;
    }

    /// Rotate about the z-axis,
    /// equivalent to *= Quaternion(cos(theta/2), 0, 0, sin(theta/2))
    pub fn rotate_z(&mut self, theta: f32) {
        let (sin, cos) = (theta / 2.0).sin_cos();
        let wt: f32 = self.w * cos - self.z * sin;
        let xt: f32 = self.x * cos - self.y * sin;
        self.y = self.x * sin + self.y * cos;
        self.z = -self.w * sin + self.z * cos;
        self.w = wt;
        self.x = xt;
    }

    pub fn rotate(self, v: &Vector3d<f32>) -> Vector3d<f32> {
        let x2: f32 = self.x * self.x;
        let y2: f32 = self.y * self.y;
        let z2: f32 = self.z * self.z;
        Vector3d::<f32> {
            x: 2.0
                * (v.x * (0.5 - y2 - z2)
                    + v.y * (self.x * self.y - self.w * self.z)
                    + v.z * (self.w * self.y + self.x * self.z)),
            y: 2.0
                * (v.x * (self.w * self.z + self.x * self.y)
                    + v.y * (0.5 - x2 - z2)
                    + v.z * (self.y * self.z - self.w * self.x)),
            z: 2.0
                * (v.x * (self.x * self.z - self.w * self.y)
                    + v.y * (self.w * self.x + self.y * self.z)
                    + v.z * (0.5 - x2 - y2)),
        }
    }

    pub fn calculate_roll_radians(self) -> f32 {
        (self.w * self.x + self.y * self.z).atan2(0.5 - self.x * self.x - self.y * self.y)
    }

    pub fn calculate_pitch_radians(self) -> f32 {
        (2.0 * (self.w * self.y - self.x * self.z)).asin()
    }

    pub fn calculate_yaw_radians(self) -> f32 {
        (self.w * self.z + self.x * self.y).atan2(0.5 - self.y * self.y - self.z * self.z)
    }

    pub fn calculate_roll_degrees(self) -> f32 {
        self.calculate_roll_radians().to_degrees()
    }

    pub fn calculate_pitch_degrees(self) -> f32 {
        self.calculate_pitch_radians().to_degrees()
    }

    pub fn calculate_yaw_degrees(self) -> f32 {
        self.calculate_yaw_radians().to_degrees()
    }

    pub fn sin_roll(self) -> f32 {
        let a: f32 = self.w * self.x + self.y * self.z;
        let b: f32 = 0.5 - self.x * self.x - self.y * self.y;
        a * (a * a + b * b).reciprocal_sqrt()
    }

    /// clip sin(roll_angle) to +/-1.0 when roll angle outside range [-90 degrees, 90 degrees]
    pub fn sin_roll_clipped(self) -> f32 {
        let a: f32 = self.w * self.x + self.y * self.z;
        let b: f32 = 0.5 - self.x * self.x - self.y * self.y;
        if b < 0.0 {
            if a < 0.0 {
                return -1.0;
            }
            return 1.0;
        }
        a * (a * a + b * b).reciprocal_sqrt()
    }

    pub fn cos_roll(self) -> f32 {
        let a: f32 = self.w * self.x + self.y * self.z;
        let b: f32 = 0.5 - self.x * self.x - self.y * self.y;
        b * (a * a + b * b).reciprocal_sqrt()
    }

    pub fn sin_pitch(self) -> f32 {
        2.0 * (self.w * self.y - self.x * self.z)
    }

    pub fn cos_pitch(self) -> f32 {
        let s: f32 = self.sin_pitch();
        (1.0_f32 - s * s).sqrt()
    }

    pub fn tan_pitch(self) -> f32 {
        let s: f32 = self.sin_pitch();
        s * (1.0_f32 - s * s).reciprocal_sqrt()
    }

    pub fn cos_yaw(self) -> f32 {
        let a: f32 = self.w * self.z + self.x * self.y;
        let b: f32 = 0.5 - self.y * self.y - self.z * self.z;
        b * (a * a + b * b).reciprocal_sqrt()
    }

    pub fn sin_yaw(self) -> f32 {
        let a: f32 = self.w * self.z + self.x * self.y;
        let b: f32 = 0.5 - self.y * self.y - self.z * self.z;
        a * (a * a + b * b).reciprocal_sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}

    #[test]
    fn normal_types() {
        is_normal::<Quaternion>();
    }
    #[test]
    fn default() {
        let a: Quaternion = Quaternion::default();
        assert_eq!(
            a,
            Quaternion {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        );
        assert!(a.is_one());
        let z = Quaternion::zero();
        assert!(z.is_zero());
        let i = Quaternion::one();
        assert!(i.is_one());
    }
    #[test]
    fn from() {
        let a = Quaternion::from((0.0, 0.0, 0.0));
        let b = Quaternion::from_roll_pitch_yaw_angles_radians(0.0, 0.0, 0.0);
        assert_eq!(a, b);
        let c = Quaternion::from((0.0, 0.0));
        let d = Quaternion::from_roll_pitch_angles_radians(0.0, 0.0);
        assert_eq!(c, d);
    }
    #[test]
    fn neg() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        assert_eq!(
            -a,
            Quaternion {
                w: -2.0,
                x: -3.0,
                y: -5.0,
                z: -7.0,
            }
        );

        let b = -a;
        assert_eq!(
            b,
            Quaternion {
                w: -2.0,
                x: -3.0,
                y: -5.0,
                z: -7.0,
            }
        );
    }
    #[test]
    fn add() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let b = Quaternion {
            w: 11.0,
            x: 13.0,
            y: 17.0,
            z: 19.0,
        };
        assert_eq!(
            a + b,
            Quaternion {
                w: 13.0,
                x: 16.0,
                y: 22.0,
                z: 26.0
            }
        );
    }
    #[test]
    fn add_assign() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let b = Quaternion {
            w: 11.0,
            x: 13.0,
            y: 17.0,
            z: 19.0,
        };
        let mut c = a;
        c += b;
        assert_eq!(
            c,
            Quaternion {
                w: 13.0,
                x: 16.0,
                y: 22.0,
                z: 26.0
            }
        );
    }
    #[test]
    fn sub() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let b = Quaternion {
            w: 11.0,
            x: 13.0,
            y: 17.0,
            z: 23.0,
        };
        let c = a - b;
        assert_eq!(
            c,
            Quaternion {
                w: -9.0,
                x: -10.0,
                y: -12.0,
                z: -16.0,
            }
        );
    }
    #[test]
    fn sub_assign() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let b = Quaternion {
            w: 11.0,
            x: 13.0,
            y: 17.0,
            z: 23.0,
        };
        let mut c = a;
        c -= b;
        assert_eq!(
            c,
            Quaternion {
                w: -9.0,
                x: -10.0,
                y: -12.0,
                z: -16.0
            }
        );
    }
    #[test]
    fn mul() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        assert_eq!(
            a * 2.0,
            Quaternion {
                w: 4.0,
                x: 6.0,
                y: 10.0,
                z: 14.0
            }
        );
        assert_eq!(
            2.0 * a,
            Quaternion {
                w: 4.0,
                x: 6.0,
                y: 10.0,
                z: 14.0
            }
        );
    }
    #[test]
    fn mul_assign() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let mut b = a;
        b *= 2.0;
        assert_eq!(
            b,
            Quaternion {
                w: 4.0,
                x: 6.0,
                y: 10.0,
                z: 14.0,
            }
        );
    }
    #[test]
    fn div() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        assert_eq!(
            a / 2.0,
            Quaternion {
                w: 1.0,
                x: 1.5,
                y: 2.5,
                z: 3.5,
            }
        );
    }
    #[test]
    fn div_assign() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let mut b = a;
        b /= 2.0;
        assert_eq!(
            b,
            Quaternion {
                w: 1.0,
                x: 1.5,
                y: 2.5,
                z: 3.5,
            }
        );
    }
    #[test]
    fn new() {
        let a = Quaternion::new(2.0, 3.0, 5.0, 7.0);
        assert_eq!(
            a,
            Quaternion {
                w: 2.0,
                x: 3.0,
                y: 5.0,
                z: 7.0,
            }
        );
    }
    #[test]
    fn squared_norm() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        assert_eq!(a.squared_norm(), 87.0);
    }
    #[test]
    fn norm() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        assert_eq!(a.norm(), 87.0_f32.sqrt());
        let z = Quaternion {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(z.norm(), 0.0);
    }
    #[test]
    fn normalized() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let b = a / 87.0_f32.sqrt();
        assert_eq!(a.normalized(), b);
        let z = Quaternion {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(z.normalized(), z);
    }
    #[test]
    fn normalize() {
        let a = Quaternion {
            w: 2.0,
            x: 3.0,
            y: 5.0,
            z: 7.0,
        };
        let a_normalized = a.normalized();
        let mut b = a;
        b.normalize();
        assert_eq!(b, a_normalized);
        let z = Quaternion {
            w: 0.0,
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
        let a = Quaternion {
            w: -2.0,
            x: 3.0,
            y: -5.0,
            z: -7.0,
        };
        assert_eq!(
            a.abs(),
            Quaternion {
                w: 2.0,
                x: 3.0,
                y: 5.0,
                z: 7.0
            }
        );
    }
    #[test]
    fn abs_in_place() {
        let a = Quaternion {
            w: -2.0,
            x: -3.0,
            y: 5.0,
            z: 7.0,
        };
        let mut b = a;
        b.abs_in_place();
        assert_eq!(b, a.abs());
    }
    #[test]
    fn clamp() {
        let a = Quaternion {
            w: -5.0,
            x: -2.0,
            y: 3.0,
            z: 5.0,
        };
        assert_eq!(
            a.clamp(-1.0, 4.0),
            Quaternion {
                w: -1.0,
                x: -1.0,
                y: 3.0,
                z: 4.0
            }
        );
    }
    #[test]
    fn clamp_in_place() {
        let a = Quaternion {
            w: -5.0,
            x: -2.0,
            y: 3.0,
            z: 5.0,
        };
        let mut b = a;
        b.clamp_in_place(-1.0, 4.0);
        assert_eq!(b, a.clamp(-1.0, 4.0));
    }
}
