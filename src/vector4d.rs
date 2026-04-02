use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{One, Signed, Zero, float::FloatCore};

use crate::{SqrtMethods, Vector2d, Vector3d, Vector4dMath};

/// 3-dimensional `{x, y, z}` vector of `f32` values
pub type Vector4df32 = Vector4d<f32>;
/// 3-dimensional `{x, y, z}` vector of `f64` values
pub type Vector4df64 = Vector4d<f64>;

// **** Define ****
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector4d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub t: T,
}

// **** Zero ****
/// Zero vector
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// # use num_traits::zero;
/// let z: Vector4df32 = zero();
///
/// assert_eq!(z, Vector4df32 { x: 0.0, y: 0.0, z: 0.0, t: 0.0 });
/// ```
impl<T> Zero for Vector4d<T>
where
    T: Zero + PartialEq + Vector4dMath,
{
    fn zero() -> Self {
        Self { x: T::zero(), y: T::zero(), z: T::zero(), t: T::zero() }
    }

    fn is_zero(&self) -> bool {
        self.x == T::zero() && self.y == T::zero() && self.z == T::zero() && self.z == T::zero()
    }
}

// **** Neg ****
/// Negate vector
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 };
/// let r = -v;
///
/// assert_eq!(r, Vector4df32 { x: -2.0, y: -3.0, z: -5.0, t: -7.0 });
/// ```
impl<T> Neg for Vector4d<T>
where
    T: Vector4dMath,
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        T::v4_neg(self)
    }
}

// **** Add ****
/// Add two vectors
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let u = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let v = Vector4df32::new(11.0, 13.0, 17.0, 19.0);
/// let r = u + v;
///
/// assert_eq!(r, Vector4df32 { x: 13.0, y: 16.0, z: 22.0, t: 26.0 });
/// ```
impl<T> Add for Vector4d<T>
where
    T: Vector4dMath,
{
    type Output = Vector4d<T>;
    fn add(self, other: Self) -> Self {
        T::v4_add(self, other)
    }
}

// **** AddAssign ****
/// Add one vector to another
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let mut r = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let u = Vector4df32::new(11.0, 13.0, 17.0, 19.0);
/// r += u;
///
/// assert_eq!(r, Vector4df32 { x: 13.0, y: 16.0, z: 22.0, t: 26.0 });
///
/// # use num_traits::zero;
/// let z: Vector4df32 = zero();
/// let r = u + z;
/// assert_eq!(r, u);
/// ```
impl<T> AddAssign for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** Sub ****
/// Subtract two vectors
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let u = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let v = Vector4df32::new(11.0, 13.0, 17.0, 23.0);
/// let r = u - v;
///
/// assert_eq!(r, Vector4df32 { x: -9.0, y: -10.0, z: -12.0, t: -16.0 });
/// ```
impl<T> Sub for Vector4d<T>
where
    T: Add<Output = T> + Vector4dMath,
{
    type Output = Vector4d<T>;
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****
/// Subtract one vector from another
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let mut r = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let     v = Vector4df32::new(11.0, 13.0, 17.0, 23.0);
/// r -= v;
///
/// assert_eq!(r, Vector4df32 { x: -9.0, y: -10.0, z: -12.0, t: -16.0 });
/// ```
impl<T> SubAssign for Vector4d<T>
where
    T: Copy + Add<Output = T> + Vector4dMath,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Mul Scalar ****
/// Pre-multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let r = 2.0 * v;
///
/// assert_eq!(r, Vector4df32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
/// ```
impl Mul<Vector4d<f32>> for f32 {
    type Output = Vector4d<f32>;
    fn mul(self, other: Vector4d<f32>) -> Vector4d<f32> {
        Vector4d { x: self * other.x, y: self * other.y, z: self * other.z, t: self * other.t }
    }
}

impl Mul<Vector4d<f64>> for f64 {
    type Output = Vector4d<f64>;
    fn mul(self, other: Vector4d<f64>) -> Vector4d<f64> {
        Vector4d { x: self * other.x, y: self * other.y, z: self * other.z, t: self * other.t }
    }
}

// **** Mul ****
/// Multiply vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let r = v * 2.0;
///
/// assert_eq!(r, Vector4df32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
/// ```
impl<T> Mul<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;
    fn mul(self, k: T) -> Self {
        T::v4_mul_scalar(self, k)
    }
}

// **** MulAssign ****
/// In-place multiply a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let mut v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// v *= 2.0;
///
/// assert_eq!(v, Vector4df32 { x: 4.0, y: 6.0, z: 10.0, t: 14.0 });
/// ```
impl<T> MulAssign<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    fn mul_assign(&mut self, k: T) {
        *self = *self * k;
    }
}

// **** Div by scalar ****
/// Divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// let r = v / 2.0;
///
/// assert_eq!(r, Vector4df32 { x: 1.0, y: 1.5, z: 2.5, t: 3.5 });
/// ```
impl<T> Div<T> for Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    type Output = Self;
    fn div(self, k: T) -> Self {
        T::v4_div_scalar(self, k)
    }
}

/// In-place divide a vector by a constant
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let mut v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// v /= 2.0;
///
/// assert_eq!(v, Vector4df32 { x: 1.0, y: 1.5, z: 2.5, t: 3.5 });
/// ```
impl<T> DivAssign<T> for Vector4d<T>
where
    T: Copy + Div<Output = T> + Vector4dMath,
{
    fn div_assign(&mut self, k: T) {
        *self = self.div(k);
    }
}

// **** Index ****
/// Access vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
///
/// assert_eq!(v[0], 2.0);
/// assert_eq!(v[1], 3.0);
/// assert_eq!(v[2], 5.0);
/// assert_eq!(v[3], 7.0);
/// ```
impl<T> Index<usize> for Vector4d<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.t, // default to t component if index out of range
        }
    }
}

// **** IndexMut ****
// Set vector component by index
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let mut v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
/// v[0] = 7.0;
/// v[1] = 11.0;
/// v[2] = 13.0;
/// v[3] = 17.0;
///
/// assert_eq!(v, Vector4df32 { x:7.0, y:11.0, z:13.0, t: 17.0 });
/// ```
impl<T> IndexMut<usize> for Vector4d<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.t,
            _ => &mut self.t, // default to t component if index out of range
        }
    }
}

// **** impl new ****
impl<T> Vector4d<T>
where
    T: Copy,
{
    /// Create a vector
    pub const fn new(x: T, y: T, z: T, t: T) -> Self {
        Self { x, y, z, t }
    }
}

// **** impl abs ****
impl<T> Vector4d<T>
where
    T: Copy + Signed,
{
    /// Return a copy of the vector with all components set to their absolute values
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs(), t: self.t.abs() }
    }

    /// Set all components of the vector to their absolute values
    pub fn abs_in_place(&mut self) {
        *self = self.abs();
    }
}

// **** impl clamp ****
impl<T> Vector4d<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the vector with all components clamped to the specified range
    pub fn clamp(self, min: T, max: T) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
            t: self.t.clamp(min, max),
        }
    }

    /// Clamp all components of the vector to the specified range
    pub fn clamp_in_place(&mut self, min: T, max: T) {
        self.x = self.x.clamp(min, max);
        self.y = self.y.clamp(min, max);
        self.z = self.z.clamp(min, max);
    }
}

// **** impl dot ****
impl<T> Vector4d<T>
where
    T: Vector4dMath + Copy,
{
    /// Vector dot product
    /// ```
    /// # use vector_quaternion_matrix::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4df32::new(11.0, 13.0, 17.0, 19.0);
    ///
    /// let x = v.dot(w);
    ///
    /// assert_eq!(x, 279.0);
    /// ```
    #[inline(always)]
    pub fn dot(self, other: Self) -> T {
        T::v4_dot(self, other)
    }
}

// **** impl norm_squared ****
impl<T> Vector4d<T>
where
    T: Copy + Add<Output = T> + Vector4dMath + Vector4dMath,
{
    /// Return square of Euclidean norm
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    /// Return distance between two points, squared
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

impl<T> Vector4d<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Return the sum of all components of the vector
    pub fn sum(self) -> T {
        self.x + self.y + self.z
    }

    /// Return the product of all components of the vector
    pub fn product(self) -> T {
        self.x * self.y * self.z
    }
}

// **** impl mean ****
impl<T> Vector4d<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T>,
{
    /// Return the mean of all components of the vector
    pub fn mean(self) -> T {
        let three = T::one() + T::one() + T::one();
        (self.x + self.y + self.z) / three
    }
}

impl<T> Vector4d<T>
where
    T: Copy + Vector4dMath,
{
    /// Return the max element in the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4df32::new(3.0, 5.0, 7.0, 2.0);
    /// let x = Vector4df32::new(5.0, 7.0, 3.0, 2.0);
    /// assert_eq!(7.0, v.max());
    /// assert_eq!(7.0, w.max());
    /// assert_eq!(7.0, x.max());
    /// ```
    pub fn max(self) -> T {
        T::v4_max(self)
    }

    /// Return the max element in the vector
    /// ```
    /// # use vector_quaternion_matrix::Vector4df32;
    /// let v = Vector4df32::new(2.0, 3.0, 5.0, 7.0);
    /// let w = Vector4df32::new(3.0, 5.0, 7.0, 2.0);
    /// let x = Vector4df32::new(5.0, 7.0, 3.0, 2.0);
    /// assert_eq!(2.0, v.min());
    /// assert_eq!(2.0, w.min());
    /// assert_eq!(2.0, x.min());
    /// ```
    pub fn min(self) -> T {
        T::v4_min(self)
    }
}

// **** impl norm ****
impl<T> Vector4d<T>
where
    T: Copy + Add<Output = T> + SqrtMethods + Vector4dMath + Vector4dMath,
{
    /// Return Euclidean norm
    pub fn norm(self) -> T {
        Self::norm_squared(self).sqrt()
    }
}

impl<T> Vector4d<T>
where
    T: Copy + Zero + PartialEq + SqrtMethods + Vector4dMath + Vector4dMath,
{
    /// Return normalized form of the vector
    pub fn normalized(self) -> Self {
        let norm = self.norm();
        // If norm == 0.0 then the vector is already normalized
        if norm == T::zero() {
            return self;
        }
        self * T::v4_reciprocal(norm)
    }

    /// Normalize the vector in place
    pub fn normalize(&mut self) -> Self {
        let norm = self.norm();
        //#[allow(clippy::assign_op_pattern)]
        // If norm == 0.0 then the vector is already normalized
        if norm != T::zero() {
            *self *= T::v4_reciprocal(norm);
        }
        *self
    }
}

impl<T> Vector4d<T>
where
    T: Copy + Zero + SqrtMethods + Vector4dMath + Vector4dMath,
{
    // Return distance between two points
    pub fn distance(self, other: Self) -> T {
        self.distance_squared(other).sqrt()
    }
}

// **** From ****
/// Vector4d from Vector2d
/// ```
/// # use vector_quaternion_matrix::{Vector2df32,Vector4df32};
/// let v = Vector4df32::from(Vector2df32 { x: 2.0, y: 3.0 });
/// let w: Vector4df32 = Vector2df32 { x: 7.0, y: 11.0 }.into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 0.0, t: 0.0 });
/// assert_eq!(w, Vector4df32 { x: 7.0, y: 11.0, z: 0.0, t: 0.0 });
/// ```
impl<T> From<Vector2d<T>> for Vector4d<T>
where
    T: Zero,
{
    fn from(other: Vector2d<T>) -> Self {
        Self { x: other.x, y: other.y, z: T::zero(), t: T::zero() }
    }
}

// **** From ****
/// Vector4d from Vector3d
/// ```
/// # use vector_quaternion_matrix::{Vector3df32,Vector4df32};
/// let v = Vector4df32::from(Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
/// let w: Vector4df32 = Vector3df32 { x: 7.0, y: 11.0, z: 13.0 }.into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 0.0 });
/// assert_eq!(w, Vector4df32 { x: 7.0, y: 11.0, z: 13.0, t: 0.0 });
/// ```
impl<T> From<Vector3d<T>> for Vector4d<T>
where
    T: Zero,
{
    fn from(other: Vector3d<T>) -> Self {
        Self { x: other.x, y: other.y, z: other.z, t: T::zero() }
    }
}

// **** From Tuple ****
/// Vector from tuple
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32::from((2.0, 3.0, 5.0, 7.0));
/// let w: Vector4df32 = (11.0, 13.0, 17.0, 19.0).into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 });
/// assert_eq!(w, Vector4df32 { x: 11.0, y: 13.0, z: 17.0, t: 19.0 });
/// ```
impl<T> From<(T, T, T, T)> for Vector4d<T> {
    fn from((x, y, z, t): (T, T, T, T)) -> Self {
        Self { x, y, z, t }
    }
}

/// Vector from array
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32::from([2.0, 3.0, 5.0, 7.0]);
/// let w: Vector4df32 = [11.0, 13.0, 17.0, 19.0].into();
///
/// assert_eq!(v, Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 });
/// assert_eq!(w, Vector4df32 { x: 11.0, y: 13.0, z: 17.0, t: 19.0 });
/// ```
impl<T> From<[T; 4]> for Vector4d<T>
where
    T: Copy,
{
    fn from(v: [T; 4]) -> Self {
        Self { x: v[0], y: v[1], z: v[2], t: v[3] }
    }
}

/// Array from vector
/// ```
/// # use vector_quaternion_matrix::Vector4df32;
/// let v = Vector4df32 { x: 2.0, y: 3.0, z: 5.0, t: 7.0 };
///
/// let a = <[f32; 4]>::from(v);
/// let b: [f32; 4] = v.into();
///
/// assert_eq!(a, [2.0, 3.0, 5.0, 7.0]);
/// assert_eq!(b, [2.0, 3.0, 5.0, 7.0]);
/// ```
impl<T> From<Vector4d<T>> for [T; 4] {
    fn from(v: Vector4d<T>) -> Self {
        [v.x, v.y, v.z, v.t]
    }
}
