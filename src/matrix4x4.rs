#![allow(unused)]
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

use crate::{MathConstants, Matrix2x2, Matrix4x4Math, Quaternion, QuaternionMath, SqrtMethods, Vector3d, Vector4d};

/// 4x4 matrix of `f32` values<br>
pub type Matrix4x4f32 = Matrix4x4<f32>;
/// 4x4 matrix of `f64` values<br><br>
pub type Matrix4x4f64 = Matrix4x4<f64>;

// **** Define ****

/// `Matrix4x4<T>`: 4x4 Matrix of type `T`.<br>
/// Aliases `Matrix4x4f32` and `Matrix4x4f64` are provided.<br>
/// Internal implementation is a flattened 4x4 matrix: an array of 9 elements stored in row-major order<br>
/// That is the element `m[row][col]` is at array position `[row * 3 + col]`, so element `m12` is at `a[5]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix4x4<T> {
    // Flattened 4x4 matrix: 16 elements in row-major order
    pub(crate) a: [T; 16],
}

// **** New ****

/// Create a matrix.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
///                              5.0, 11.0, 47.0, 109.0,
///                             23.0, 31.0, 41.0, 103.0,
///                             67.0, 73.0, 83.0,  97.0]);
/// assert_eq!(m, Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                     5.0, 11.0, 47.0, 109.0,
///                                    23.0, 31.0, 41.0, 103.0,
///                                    67.0, 73.0, 83.0,  97.0]));
/// ```
impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Create a matrix.
    #[inline]
    pub const fn new(input: [T; 16]) -> Self {
        Self { a: input }
    }
}

// **** Zero ****

/// Zero matrix.
/// ```
/// # use vqm::Matrix4x4f32;
/// # use num_traits::Zero;
/// let z = Matrix4x4f32::zero();
///
/// assert_eq!(z, Matrix4x4f32::from([ 0.0, 0.0, 0.0, 0.0,
///                                    0.0, 0.0, 0.0, 0.0,
///                                    0.0, 0.0, 0.0, 0.0,
///                                    0.0, 0.0, 0.0, 0.0]));
/// ```
impl<T> Zero for Matrix4x4<T>
where
    T: Copy + Zero + PartialEq + Matrix4x4Math,
{
    #[inline]
    fn zero() -> Self {
        Self {
            a: [
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(), //
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(), //
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(), //
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(),
            ], //
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.a.iter().all(|&x| x == T::zero())
    }
}

// **** One ****

/// Identity matrix.
/// ```
/// # use vqm::Matrix4x4f32;
/// # use num_traits::One;
/// let i = Matrix4x4f32::one();
///
/// assert_eq!(i, Matrix4x4f32::from([ 1.0, 0.0, 0.0, 0.0,
///                                    0.0, 1.0, 0.0, 0.0,
///                                    0.0, 0.0, 1.0, 0.0,
///                                    0.0, 0.0, 0.0, 1.0]));
/// ```
impl<T> One for Matrix4x4<T>
where
    T: Copy + Zero + One + PartialEq + Matrix4x4Math,
{
    #[inline]
    fn one() -> Self {
        Self {
        a :   [ T::one(),  T::zero(), T::zero(), T::zero(), //
                T::zero(), T::one(),  T::zero(), T::zero(), //
                T::zero(), T::zero(), T::one(),  T::zero(), //
                T::zero(), T::zero(), T::zero(), T::one()] //
        }
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.a == [ T::one(), T::zero(), T::zero(), T::zero(), //
                    T::zero(), T::one(), T::zero(), T::zero(), //
                    T::zero(), T::zero(), T::one(), T::zero(), //
                    T::zero(), T::zero(), T::zero(), T::one()] //
    }
}

// **** Neg ****

/// Negate matrix.
/// ```
/// # use vqm::Matrix4x4f32;
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
/// m = - m;
///
/// assert_eq!(m, Matrix4x4f32::from([ -2.0, -17.0, -59.0, -127.0,
///                                    -5.0, -11.0, -47.0, -109.0,
///                                   -23.0, -31.0, -41.0, -103.0,
///                                   -67.0, -73.0, -83.0,  -97.0]));
/// ```
impl<T> Neg for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        T::m4x4_neg(self)
    }
}

// **** Add ****

/// Add two matrices.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
/// let n = Matrix4x4f32::from([  3.0, 19.0, 61.0, 131.0,
///                               7.0, 13.0, 53.0, 113.0,
///                              29.0, 37.0, 43.0, 107.0,
///                              71.0, 79.0, 89.0, 101.0]);
/// let r = m + n;
/// assert_eq!(r, Matrix4x4f32::from([  5.0, 36.0, 120.0, 258.0,
///                                    12.0, 24.0, 100.0, 222.0,
///                                    52.0, 68.0,  84.0, 210.0,
///                                   138.0,152.0, 172.0, 198.0]));
///
/// # use num_traits::Zero;
///
/// let z = Matrix4x4f32::zero();
/// let r2 = m + z;
///
/// assert_eq!(r2, m);
/// ```
impl<T> Add for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        T::m4x4_add(self, other)
    }
}


// **** AddAssign ****

/// Add one matrix to another.
/// ```
/// # use vqm::Matrix4x4f32;
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
/// let n = Matrix4x4f32::from([  3.0, 19.0, 61.0, 131.0,
///                               7.0, 13.0, 53.0, 113.0,
///                              29.0, 37.0, 43.0, 107.0,
///                              71.0, 79.0, 89.0, 101.0]);
/// m += n;
///
/// assert_eq!(m, Matrix4x4f32::from([  5.0, 36.0, 120.0, 258.0,
///                                    12.0, 24.0, 100.0, 222.0,
///                                    52.0, 68.0,  84.0, 210.0,
///                                   138.0,152.0, 172.0, 198.0]));
/// ```
impl<T> AddAssign for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

/// Multiply vector by constant and add another vector.
/// ```
/// # use vqm::Matrix4x4f32;
/// # use num_traits::MulAdd;
/// let m = Matrix4x4f32::from([ 2.0, 17.0, 59.0, 127.0,
///                              5.0, 11.0, 47.0, 109.0,
///                             23.0, 31.0, 41.0, 103.0,
///                             67.0, 73.0, 83.0,  97.0]);
/// let n = Matrix4x4f32::from([ 3.0, 19.0, 61.0, 131.0,
///                              7.0, 13.0, 53.0, 113.0,
///                             29.0, 37.0, 43.0, 107.0,
///                             71.0, 79.0, 89.0, 101.0]);
/// let k = 137.0;
/// let r = m.mul_add(k, n);
/// //assert_eq!(r, Matrix4x4f32::from([  277.0, 2348.0, 8144.0,
///   //                                  692.0, 1520.0, 6492.0,
///     //                               3180.0, 4284.0, 5660.0]));
/// ```
impl<T> MulAdd<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m4x4_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

/// Multiply vector by constant and add another vector in place.
/// ```
/// # use vqm::Matrix4x4f32;
/// # use num_traits::MulAddAssign;
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
/// let n = Matrix4x4f32::from([  3.0, 19.0, 61.0, 131.0,
///                               7.0, 13.0, 53.0, 113.0,
///                              29.0, 37.0, 43.0, 107.0,
///                              71.0, 79.0, 89.0, 101.0]);
/// let k = 137.0;
/// m.mul_add_assign(k, n);
///
/// //assert_eq!(m, Matrix3x3f32::from([  277.0, 2348.0, 8144.0,
///   //                                  692.0, 1520.0, 6492.0,
///     //                               3180.0, 4284.0, 5660.0]));
/// ```
impl<T> MulAddAssign<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

/// Subtract two matrices.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
///
/// let n = Matrix4x4f32::from([  3.0, 13.0, 61.0, 131.0,
///                               7.0, 19.0, 53.0, 113.0,
///                              29.0, 37.0, 43.0, 107.0,
///                              71.0, 79.0, 89.0, 101.0]);
/// let r = m - n;
///
/// assert_eq!(r, Matrix4x4f32::from([  -1.0,  4.0, -2.0, -4.0,
///                                     -2.0, -8.0, -6.0, -4.0,
///                                     -6.0, -6.0, -2.0, -4.0,
///                                     -4.0, -6.0, -6.0, -4.0]));
/// ```
impl<T> Sub for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

/// Subtract one matrix from another.
/// ```
/// # use vqm::Matrix4x4f32;
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
/// let n = Matrix4x4f32::from([  3.0, 13.0, 43.0, 131.0,
///                               7.0, 19.0, 37.0, 113.0,
///                              29.0, 61.0, 53.0, 107.0,
///                              71.0, 79.0, 89.0, 101.0]);
/// m -= n;
///
/// ```
impl<T> SubAssign for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

/// Pre-multiply a matrix by a constant.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
/// let r = 2.0 * m;
///
/// assert_eq!(r, Matrix4x4f32::from([  4.0, 34.0, 118.0, 254.0,
///                                    10.0, 22.0,  94.0, 218.0,
///                                    46.0, 62.0,  82.0, 206.0,
///                                   134.0,146.0, 166.0, 194.0]));
/// ```
impl Mul<Matrix4x4<f32>> for f32 {
    type Output = Matrix4x4<f32>;
    #[inline]
    fn mul(self, other: Matrix4x4<f32>) -> Matrix4x4<f32> {
        f32::m4x4_mul_scalar(other, self)
    }
}

impl Mul<Matrix4x4<f64>> for f64 {
    type Output = Matrix4x4<f64>;
    #[inline]
    fn mul(self, other: Matrix4x4<f64>) -> Matrix4x4<f64> {
        f64::m4x4_mul_scalar(other, self)
    }
}
// **** Mul ****

/// Multiply a matrix by a constant.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([ 2.0, 17.0, 59.0, 127.0,
///                              5.0, 11.0, 47.0, 109.0,
///                             23.0, 31.0, 41.0, 103.0,
///                             67.0, 73.0, 83.0,  97.0]);
/// let r = m * 2.0;
///
/// ```
impl<T> Mul<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: T) -> Self {
        T::m4x4_mul_scalar(self, other)
    }
}

// **** MulAssign ****

/// In-place multiply a matrix by a constant.
/// ```
/// # use vqm::Matrix4x4f32;
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
/// m *= 2.0;
///
/// ```
impl<T> MulAssign<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

/// Multiply a vector by a matrix.
/// ```
/// # use vqm::Matrix4x4f32;
/// # use vqm::Vector4df32;
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
/// let v = Vector4df32{x:3.0, y:7.0, z:13.0, t:17.0};
/// let r = m * v;
/// assert_eq!(r, Vector4df32{x:3051.0, y:2556.0, z:2570.0, t:3440.0});
/// ```
impl<T> Mul<Vector4d<T>> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Vector4d<T>;
    #[inline]
    fn mul(self, other: Vector4d<T>) -> Vector4d<T> {
        T::m4x4_mul_vector(self, other)
    }
}

/// Pre-multiply a vector by a matrix.
/// ```
/// # use vqm::{Matrix4x4f32,Vector4df32};
/// let m = Matrix4x4f32::from([  2.0,   3.0,   5.0,   7.0,
///                              11.0,  13.0,  17.0,  19.0,
///                              23.0,  29.0,  31.0,  37.0,
///                              41.0,  43.0,  47.0,  53.0]);
/// let v = Vector4df32{x:3.0, y:7.0,  z:13.0, t:17.0};
/// let r = v * m;
///
/// //assert_eq!(r, Vector4df32{x:59.0*2.0 + 61.0*11.0 + 67.0*23.0 + 71.0*41.0,
///   //                        y:59.0*3.0 + 61.0*13.0 + 67.0*29.0 + 71.0*43.0,
///     //                      z:59.0*5.0 + 61.0*17.0 + 67.0*31.0 + 71.0*47.0,
///       //                    t:59.0*7.0 + 61.0*19.0 + 67.0*37.0 + 71.0*53.0});
/// ```
impl<T> Mul<Matrix4x4<T>> for Vector4d<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Matrix4x4<T>) -> Self {
        T::m4x4_vector_mul(self, other)
    }
}
/// Multiply two matrices.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
///
/// let n = Matrix4x4f32::from([  3.0, 19.0, 61.0, 131.0,
///                               7.0, 13.0, 53.0, 113.0,
///                              29.0, 37.0, 43.0, 107.0,
///                              71.0, 79.0, 89.0, 101.0]);
/// let r = m * n;
///
/// assert_eq!(r, Matrix4x4f32::from([
///    2.0*  3.0 + 17.0*  7.0 + 59.0* 29.0 + 127.0* 71.0,   
///    2.0* 19.0 + 17.0* 13.0 + 59.0* 37.0 + 127.0* 79.0,
///    2.0* 61.0 + 17.0* 53.0 + 59.0* 43.0 + 127.0* 89.0,
///    2.0*131.0 + 17.0*113.0 + 59.0*107.0 + 127.0*101.0,
///
///    5.0*  3.0 + 11.0*  7.0 + 47.0* 29.0 + 109.0* 71.0,
///    5.0* 19.0 + 11.0* 13.0 + 47.0* 37.0 + 109.0* 79.0,
///    5.0* 61.0 + 11.0* 53.0 + 47.0* 43.0 + 109.0* 89.0,
///    5.0*131.0 + 11.0*113.0 + 47.0*107.0 + 109.0*101.0,
///
///   23.0*  3.0 + 31.0*  7.0 + 41.0* 29.0 + 103.0* 71.0,
///   23.0* 19.0 + 31.0* 13.0 + 41.0* 37.0 + 103.0* 79.0,
///   23.0* 61.0 + 31.0* 53.0 + 41.0* 43.0 + 103.0* 89.0,
///   23.0*131.0 + 31.0*113.0 + 41.0*107.0 + 103.0*101.0,
/// 
///   67.0*  3.0 + 73.0*  7.0 + 83.0* 29.0 +  97.0* 71.0,
///   67.0* 19.0 + 73.0* 13.0 + 83.0* 37.0 +  97.0* 79.0,
///   67.0* 61.0 + 73.0* 53.0 + 83.0* 43.0 +  97.0* 89.0,
///   67.0*131.0 + 73.0*113.0 + 83.0*107.0 +  97.0*101.0,
/// ]));
///
/// # use num_traits::{One,one};
///
/// let i = Matrix4x4f32::one();
/// let r2 = m * i;
///
/// assert_eq!(r2, m);
/// ```
impl<T> Mul<Matrix4x4<T>> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        T::m4x4_mul(self, other)
    }
}

/// Multiply one matrix by another.
/// ```
/// # use vqm::Matrix4x4f32;
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
/// let n = Matrix4x4f32::from([  3.0, 19.0, 61.0, 131.0,
///                               7.0, 13.0, 53.0, 113.0,
///                              29.0, 37.0, 43.0, 107.0,
///                              71.0, 79.0, 89.0, 101.0]);
/// m *= n;
///
/// assert_eq!(m, Matrix4x4f32::from([
///    2.0*  3.0 + 17.0*  7.0 + 59.0* 29.0 + 127.0* 71.0,   
///    2.0* 19.0 + 17.0* 13.0 + 59.0* 37.0 + 127.0* 79.0,
///    2.0* 61.0 + 17.0* 53.0 + 59.0* 43.0 + 127.0* 89.0,
///    2.0*131.0 + 17.0*113.0 + 59.0*107.0 + 127.0*101.0,
///
///    5.0*  3.0 + 11.0*  7.0 + 47.0* 29.0 + 109.0* 71.0,
///    5.0* 19.0 + 11.0* 13.0 + 47.0* 37.0 + 109.0* 79.0,
///    5.0* 61.0 + 11.0* 53.0 + 47.0* 43.0 + 109.0* 89.0,
///    5.0*131.0 + 11.0*113.0 + 47.0*107.0 + 109.0*101.0,
///
///   23.0*  3.0 + 31.0*  7.0 + 41.0* 29.0 + 103.0* 71.0,
///   23.0* 19.0 + 31.0* 13.0 + 41.0* 37.0 + 103.0* 79.0,
///   23.0* 61.0 + 31.0* 53.0 + 41.0* 43.0 + 103.0* 89.0,
///   23.0*131.0 + 31.0*113.0 + 41.0*107.0 + 103.0*101.0,
/// 
///   67.0*  3.0 + 73.0*  7.0 + 83.0* 29.0 +  97.0* 71.0,
///   67.0* 19.0 + 73.0* 13.0 + 83.0* 37.0 +  97.0* 79.0,
///   67.0* 61.0 + 73.0* 53.0 + 83.0* 43.0 +  97.0* 89.0,
///   67.0*131.0 + 73.0*113.0 + 83.0*107.0 +  97.0*101.0,
/// ]));
/// ```
impl<T> MulAssign<Matrix4x4<T>> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    #[inline]
    fn mul_assign(&mut self, other: Matrix4x4<T>) {
        *self = *self * other;
    }
}


// **** Div ****

/// Divide a matrix by a constant.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
/// let r = m / 2.0;
///
/// assert_eq!(r, Matrix4x4f32::from([ 1.0,  8.5, 29.5, 63.5,
///                                    2.5,  5.5, 23.5, 54.5,
///                                   11.5, 15.5, 20.5, 51.5,
///                                   33.5, 36.5, 41.5, 48.5]));
/// ```
impl<T> Div<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    type Output = Self;

    #[inline]
    fn div(self, other: T) -> Self {
        T::m4x4_div_scalar(self, other)
    }
}

// **** DivAssign ****

/// In-place divide a matrix by a constant.
/// ```
/// # use vqm::Matrix4x4f32;
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
/// m /= 2.0;
///
/// assert_eq!(m, Matrix4x4f32::from([ 1.0,  8.5, 29.5, 63.5,
///                                    2.5,  5.5, 23.5, 54.5,
///                                   11.5, 15.5, 20.5, 51.5,
///                                   33.5, 36.5, 41.5, 48.5]));
/// ```
impl<T> DivAssign<T> for Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** Index ****

/// Access matrix element by index.
/// ```
/// # use vqm::Matrix4x4f32;
///
/// let m = Matrix4x4f32::from([ 2.0, 17.0, 59.0, 127.0,
///                              5.0, 11.0, 47.0, 109.0,
///                             23.0, 31.0, 41.0, 103.0,
///                             67.0, 73.0, 83.0,  97.0]);
///
/// assert_eq!(m[0], 2.0);
/// assert_eq!(m[1], 17.0);
/// assert_eq!(m[2], 59.0);
/// assert_eq!(m[3], 127.0);
/// assert_eq!(m[4], 5.0);
/// assert_eq!(m[5], 11.0);
/// assert_eq!(m[6], 47.0);
/// assert_eq!(m[7], 109.0);
/// assert_eq!(m[8], 23.0);
/// assert_eq!(m[9], 31.0);
/// assert_eq!(m[10], 41.0);
/// assert_eq!(m[11], 103.0);
/// assert_eq!(m[12], 67.0);
/// assert_eq!(m[13], 73.0);
/// assert_eq!(m[14], 83.0);
/// assert_eq!(m[15], 97.0);
/// ```
impl<T> Index<usize> for Matrix4x4<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

// **** IndexMut ****

/// Set matrix element by index.
/// ```
/// # use vqm::Matrix4x4f32;
///
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
///
/// m[0] = 3.0;
/// m[1] = 19.0;
/// m[2] = 61.0;
/// m[3] = 131.0;
/// m[4] = 7.0;
/// m[5] = 13.0;
/// m[6] = 53.0;
/// m[7] = 113.0;
/// m[8] = 29.0;
/// m[9] = 37.0;
/// m[10] = 43.0;
/// m[11] = 107.0;
/// m[12] = 71.0;
/// m[13] = 79.0;
/// m[14] = 89.0;
/// m[15] = 101.0;
///
/// assert_eq!(m, Matrix4x4f32::from([  3.0, 19.0, 61.0, 131.0,
///                                     7.0, 13.0, 53.0, 113.0,
///                                    29.0, 37.0, 43.0, 107.0,
///                                    71.0, 79.0, 89.0, 101.0]));
/// ```
impl<T> IndexMut<usize> for Matrix4x4<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

/// Access matrix element by ordered pair (row, column).
/// ```
/// # use vqm::Matrix4x4f32;
///
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
///
/// assert_eq!(m[(0,0)], 2.0);
/// assert_eq!(m[(0,1)], 17.0);
/// assert_eq!(m[(0,2)], 59.0);
/// assert_eq!(m[(0,3)], 127.0);
/// assert_eq!(m[(1,0)], 5.0);
/// assert_eq!(m[(1,1)], 11.0);
/// assert_eq!(m[(1,2)], 47.0);
/// assert_eq!(m[(1,3)], 109.0);
/// assert_eq!(m[(2,0)], 23.0);
/// assert_eq!(m[(2,1)], 31.0);
/// assert_eq!(m[(2,2)], 41.0);
/// assert_eq!(m[(2,3)], 103.0);
/// assert_eq!(m[(3,0)], 67.0);
/// assert_eq!(m[(3,1)], 73.0);
/// assert_eq!(m[(3,2)], 83.0);
/// assert_eq!(m[(3,3)], 97.0);
/// ```
impl<T> Index<(usize, usize)> for Matrix4x4<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.a[row * 4 + col]
    }
}

/// Set matrix element by ordered pair (row, column).
/// ```
/// # use vqm::Matrix4x4f32;
///
/// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                                   5.0, 11.0, 47.0, 109.0,
///                                  23.0, 31.0, 41.0, 103.0,
///                                  67.0, 73.0, 83.0,  97.0]);
///
/// m[(0,0)] = 3.0;
/// m[(0,1)] = 19.0;
/// m[(0,2)] = 61.0;
/// m[(0,3)] = 131.0;
/// m[(1,0)] = 7.0;
/// m[(1,1)] = 13.0;
/// m[(1,2)] = 53.0;
/// m[(1,3)] = 113.0;
/// m[(2,0)] = 29.0;
/// m[(2,1)] = 37.0;
/// m[(2,2)] = 43.0;
/// m[(2,3)] = 107.0;
/// m[(3,0)] = 71.0;
/// m[(3,1)] = 79.0;
/// m[(3,2)] = 89.0;
/// m[(3,3)] = 101.0;
///
/// assert_eq!(m, Matrix4x4f32::from([  3.0, 19.0, 61.0, 131.0,
///                                     7.0, 13.0, 53.0, 113.0,
///                                    29.0, 37.0, 43.0, 107.0,
///                                    71.0, 79.0, 89.0, 101.0]));
/// ```
impl<T> IndexMut<(usize, usize)> for Matrix4x4<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.a[row * 4 + col]
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    pub fn set_row(&mut self, row: usize, value: Vector4d<T>) {
        match row {
            0 => {
                self.a[0] = value.x;
                self.a[1] = value.y;
                self.a[2] = value.z;
            }
            1 => {
                self.a[3] = value.x;
                self.a[4] = value.y;
                self.a[5] = value.z;
            }
            _ => {
                self.a[6] = value.x;
                self.a[7] = value.y;
                self.a[8] = value.z;
            }
        }
    }

    /// Return matrix row as a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4df32};
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let v = m.row(0);
    ///
    /// assert_eq!(v, Vector4df32{ x: 2.0, y: 17.0, z: 59.0, t:127.0 });
    /// assert_eq!(m.row(1), Vector4df32{ x: 5.0, y: 11.0, z: 47.0, t:109.0 });
    /// assert_eq!(m.row(2), Vector4df32{ x: 23.0, y: 31.0, z: 41.0, t:103.0 });
    /// assert_eq!(m.row(3), Vector4df32{ x: 67.0, y: 73.0, z: 83.0, t:97.0 });
    /// ```
    pub fn row(self, row: usize) -> Vector4d<T> {
        match row {
            0 => Vector4d::<T> { x: self.a[0], y: self.a[1], z: self.a[2], t: self.a[3] },
            1 => Vector4d::<T> { x: self.a[4], y: self.a[5], z: self.a[6], t: self.a[7] },
            2 => Vector4d::<T> { x: self.a[8], y: self.a[9], z: self.a[10], t: self.a[11] },
            // default to last row if row out of range
            _ => Vector4d::<T> { x: self.a[12], y: self.a[13], z: self.a[14], t: self.a[15] },
        }
    }

    pub fn set_column(&mut self, column: usize, value: Vector4d<T>) {
        match column {
            0 => {
                self.a[0] = value.x;
                self.a[4] = value.y;
                self.a[8] = value.z;
                self.a[12] = value.t;
            }
            1 => {
                self.a[1] = value.x;
                self.a[5] = value.y;
                self.a[9] = value.z;
                self.a[13] = value.t;
            }
            2 => {
                self.a[2] = value.x;
                self.a[6] = value.y;
                self.a[10] = value.z;
                self.a[14] = value.t;
            }
            _ => {
                self.a[3] = value.x;
                self.a[7] = value.y;
                self.a[11] = value.z;
                self.a[15] = value.t;
            }
        }
    }

    /// Return matrix column as a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4df32};
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let v = m.column(0);
    ///
    /// assert_eq!(v, Vector4df32{ x: 2.0, y: 5.0, z: 23.0, t: 67.0 });
    /// assert_eq!(m.column(1), Vector4df32{ x: 17.0, y: 11.0, z: 31.0, t: 73.0 });
    /// assert_eq!(m.column(2), Vector4df32{ x: 59.0, y: 47.0, z: 41.0, t: 83.0 });
    /// assert_eq!(m.column(3), Vector4df32{ x: 127.0, y: 109.0, z: 103.0, t: 97.0 });
    /// ```
    pub fn column(self, column: usize) -> Vector4d<T> {
        match column {
            0 => Vector4d::<T> { x: self.a[0], y: self.a[4], z: self.a[8], t: self.a[12] },
            1 => Vector4d::<T> { x: self.a[1], y: self.a[5], z: self.a[9], t: self.a[13] },
            2 => Vector4d::<T> { x: self.a[2], y: self.a[6], z: self.a[10], t: self.a[14] },
            // default to last column if column out of range
            _ => Vector4d::<T> { x: self.a[3], y: self.a[7], z: self.a[11], t: self.a[15] },
        }
    }

    /// Return matrix diagonal as a vector.
    /// ```
    /// # use vqm::{Matrix4x4f32,Vector4df32};
    ///
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let v = m.diagonal();
    ///
    /// assert_eq!(v, Vector4df32{ x: 2.0, y: 11.0, z: 41.0, t: 97.0 });
    /// ```
    pub fn diagonal(self) -> Vector4d<T> {
        Vector4d::<T> { x: self.a[0], y: self.a[5], z: self.a[10], t: self.a[15] }
    }
}

// **** abs ****

impl<T> Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Return a copy of the matrix with all components set to their absolute values.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, -17.0,  59.0,  127.0,
    ///                               5.0, -11.0,  47.0,  109.0,
    ///                              23.0,  31.0, -41.0, -103.0,
    ///                              67.0,  73.0, -83.0,  97.0]);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                                     5.0, 11.0, 47.0, 109.0,
    ///                                    23.0, 31.0, 41.0, 103.0,
    ///                                    67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    pub fn abs(self) -> Self {
        T::m4x4_abs(self)
    }

    /// Set all components of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::from([  2.0, -17.0,  59.0,  127.0,
    ///                                   5.0, -11.0,  47.0,  109.0,
    ///                                  23.0,  31.0,  41.0, -103.0,
    ///                                  67.0,  73.0, -83.0,   97.0]);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                                     5.0, 11.0, 47.0, 109.0,
    ///                                    23.0, 31.0, 41.0, 103.0,
    ///                                    67.0, 73.0, 83.0,  97.0]));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m4x4_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix4x4<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all components clamped to the specified range.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, -59.0, 127.0,
    ///                               5.0, 11.0,  47.0, 109.0,
    ///                              23.0, 31.0, -41.0, 103.0,
    ///                              67.0, 73.0,  83.0,  97.0]);
    /// let n = m.clamp(7.0, 17.0);
    ///
    /// assert_eq!(n, Matrix4x4f32::from([ 7.0, 17.0,  7.0, 17.0,
    ///                                    7.0, 11.0, 17.0, 17.0,
    ///                                   17.0, 17.0,  7.0, 17.0,
    ///                                   17.0, 17.0, 17.0, 17.0]));
    /// ```
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut a = self.a;
        for it in &mut a {
            *it = it.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all components of the matrix to the specified range.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::from([  2.0, 17.0, -59.0, 127.0,
    ///                                   5.0, 11.0,  47.0, 109.0,
    ///                                  23.0, 31.0, -41.0, 103.0,
    ///                                  67.0, 73.0,  83.0,  97.0]);
    /// m.clamp_in_place(7.0, 17.0);
    ///
    /// assert_eq!(m, Matrix4x4f32::from([ 7.0, 17.0,  7.0, 17.0,
    ///                                    7.0, 11.0, 17.0, 17.0,
    ///                                   17.0, 17.0,  7.0, 17.0,
    ///                                   17.0, 17.0, 17.0, 17.0]));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix4x4f32::from([  2.0,  5.0, 23.0, 67.0,
    ///                                    17.0, 11.0, 31.0, 73.0,
    ///                                    59.0, 47.0, 41.0, 83.0,
    ///                                   127.0,109.0,103.0, 97.0]));
    /// ```
    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            a: [
                self.a[0], self.a[4], self.a[8], self.a[12], //
                self.a[1], self.a[5], self.a[9], self.a[13], //
                self.a[2], self.a[6], self.a[10], self.a[14], //
                self.a[3], self.a[7], self.a[11], self.a[15], //
            ],
        }
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                                   5.0, 11.0, 47.0, 109.0,
    ///                                  23.0, 31.0, 41.0, 103.0,
    ///                                  67.0, 73.0, 83.0,  97.0]);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix4x4f32::from([  2.0,  5.0, 23.0, 67.0,
    ///                                    17.0, 11.0, 31.0, 73.0,
    ///                                    59.0, 47.0, 41.0, 83.0,
    ///                                   127.0,109.0,103.0, 97.0]));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + Matrix4x4Math,
{
    /// Return the adjugate of this matrix, ie the transpose of the cofactor matrix.
    /// Equivalent to the inverse but without dividing by the determinant of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::One;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let (n,d) = m.adjugate();
    /// assert_eq!(d, m.determinant());
    ///
    /// assert!((n*m/m.determinant()).is_near_identity());
    /// assert_eq!(Matrix4x4f32::one(), n*m/m.determinant());
    /// ```
    #[inline]
    pub fn adjugate(self) -> (Self,T) {
        let (adjugate, determinant) = T::m4x4_adjugate(self);
        (adjugate, determinant)
    }

    /// Adjugate matrix, in-place.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                                   5.0, 11.0, 47.0, 109.0,
    ///                                  23.0, 31.0, 41.0, 103.0,
    ///                                  67.0, 73.0, 83.0,  97.0]);
    /// let mut n = m;
    /// n.adjugate_in_place();
    ///
    /// assert_eq!(m.adjugate().0, n);
    /// ```
    #[inline]
    pub fn adjugate_in_place(&mut self) -> &mut Self {
        *self = self.adjugate().0;
        self
    }
    /// Return the inverse of this matrix. Does not check if the determinant is non-zero before inverting.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let n = m.inverse();
    ///
    /// ```
    #[inline]
    pub fn inverse(self) -> Self {
        let (adjugate, determinant) = T::m4x4_adjugate(self);
        adjugate
    }

    /// Invert this matrix, in-place. Does not check if the determinant is non-zero before inverting.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let mut m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                                   5.0, 11.0, 47.0, 109.0,
    ///                                  23.0, 31.0, 41.0, 103.0,
    ///                                  67.0, 73.0, 83.0,  97.0]);
    /// m.invert_in_place();
    /// ```
    #[inline]
    pub fn invert_in_place(&mut self) -> &mut Self {
        let (adjugate, determinant) = T::m4x4_adjugate(*self);
        *self = adjugate / determinant;
        self
    }

    /// Matrix determinant.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let d = m.determinant();
    ///
    /// //assert_eq!(-78.0, d);
    ///
    /// ```
    #[inline]
    pub fn determinant(self) -> T {
        T::m4x4_determinant(self)
    }

    /// Return trace of matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let t = m.trace();
    ///
    /// assert_eq!(t, 151.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T {
        T::m4x4_trace(self)
    }
}

impl<T> Matrix4x4<T>
where
    T: Copy + Zero + One + Matrix4x4Math + MathConstants + PartialOrd + Signed,
{
    /// Return inverse of matrix or `T::zero()` if not invertible.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::Zero;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               2.0, 17.0, 59.0, 127.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix4x4f32::zero(), n);
    ///
    /// ```
    pub fn inverse_or_zero(self) -> Self {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return Self::zero();
        }
        adjugate / determinant
    }

    /// Return inverse of matrix or `None` if not invertible.
    /// ```
    /// # use vqm::{Matrix4x4f32};
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               2.0, 17.0, 59.0, 127.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let n = m.try_invert();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(None, n);
    ///
    /// ```
    pub fn try_invert(self) -> Option<Self> {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return None;
        }
        Some(adjugate / determinant)
    }

    /// Return the sum of all components of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 895.0);
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        T::m4x4_sum(self)
    }

    /// Return the mean of all components of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 895.0 / 16.0);
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        T::m4x4_mean(self)
    }

    /// Return the product of all components of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 8.510985e24);
    /// ```
    #[inline]
    pub fn product(self) -> T {
        T::m4x4_product(self)
    }

    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
    ///                               5.0, 11.0, 47.0, 109.0,
    ///                              23.0, 31.0, 41.0, 103.0,
    ///                              67.0, 73.0, 83.0,  97.0]);
    /// let t = m.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 *11.0 + 41.0 * 41.0 + 97.0 * 97.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T {
        T::m4x4_trace_sum_squares(self)
    }

    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::Zero;
    /// let z = Matrix4x4f32::zero();
    /// assert!(z.is_near_zero());
    /// ```
    pub fn is_near_zero(self) -> bool {
        for a in &self.a {
            if a.abs() > T::EPSILON {
                return false;
            }
        }
        true
    }

    /// Return true if matrix is near identity.
    /// ```
    /// # use vqm::Matrix4x4f32;
    /// # use num_traits::One;
    /// let i = Matrix4x4f32::one();
    /// assert!(i.is_near_identity());
    /// ```
    pub fn is_near_identity(self) -> bool {
        if self.a[1].abs() > T::EPSILON
            || self.a[2].abs() > T::EPSILON
            || self.a[3].abs() > T::EPSILON
            || self.a[4].abs() > T::EPSILON
            || self.a[6].abs() > T::EPSILON
            || self.a[7].abs() > T::EPSILON
            || self.a[8].abs() > T::EPSILON
            || self.a[9].abs() > T::EPSILON
            || self.a[11].abs() > T::EPSILON
            || self.a[12].abs() > T::EPSILON
            || self.a[13].abs() > T::EPSILON
        {
            return false;
        }
        if (self.a[0] - T::one()).abs() > T::EPSILON
            || (self.a[5] - T::one()).abs() > T::EPSILON
            || (self.a[10] - T::one()).abs() > T::EPSILON
            || (self.a[15] - T::one()).abs() > T::EPSILON
        {
            return false;
        }
        true
    }
}

// **** From ****

// **** From Array ****

/// Matrix from 1D array.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([  2.0, 17.0, 59.0, 127.0,
///                               5.0, 11.0, 47.0, 109.0,
///                              23.0, 31.0, 41.0, 103.0,
///                              67.0, 73.0, 83.0,  97.0]);
/// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
///                                    5.0, 11.0, 47.0, 109.0,
///                                   23.0, 31.0, 41.0, 103.0,
///                                   67.0, 73.0, 83.0,  97.0]));
/// ```
impl<T> From<[T; 16]> for Matrix4x4<T>
where
    T: Copy,
{
    #[inline]
    fn from(input: [T; 16]) -> Self {
        Self { a: input }
    }
}

/// Matrix from 2D array.
/// ```
/// # use vqm::Matrix4x4f32;
/// let m = Matrix4x4f32::from([[  2.0, 17.0, 59.0, 127.0],
///                             [  5.0, 11.0, 47.0, 109.0],
///                             [ 23.0, 31.0, 41.0, 103.0],
///                             [ 67.0, 73.0, 83.0,  97.0]]);
/// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
///                                    5.0, 11.0, 47.0, 109.0,
///                                   23.0, 31.0, 41.0, 103.0,
///                                   67.0, 73.0, 83.0,  97.0]));
/// ```
impl<T> From<[[T; 4]; 4]> for Matrix4x4<T>
where
    T: Copy,
{
    #[inline]
    fn from(a: [[T; 4]; 4]) -> Self {
        Self {
            a: [
                //
                a[0][0], a[0][1], a[0][2], a[0][3], //
                a[1][0], a[1][1], a[1][2], a[1][3], //
                a[2][0], a[2][1], a[2][2], a[2][3], //
                a[3][0], a[3][1], a[3][2], a[3][3], //
            ],
        }
    }
}

/// Matrix from array of vectors.
/// ```
/// # use vqm::{Matrix4x4f32,Vector4df32};
/// let m = Matrix4x4f32::from([ Vector4df32::new( 2.0, 17.0, 59.0, 127.0),
///                              Vector4df32::new( 5.0, 11.0, 47.0, 109.0),
///                              Vector4df32::new(23.0, 31.0, 41.0, 103.0),
///                              Vector4df32::new(67.0, 73.0, 83.0,  97.0) ]);
/// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
///                                    5.0, 11.0, 47.0, 109.0,
///                                   23.0, 31.0, 41.0, 103.0,
///                                   67.0, 73.0, 83.0,  97.0]));
/// ```
impl<T> From<[Vector4d<T>; 4]> for Matrix4x4<T>
where
    T: Copy,
{
    #[inline]
    fn from(v: [Vector4d<T>; 4]) -> Self {
        Self {
            a: [
                //
                v[0].x, v[0].y, v[0].z, v[0].t, //
                v[1].x, v[1].y, v[1].z, v[1].t, //
                v[2].x, v[2].y, v[2].z, v[2].t, //
                v[3].x, v[3].y, v[3].z, v[3].t, //
            ],
        }
    }
}

/// Matrix from tuple of vectors.
/// ```
/// # use vqm::{Matrix4x4f32,Vector4df32};
/// let m = Matrix4x4f32::from(( Vector4df32::new( 2.0, 17.0, 59.0, 127.0),
///                              Vector4df32::new( 5.0, 11.0, 47.0, 109.0),
///                              Vector4df32::new(23.0, 31.0, 41.0, 103.0),
///                              Vector4df32::new(67.0, 73.0, 83.0, 97.0) ));
/// assert_eq!(m, Matrix4x4f32::new([  2.0, 17.0, 59.0, 127.0,
///                                    5.0, 11.0, 47.0, 109.0,
///                                   23.0, 31.0, 41.0, 103.0,
///                                   67.0, 73.0, 83.0,  97.0]));
/// ```
impl<T> From<(Vector4d<T>, Vector4d<T>, Vector4d<T>, Vector4d<T>)> for Matrix4x4<T> {
    #[inline]
    fn from(v: (Vector4d<T>, Vector4d<T>, Vector4d<T>, Vector4d<T>)) -> Self {
        Self {
            a: [
                //
                v.0.x, v.0.y, v.0.z, v.0.t, //
                v.1.x, v.1.y, v.1.z, v.1.t, //
                v.2.x, v.2.y, v.2.z, v.2.t, //
                v.3.x, v.3.y, v.3.z, v.3.t, //
            ],
        }
    }
}

// **** From Matrix ****

/// Matrix4x4 from Matrix2x2.
/// ```
/// # use vqm::{Matrix2x2f32,Matrix4x4f32};
/// let m2 = Matrix2x2f32::from([ 2.0, 17.0,
///                               5.0, 11.0]);
/// let n2 = Matrix2x2f32::from([ 3.0, 19.0,
///                               7.0, 13.0]);
/// let m3: Matrix4x4f32 = m2.into();
/// let n3 = Matrix4x4f32::from(m2);
///
/// assert_eq!(m3, Matrix4x4f32::from([ 2.0, 17.0, 0.0, 0.0,
///                                     5.0, 11.0, 0.0, 0.0,
///                                     0.0,  0.0, 0.0, 0.0,
///                                     0.0,  0.0, 0.0, 0.0]));
/// ```
impl<T> From<Matrix2x2<T>> for Matrix4x4<T>
where
    T: Copy + Zero,
{
    #[inline]
    fn from(m: Matrix2x2<T>) -> Self {
        Self { a: [
            m[0], m[1], T::zero(), T::zero(), //
            m[2], m[3],  T::zero(), T::zero(), //
            T::zero(), T::zero(), T::zero(), T::zero(), //
            T::zero(), T::zero(), T::zero(), T::zero() //
            ] }
    }
}

/// Matrix2x2 from Matrix4x4. Takes top left of m4x4, discarding other values.
/// ```
/// # use vqm::{Matrix2x2f32,Matrix4x4f32};
/// let m2x2 = Matrix2x2f32::from([ 2.0, 17.0,
///                                 5.0, 11.0]);
/// let m4x4 = Matrix4x4f32::from([ 2.0, 17.0, 59.0, 127.0,
///                                 5.0, 11.0, 47.0, 109.0,
///                                23.0, 31.0, 41.0, 103.0,
///                                67.0, 73.0, 83.0,  97.0]);
/// assert_eq!(m2x2, Matrix2x2f32::from(m4x4));
/// ```
impl<T> From<Matrix4x4<T>> for Matrix2x2<T>
where
    T: Copy,
{
    #[inline]
    fn from(m: Matrix4x4<T>) -> Self {
        Self { a: [m.a[0], m.a[1], m.a[4], m.a[5]] }
    }
}
