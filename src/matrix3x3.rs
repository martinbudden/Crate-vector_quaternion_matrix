use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    RangeInclusive, Sub, SubAssign,
};
use core::slice::{ChunksExact, ChunksExactMut};
use num_traits::{ConstOne, ConstZero, MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};
#[cfg(feature = "serde")]
use {
    sequential_storage::map::PostcardValue,
    serde::{Deserialize, Serialize},
};

use crate::{MathConstants, Matrix2x2, Matrix3x3Math, Quaternion, QuaternionMath, SqrtMethods, Vector3d};

/// 3x3 matrix of `f32` values<br>
pub type Matrix3x3f32 = Matrix3x3<f32>;
/// 3x3 matrix of `f64` values<br><br>
pub type Matrix3x3f64 = Matrix3x3<f64>;

// **** Define ****

/// `Matrix3x3<T>`: 3x3 Matrix of type `T`.<br>
/// Aliases `Matrix3x3f32` and `Matrix3x3f64` are provided.<br>
/// Internal implementation is a flattened 3x3 matrix: an array of 9 elements stored in row-major order.
/// That is the element `m[row][col]` is at array position `[row * 3 + col]`.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", allow(clippy::unsafe_derive_deserialize))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "align", repr(C, align(64)))]
#[cfg_attr(not(feature = "align"), repr(C))]
pub struct Matrix3x3<T> {
    // Flattened 3x3 matrix: 9 elements in row-major order
    pub(crate) a: [T; 9],
}

#[cfg(feature = "serde")]
impl<T> PostcardValue<'_> for Matrix3x3<T> where T: Serialize + for<'de> Deserialize<'de> {}

/// Constants to index matrix elements.
impl<T> Matrix3x3<T> {
    pub const SIZE: usize = 9;
    pub const ROW_COUNT: usize = 3;
    pub const COL_COUNT: usize = 3;
    // Row 1
    pub const M11: usize = 0;
    pub const M12: usize = 1;
    pub const M13: usize = 2;
    // Row 2
    pub const M21: usize = 3;
    pub const M22: usize = 4;
    pub const M23: usize = 5;
    // Row 3
    pub const M31: usize = 6;
    pub const M32: usize = 7;
    pub const M33: usize = 8;
}

// **** New ****

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Create a matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                    5.0, 11.0, 47.0,
    ///                                   23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub const fn new(a: [T; 9]) -> Self {
        Self::from_row_array(a)
    }
}

// **** Other constructors ****

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Create a matrix with all its elements set to a single value.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_element(2.0);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0, 2.0, 2.0,
    ///                                    2.0, 2.0, 2.0,
    ///                                    2.0, 2.0, 2.0]));
    /// ```
    #[inline]
    pub const fn from_element(value: T) -> Self {
        Self { a: [value; 9] }
    }

    /// Matrix from array of row vectors.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::from_rows([ Vector3df32::new( 2.0, 17.0, 59.0),
    ///                                   Vector3df32::new( 5.0, 11.0, 47.0),
    ///                                   Vector3df32::new(23.0, 31.0, 41.0) ]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                    5.0, 11.0, 47.0,
    ///                                   23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub const fn from_rows(v: [Vector3d<T>; 3]) -> Self {
        Self {
            a: [
                v[0].x, v[0].y, v[0].z, //
                v[1].x, v[1].y, v[1].z, //
                v[2].x, v[2].y, v[2].z, //
            ],
        }
    }

    /// Matrix from array of column vectors.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::from_columns([ Vector3df32::new( 2.0, 17.0, 59.0),
    ///                                      Vector3df32::new( 5.0, 11.0, 47.0),
    ///                                      Vector3df32::new(23.0, 31.0, 41.0) ]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0,   5.0,  23.0,
    ///                                   17.0,  11.0,  31.0,
    ///                                   59.0,  47.0,  41.0]));
    /// ```
    #[inline]
    pub const fn from_columns(v: [Vector3d<T>; 3]) -> Self {
        Self {
            a: [
                v[0].x, v[1].x, v[2].x, //
                v[0].y, v[1].y, v[2].y, //
                v[0].z, v[1].z, v[2].z, //
            ],
        }
    }

    /// Matrix from 1D row array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_row_array([  2.0, 17.0, 59.0,
    ///                                         5.0, 11.0, 47.0,
    ///                                        23.0, 31.0, 41.0]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                    5.0, 11.0, 47.0,
    ///                                   23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub const fn from_row_array(a: [T; 9]) -> Self {
        Self { a }
    }

    /// Matrix from 1D column array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_column_array([  2.0, 17.0, 59.0,
    ///                                            5.0, 11.0, 47.0,
    ///                                           23.0, 31.0, 41.0]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0,   5.0,  23.0,
    ///                                   17.0,  11.0,  31.0,
    ///                                   59.0,  47.0,  41.0]));
    /// ```
    #[inline]
    pub const fn from_column_array(a: [T; 9]) -> Self {
        Self {
            a: [
                a[0], a[3], a[6], //
                a[1], a[4], a[7], //
                a[2], a[5], a[8], //
            ],
        }
    }

    /// Matrix from 2D row array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_2d_row_array([[  2.0, 17.0, 59.0],
    ///                                          [  5.0, 11.0, 47.0],
    ///                                          [ 23.0, 31.0, 41.0]]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                    5.0, 11.0, 47.0,
    ///                                   23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub const fn from_2d_row_array(a: [[T; 3]; 3]) -> Self {
        Self {
            a: [
                a[0][0], a[0][1], a[0][2], //
                a[1][0], a[1][1], a[1][2], //
                a[2][0], a[2][1], a[2][2], //
            ],
        }
    }

    /// Matrix from padded 2D row array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_padded_2d_row_array([[  2.0, 17.0, 59.0, 127.0],
    ///                                                 [  5.0, 11.0, 47.0, 109.0],
    ///                                                 [ 23.0, 31.0, 41.0, 103.0]]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                    5.0, 11.0, 47.0,
    ///                                   23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub const fn from_padded_2d_row_array(a: [[T; 4]; 3]) -> Self {
        Self {
            a: [
                a[0][0], a[0][1], a[0][2], //
                a[1][0], a[1][1], a[1][2], //
                a[2][0], a[2][1], a[2][2], //
            ],
        }
    }

    /// Matrix from padded 2D row array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_padded_2d_column_array([[  2.0, 17.0, 59.0, 127.0],
    ///                                                    [  5.0, 11.0, 47.0, 109.0],
    ///                                                    [ 23.0, 31.0, 41.0, 103.0]]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0,  5.0, 23.0,
    ///                                   17.0, 11.0, 31.0,
    ///                                   59.0, 47.0, 41.0]));
    /// ```
    #[inline]
    pub const fn from_padded_2d_column_array(a: [[T; 4]; 3]) -> Self {
        Self {
            a: [
                a[0][0], a[1][0], a[2][0], //
                a[0][1], a[1][1], a[2][1], //
                a[0][2], a[1][2], a[2][2], //
            ],
        }
    }

    /// Matrix from 2D column array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_2d_column_array([[  2.0, 17.0, 59.0],
    ///                                             [  5.0, 11.0, 47.0],
    ///                                             [ 23.0, 31.0, 41.0]]);
    /// assert_eq!(m, Matrix3x3f32::new([  2.0,   5.0,  23.0,
    ///                                   17.0,  11.0,  31.0,
    ///                                   59.0,  47.0,  41.0]));
    /// ```
    #[inline]
    pub const fn from_2d_column_array(a: [[T; 3]; 3]) -> Self {
        Self {
            a: [
                a[0][0], a[1][0], a[2][0], //
                a[0][1], a[1][1], a[2][1], //
                a[0][2], a[1][2], a[2][2], //
            ],
        }
    }

    /// Try to create a matrix from a row slice.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let valid_data = [2.0; 9];
    /// let invalid_data = [2.0; 3];
    /// let Some(m) = Matrix3x3f32::try_from_row_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix3x3), but got None");
    /// };
    /// assert_eq!(2.0, m[0]);
    /// let None = Matrix3x3f32::try_from_row_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    pub fn try_from_row_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 9 {
            return None;
        }
        let mut a = [slice[0]; 9];
        a.copy_from_slice(&slice[0..9]);
        Some(Self { a })
    }

    /// Try to create a matrix from a column slice.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let valid_data = [2.0; 9];
    /// let invalid_data = [2.0; 3];
    /// let Some(m) = Matrix3x3f32::try_from_column_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix3x3), but got None");
    /// };
    /// assert_eq!(2.0, m[0]);
    /// let None = Matrix3x3f32::try_from_column_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn try_from_column_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 9 {
            return None;
        }
        let a = [
            slice[0], slice[3], slice[6],
            slice[1], slice[4], slice[7],
            slice[2], slice[5], slice[8]
        ];
        Some(Self { a })
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy + ConstZero,
{
    /// Create a matrix with the diagonal set to a single value.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_diagonal_element(2.0);
    /// assert_eq!(m, Matrix3x3f32::new([ 2.0, 0.0, 0.0,
    ///                                   0.0, 2.0, 0.0,
    ///                                   0.0, 0.0, 2.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_element(value: T) -> Self {
        Self {
            a: [
                value,   T::ZERO, T::ZERO,
                T::ZERO, value,   T::ZERO,
                T::ZERO, T::ZERO, value,
            ]
        }
    }
    /// Create a matrix with the diagonal set to an array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::from_diagonal_array([2.0, 3.0, 5.0]);
    /// assert_eq!(m, Matrix3x3f32::new([ 2.0, 0.0, 0.0,
    ///                                   0.0, 3.0, 0.0,
    ///                                   0.0, 0.0, 5.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_array(a: [T;3]) -> Self {
        Self {
            a: [
                a[0],    T::ZERO, T::ZERO,
                T::ZERO, a[1],     T::ZERO,
                T::ZERO, T::ZERO, a[2],
            ]
        }
    }
    /// Create a matrix with the diagonal set to a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::from_diagonal_vector(Vector3df32::new(2.0, 3.0, 5.0));
    /// assert_eq!(m, Matrix3x3f32::new([ 2.0, 0.0, 0.0,
    ///                                   0.0, 3.0, 0.0,
    ///                                   0.0, 0.0, 5.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_vector(v: Vector3d<T>) -> Self {
        Self {
            a: [
                v.x,     T::ZERO, T::ZERO,
                T::ZERO, v.y,     T::ZERO,
                T::ZERO, T::ZERO, v.z,
            ]
        }
    }
}

// **** Zero ****

impl<T> Zero for Matrix3x3<T>
where
    T: Copy + Zero + PartialEq + Matrix3x3Math,
{
    /// Zero matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::{Zero,zero};
    /// let z = Matrix3x3f32::zero();
    ///
    /// assert_eq!(z, Matrix3x3f32::new([ 0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0,
    ///                                   0.0, 0.0, 0.0]));
    /// assert!(z.is_zero());
    /// ```
    #[inline]
    fn zero() -> Self {
        Self { a: [T::zero(); 9] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T> ConstZero for Matrix3x3<T>
where
    T: Copy + ConstZero + PartialEq + Matrix3x3Math,
{
    /// Const zero matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::{zero,Zero,ConstZero};
    /// let m = Matrix3x3f32::ZERO;
    /// assert!(m.is_zero());
    /// ```
    const ZERO: Self = Self { a: [T::ZERO; 9] };
}

// **** One ****

impl<T> One for Matrix3x3<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix3x3Math,
{
    /// Identity matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::One;
    /// let i = Matrix3x3f32::one();
    ///
    /// assert_eq!(i, Matrix3x3f32::new([ 1.0, 0.0, 0.0,
    ///                                   0.0, 1.0, 0.0,
    ///                                   0.0, 0.0, 1.0]));
    /// ```
    #[inline]
    fn one() -> Self {
        Self::ONE
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

impl<T> ConstOne for Matrix3x3<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix3x3Math,
{
    /// Const identity matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::ConstOne;
    /// let i = Matrix3x3f32::ONE;
    ///
    /// assert_eq!(i, Matrix3x3f32::new([ 1.0, 0.0, 0.0,
    ///                                   0.0, 1.0, 0.0,
    ///                                   0.0, 0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    const ONE: Self = Self {
        a: [
            T::ONE,  T::ZERO, T::ZERO,
            T::ZERO, T::ONE,  T::ZERO,
            T::ZERO, T::ZERO, T::ONE,
        ]
    };
}

impl<T> Matrix3x3<T>
where
    T: Copy + Zero + One,
{
    /// Identity matrix.
    /// Alias for `one()` that does not require `num_traits::One`.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let i = Matrix3x3f32::identity();
    ///
    /// assert_eq!(i, Matrix3x3f32::new([ 1.0, 0.0, 0.0,
    ///                                   0.0, 1.0, 0.0,
    ///                                   0.0, 0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: [
                T::one(),  T::zero(), T::zero(),
                T::zero(), T::one(),  T::zero(),
                T::zero(), T::zero(), T::one()
            ],
        }
    }
}

// **** Neg ****

impl<T> Neg for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Negate matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// m = - m;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([ -2.0, -17.0, -59.0,
    ///                                   -5.0, -11.0, -47.0,
    ///                                  -23.0, -31.0, -41.0]));
    /// ```
    #[inline]
    fn neg(self) -> Self {
        T::m3x3_neg(self)
    }
}

// **** Add ****

impl<T> Add for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Add two matrices.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                              7.0, 13.0, 53.0,
    ///                             29.0, 37.0, 43.0]);
    /// let r = m + n;
    ///
    /// assert_eq!(r, Matrix3x3f32::new([  5.0, 36.0, 120.0,
    ///                                   12.0, 24.0, 100.0,
    ///                                   52.0, 68.0,  84.0]));
    ///
    /// # use num_traits::Zero;
    ///
    /// let z = Matrix3x3f32::zero();
    /// let r2 = m + z;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        T::m3x3_add(self, other)
    }
}

// **** AddAssign ****

impl<T> AddAssign for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Add one matrix to another.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                              7.0, 13.0, 53.0,
    ///                             29.0, 37.0, 43.0]);
    /// m += n;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  5.0, 36.0, 120.0,
    ///                                   12.0, 24.0, 100.0,
    ///                                   52.0, 68.0,  84.0]));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Multiply vector by constant and add another vector.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::MulAdd;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                              23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                              7.0, 13.0, 53.0,
    ///                             29.0, 37.0, 43.0]);
    /// let k = 137.0;
    /// let r = m.mul_add(k, n);
    ///
    /// assert_eq!(r, Matrix3x3f32::new([  277.0,  2348.0,  8144.0,
    ///                                    692.0,  1520.0,  6492.0,
    ///                                   3180.0,  4284.0,  5660.0]));
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m3x3_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Multiply matrix by constant and add another matrix in place.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::MulAddAssign;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                              7.0, 13.0, 53.0,
    ///                             29.0, 37.0, 43.0]);
    /// let k = 137.0;
    /// m.mul_add_assign(k, n);
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  277.0,  2348.0,  8144.0,
    ///                                    692.0,  1520.0,  6492.0,
    ///                                   3180.0,  4284.0,  5660.0]));
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Subtract two matrices.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 13.0, 43.0,
    ///                              7.0, 19.0, 37.0,
    ///                             29.0, 61.0, 53.0]);
    /// let r = m - n;
    ///
    /// assert_eq!(r, Matrix3x3f32::new([  -1.0,  4.0, 16.0,
    ///                                    -2.0, -8.0, 10.0,
    ///                                    -6.0,-30.0,-12.0]));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Subtract one matrix from another.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 13.0, 43.0,
    ///                              7.0, 19.0, 37.0,
    ///                             29.0, 61.0, 53.0]);
    /// m -= n;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  -1.0,  4.0, 16.0,
    ///                                    -2.0, -8.0, 10.0,
    ///                                    -6.0,-30.0,-12.0]));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

impl Mul<Matrix3x3<f32>> for f32 {
    type Output = Matrix3x3<f32>;

    /// Pre-multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix3x3f32::new([  4.0, 34.0, 118.0,
    ///                                   10.0, 22.0,  94.0,
    ///                                   46.0, 62.0,  82.0]));
    /// ```
    #[inline]
    fn mul(self, other: Matrix3x3<f32>) -> Matrix3x3<f32> {
        f32::m3x3_mul_scalar(other, self)
    }
}

impl Mul<Matrix3x3<f64>> for f64 {
    type Output = Matrix3x3<f64>;

    #[inline]
    fn mul(self, other: Matrix3x3<f64>) -> Matrix3x3<f64> {
        f64::m3x3_mul_scalar(other, self)
    }
}

// **** Mul ****

impl<T> Mul<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let r = m * 2.0;
    ///
    /// assert_eq!(r, Matrix3x3f32::new([  4.0, 34.0, 118.0,
    ///                                   10.0, 22.0,  94.0,
    ///                                   46.0, 62.0,  82.0]));
    /// ```
    #[inline]
    fn mul(self, other: T) -> Self {
        T::m3x3_mul_scalar(self, other)
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// In-place multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// m *= 2.0;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  4.0, 34.0, 118.0,
    ///                                   10.0, 22.0,  94.0,
    ///                                   46.0, 62.0,  82.0]));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T> Mul<Vector3d<T>> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Vector3d<T>;

    /// Multiply a vector by a matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use vqm::Vector3df32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let v = Vector3df32{x:3.0, y:7.0, z:13.0};
    /// let r = m * v;
    ///
    /// assert_eq!(r, Vector3df32{x:892.0, y:703.0, z:819.0});
    /// ```
    #[inline]
    fn mul(self, other: Vector3d<T>) -> Vector3d<T> {
        T::m3x3_mul_vector(self, other)
    }
}

#[cfg(not(feature = "uom"))]
impl<T> Mul<Matrix3x3<T>> for Vector3d<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Pre-multiply a vector by a matrix.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::new([  2.0,   3.0,   5.0,
    ///                             11.0,  13.0,  17.0,
    ///                             23.0,  29.0,  31.0]);
    /// let v = Vector3df32{x:59.0, y:61.0, z:67.0};
    /// let r = v * m;
    ///
    /// assert_eq!(r, Vector3df32{x:2330.0, y:2913.0, z:3409.0});
    /// ```
    #[inline]
    fn mul(self, other: Matrix3x3<T>) -> Self {
        T::m3x3_vector_mul(self, other)
    }
}

impl<T> Mul<Matrix3x3<T>> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Multiply two matrices.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                              7.0, 13.0, 53.0,
    ///                             29.0, 37.0, 43.0]);
    /// let r = m * n;
    ///
    /// assert_eq!(r, Matrix3x3f32::new([
    ///    2.0*3.0 + 17.0*7.0 + 59.0*29.0,   2.0*19.0 + 17.0*13.0 + 59.0*37.0,   2.0*61.0 + 17.0*53.0 + 59.0*43.0,
    ///    5.0*3.0 + 11.0*7.0 + 47.0*29.0,   5.0*19.0 + 11.0*13.0 + 47.0*37.0,   5.0*61.0 + 11.0*53.0 + 47.0*43.0,
    ///   23.0*3.0 + 31.0*7.0 + 41.0*29.0,  23.0*19.0 + 31.0*13.0 + 41.0*37.0,  23.0*61.0 + 31.0*53.0 + 41.0*43.0,
    /// ]));
    ///
    /// # use num_traits::One;
    ///
    /// let i = Matrix3x3f32::one();
    /// let r2 = m * i;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        T::m3x3_mul(self, other)
    }
}

impl<T> MulAssign<Matrix3x3<T>> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Multiply one matrix by another.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// let n = Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                              7.0, 13.0, 53.0,
    ///                             29.0, 37.0, 43.0]);
    /// m *= n;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([
    ///    2.0*3.0 + 17.0*7.0 + 59.0*29.0,   2.0*19.0 + 17.0*13.0 + 59.0*37.0,   2.0*61.0 + 17.0*53.0 + 59.0*43.0,
    ///    5.0*3.0 + 11.0*7.0 + 47.0*29.0,   5.0*19.0 + 11.0*13.0 + 47.0*37.0,   5.0*61.0 + 11.0*53.0 + 47.0*43.0,
    ///   23.0*3.0 + 31.0*7.0 + 41.0*29.0,  23.0*19.0 + 31.0*13.0 + 41.0*37.0,  23.0*61.0 + 31.0*53.0 + 41.0*43.0,
    /// ]));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Matrix3x3<T>) {
        *self = *self * other;
    }
}

// **** Outer Product ****

impl<T> Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Calculates the outer product of a column vector and a row vector to give a matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use vqm::Vector3df32;
    /// let row = Vector3df32{x:2.0, y:5.0, z:11.0};
    /// let col = Vector3df32{x:3.0, y:7.0, z:13.0};
    /// let m = Matrix3x3f32::outer_product(col, row);
    /// assert_eq!(m, Matrix3x3f32::new([ 6.0,  15.0,  33.0,
    ///                                  14.0,  35.0,  77.0,
    ///                                  26.0,  65.0, 143.0]));
    ///```
    #[inline]
    pub fn outer_product(col: Vector3d<T>, row: Vector3d<T>) -> Self {
        T::m3x3_vector_outer_product(col, row)
    }
}

impl<T> Vector3d<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Calculates the outer product with another vector to give a matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use vqm::Vector3df32;
    /// let row = Vector3df32{x:2.0, y:5.0, z:11.0};
    /// let col = Vector3df32{x:3.0, y:7.0, z:13.0};
    /// let m = col.outer_product(row);
    /// assert_eq!(m, Matrix3x3f32::new([ 6.0,  15.0,  33.0,
    ///                                  14.0,  35.0,  77.0,
    ///                                  26.0,  65.0, 143.0]));
    ///```
    #[inline]
    pub fn outer_product(self, row: Vector3d<T>) -> Matrix3x3<T> {
        T::m3x3_vector_outer_product(self, row)
    }
}

// **** Div ****

impl<T> Div<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    type Output = Self;

    /// Divide a matrix by a constant.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let r = m / 2.0;
    ///
    /// assert_eq!(r, Matrix3x3f32::new([ 1.0,  8.5, 29.5,
    ///                                   2.5,  5.5, 23.5,
    ///                                  11.5, 15.5, 20.5]));
    /// ```
    #[inline]
    fn div(self, other: T) -> Self {
        T::m3x3_div_scalar(self, other)
    }
}

// **** DivAssign ****

impl<T> DivAssign<T> for Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// In-place divide a matrix by a constant.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// m /= 2.0;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([ 1.0,  8.5, 29.5,
    ///                                   2.5,  5.5, 23.5,
    ///                                  11.5, 15.5, 20.5]));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** AsRef ****

impl<T> AsRef<[T; 9]> for Matrix3x3<T> {
    /// Immutable reference to the raw array
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let a: &[f32; 9] = m.as_ref();
    /// assert_eq!(5.0, a[Matrix3x3f32::M21]);
    /// ```
    #[inline]
    fn as_ref(&self) -> &[T; 9] {
        &self.a
    }
}

impl<T> AsMut<[T; 9]> for Matrix3x3<T> {
    /// Mutable reference to the raw array
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// let a: &mut [f32; 9] = m.as_mut();
    /// a[Matrix3x3f32::M12] = 7.0;
    /// assert_eq!(7.0, m[Matrix3x3f32::M12]);
    /// ```
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 9] {
        &mut self.a
    }
}

// **** Deref ****

impl<T> Deref for Matrix3x3<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.a
    }
}

impl<T> DerefMut for Matrix3x3<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.a
    }
}

// **** Index ****

impl<T> Index<usize> for Matrix3x3<T> {
    type Output = T;

    /// Access matrix element by index.
    /// ```
    /// # use vqm::Matrix3x3f32;
    ///
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    ///
    /// assert_eq!(m[Matrix3x3f32::M11], 2.0);
    /// assert_eq!(m[Matrix3x3f32::M12], 17.0);
    /// assert_eq!(m[Matrix3x3f32::M13], 59.0);
    /// assert_eq!(m[Matrix3x3f32::M21], 5.0);
    /// assert_eq!(m[Matrix3x3f32::M22], 11.0);
    /// assert_eq!(m[Matrix3x3f32::M23], 47.0);
    /// assert_eq!(m[Matrix3x3f32::M31], 23.0);
    /// assert_eq!(m[Matrix3x3f32::M32], 31.0);
    /// assert_eq!(m[Matrix3x3f32::M33], 41.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> Index<Range<usize>> for Matrix3x3<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<RangeFull> for Matrix3x3<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        &self.a
    }
}

impl<T> Index<RangeInclusive<usize>> for Matrix3x3<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix3x3<T> {
    type Output = T;

    /// Access matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix3x3f32;
    ///
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    ///
    /// assert_eq!(m[(0,0)], 2.0);
    /// assert_eq!(m[(0,1)], 17.0);
    /// assert_eq!(m[(0,2)], 59.0);
    /// assert_eq!(m[(1,0)], 5.0);
    /// assert_eq!(m[(1,1)], 11.0);
    /// assert_eq!(m[(1,2)], 47.0);
    /// assert_eq!(m[(2,0)], 23.0);
    /// assert_eq!(m[(2,1)], 31.0);
    /// assert_eq!(m[(2,2)], 41.0);
    /// ```
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 3 && col < 3, "Matrix index out of bounds: row={row}, col={col}");
        &self.a[row * 3 + col]
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Matrix3x3<T> {
    /// Set matrix element by index.
    /// ```
    /// # use vqm::Matrix3x3f32;
    ///
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    ///
    /// m[Matrix3x3f32::M11] = 3.0;
    /// m[Matrix3x3f32::M12] = 19.0;
    /// m[Matrix3x3f32::M13] = 61.0;
    /// m[Matrix3x3f32::M21] = 7.0;
    /// m[Matrix3x3f32::M22] = 13.0;
    /// m[Matrix3x3f32::M23] = 53.0;
    /// m[Matrix3x3f32::M31] = 29.0;
    /// m[Matrix3x3f32::M32] = 37.0;
    /// m[Matrix3x3f32::M33] = 43.0;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                                    7.0, 13.0, 53.0,
    ///                                   29.0, 37.0, 43.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

impl<T> IndexMut<Range<usize>> for Matrix3x3<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<RangeFull> for Matrix3x3<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        &mut self.a
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Matrix3x3<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix3x3<T> {
    /// Set matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix3x3f32;
    ///
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    ///
    /// m[(0,0)] = 3.0;
    /// m[(0,1)] = 19.0;
    /// m[(0,2)] = 61.0;
    /// m[(1,0)] = 7.0;
    /// m[(1,1)] = 13.0;
    /// m[(1,2)] = 53.0;
    /// m[(2,0)] = 29.0;
    /// m[(2,1)] = 37.0;
    /// m[(2,2)] = 43.0;
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  3.0, 19.0, 61.0,
    ///                                    7.0, 13.0, 53.0,
    ///                                   29.0, 37.0, 43.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 3 && col < 3, "Matrix index out of bounds: row={row}, col={col}");
        &mut self.a[row * 3 + col]
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Set matrix row from a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// m.set_row(1, Vector3df32::new(7.0, 13.0, 19.0));
    /// assert_eq!(Vector3df32{ x: 7.0, y: 13.0, z: 19.0 }, m.row(1));
    /// ```
    pub fn set_row(&mut self, row: usize, value: Vector3d<T>) {
        if row >= 3 {
            return;
        }
        let start = row * 3;
        // Extract a 4-element slice.
        // Because row < 4, start + 4 will never exceed 16.
        let row_slice = &mut self.a[start..start + 3];
        row_slice[0] = value.x;
        row_slice[1] = value.y;
        row_slice[2] = value.z;
    }

    /// Return matrix row as a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let v = m.row(0);
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 17.0, z: 59.0 });
    /// assert_eq!(m.row(1), Vector3df32{ x: 5.0, y: 11.0, z: 47.0 });
    /// assert_eq!(m.row(2), Vector3df32{ x: 23.0, y: 31.0, z: 41.0 });
    /// ```
    pub fn row(self, row: usize) -> Vector3d<T> {
        // Branchless clamp: restricts r to 0..=2
        let base = row.min(2) * 3;
        let chunk = &self.a[base..];
        Vector3d { x: chunk[0], y: chunk[1], z: chunk[2] }
    }

    /// Set matrix column from a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// m.set_column(1, Vector3df32::new(7.0, 13.0, 19.0));
    /// assert_eq!(Vector3df32{ x: 7.0, y: 13.0, z: 19.0 }, m.column(1));
    /// ```
    pub fn set_column(&mut self, column: usize, value: Vector3d<T>) {
        if column >= 3 {
            return;
        }
        self.a[column] = value.x;
        self.a[column + 3] = value.y;
        self.a[column + 6] = value.z;
    }

    /// Return matrix column as a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let v = m.column(0);
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 5.0, z: 23.0 });
    /// assert_eq!(m.column(1), Vector3df32{ x: 17.0, y: 11.0, z: 31.0 });
    /// assert_eq!(m.column(2), Vector3df32{ x: 59.0, y: 47.0, z: 41.0 });
    /// ```
    pub fn column(self, column: usize) -> Vector3d<T> {
        let c = column.min(2);
        // Made safe because c is clamped to 0..=2, so c + 6 <= 8
        unsafe {
            Vector3d { x: *self.a.get_unchecked(c), y: *self.a.get_unchecked(c + 3), z: *self.a.get_unchecked(c + 6) }
        }
    }

    /// Return matrix diagonal as an array.
    /// ```
    /// # use vqm::Matrix3x3f32;
    ///
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let a = m.diagonal_as_array();
    ///
    /// assert_eq!(a, [ 2.0, 11.0, 41.0 ]);
    /// ```
    pub fn diagonal_as_array(self) -> [T; 3] {
        [self.a[Self::M11], self.a[Self::M22], self.a[Self::M33]]
    }

    /// Return matrix diagonal as a vector.
    /// ```
    /// # use vqm::{Matrix3x3f32,Vector3df32};
    ///
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let v = m.diagonal_as_vector();
    ///
    /// assert_eq!(v, Vector3df32{ x: 2.0, y: 11.0, z: 41.0 });
    /// ```
    pub fn diagonal_as_vector(self) -> Vector3d<T> {
        Vector3d { x: self.a[Self::M11], y: self.a[Self::M22], z: self.a[Self::M33] }
    }
}

// **** abs ****

impl<T> Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Return a copy of the matrix with all elements set to their absolute values.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, -17.0,  59.0,
    ///                              5.0, -11.0,  47.0,
    ///                             23.0,  31.0, -41.0]);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                    5.0, 11.0, 47.0,
    ///                                   23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        T::m3x3_abs(self)
    }

    /// Set all elements of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, -17.0, 59.0,
    ///                                  5.0, -11.0, 47.0,
    ///                                 23.0, 31.0, -41.0]);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                    5.0, 11.0, 47.0,
    ///                                   23.0, 31.0, 41.0]));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m3x3_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix3x3<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all elements clamped to the specified range.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, -59.0,
    ///                              5.0, 11.0,  47.0,
    ///                             23.0, 31.0, -41.0]);
    /// let n = m.clamp(7.0, 17.0);
    ///
    /// assert_eq!(n, Matrix3x3f32::new([ 7.0, 17.0,  7.0,
    ///                                   7.0, 11.0, 17.0,
    ///                                  17.0, 17.0,  7.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut a = self.a;
        for it in &mut a {
            *it = it.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all elements of the matrix to the specified range.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, -59.0,
    ///                                  5.0, 11.0,  47.0,
    ///                                 23.0, 31.0, -41.0]);
    /// m.clamp_in_place(7.0, 17.0);
    ///
    /// assert_eq!(m, Matrix3x3f32::new([ 7.0, 17.0,  7.0,
    ///                                   7.0, 11.0, 17.0,
    ///                                  17.0, 17.0,  7.0]));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix3x3f32::new([  2.0,  5.0, 23.0,
    ///                                   17.0, 11.0, 31.0,
    ///                                   59.0, 47.0, 41.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self { a: [self.a[0], self.a[3], self.a[6], self.a[1], self.a[4], self.a[7], self.a[2], self.a[5], self.a[8]] }
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix3x3f32::new([  2.0,  5.0, 23.0,
    ///                                   17.0, 11.0, 31.0,
    ///                                   59.0, 47.0, 41.0]));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy + Matrix3x3Math,
{
    /// Return the adjugate of this matrix, ie the transpose of the cofactor matrix.
    /// Equivalent to the inverse but without dividing by the determinant of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::One;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let (n,d) = m.adjugate();
    ///
    /// assert_eq!(m.determinant(), d);
    /// assert!((n*m/m.determinant()).is_near_identity());
    /// assert_eq!(Matrix3x3f32::one(), n*m/(m.determinant()));
    /// ```
    #[inline]
    pub fn adjugate(self) -> (Self, T) {
        let (adjugate, determinant) = T::m3x3_adjugate(self);
        (adjugate, determinant)
    }

    /// Return the adjugate of this matrix, ie the transpose of the cofactor matrix, assuming this matrix is symmetric.
    /// Equivalent to the inverse but without dividing by the determinant of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::One;
    /// let m = Matrix3x3f32::new([  2.0,  5.0, 23.0,
    ///                              5.0, 11.0, 31.0,
    ///                             23.0, 31.0, 41.0]);
    /// let (n,d) = m.adjugate_symmetric();
    /// assert_eq!(n, Matrix3x3f32::new([ -510.0,  508.0, -98.0,
    ///                                    508.0, -447.0,  53.0,
    ///                                    -98.0,   53.0,  -3.0]));
    /// assert_eq!(-734.0, d);
    /// assert!((n*m/m.determinant()).is_near_identity());
    /// assert_eq!(Matrix3x3f32::one(), n*m/(m.determinant()));
    /// ```
    #[inline]
    pub fn adjugate_symmetric(self) -> (Self, T) {
        let (adjugate, determinant) = T::m3x3_adjugate_symmetric(self);
        (adjugate, determinant)
    }

    /// Adjugate matrix, in-place.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
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
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let n = m.inverse();
    ///
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        let (adjugate, determinant) = T::m3x3_adjugate(self);
        adjugate / determinant
    }

    /// Invert this matrix, in-place. Does not check if the determinant is non-zero before inverting.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let mut m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                                  5.0, 11.0, 47.0,
    ///                                 23.0, 31.0, 41.0]);
    /// m.invert_in_place();
    /// ```
    #[inline]
    pub fn invert_in_place(&mut self) -> &mut Self {
        let (adjugate, determinant) = T::m3x3_adjugate(*self);
        *self = adjugate / determinant;
        self
    }

    /// Matrix determinant.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let d = m.determinant();
    ///
    /// assert_eq!(7098.0, d);
    ///
    /// ```
    #[inline]
    pub fn determinant(self) -> T {
        T::m3x3_determinant(self)
    }

    /// Return trace of matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let t = m.trace();
    ///
    /// assert_eq!(t, 54.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T {
        T::m3x3_trace(self)
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy + Zero + One + Matrix3x3Math + MathConstants + PartialOrd + Signed,
{
    /// Return inverse of matrix or `T::zero()` if not invertible.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::Zero;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              2.0, 17.0, 59.0,
    ///                             23.0, 31.0, 41.0]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix3x3f32::zero(), n);
    ///
    /// ```
    #[must_use]
    pub fn inverse_or_zero(self) -> Self {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return Self::zero();
        }
        adjugate / determinant
    }

    /// Return inverse of matrix or `None` if not invertible.
    /// ```
    /// # use vqm::{Matrix3x3f32};
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              2.0, 17.0, 59.0,
    ///                             23.0, 31.0, 41.0]);
    /// let n = m.try_inverse();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(None, n);
    ///
    /// ```
    pub fn try_inverse(self) -> Option<Self> {
        let (adjugate, determinant) = self.adjugate();
        if determinant.abs() < T::EPSILON {
            return None;
        }
        Some(adjugate / determinant)
    }

    /// Return the sum of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 236.0);
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        T::m3x3_sum(self)
    }

    /// Return the mean of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 236.0 / 9.0);
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        T::m3x3_mean(self)
    }

    /// Return the product of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 151_588_013_830.0);
    /// ```
    #[inline]
    pub fn product(self) -> T {
        T::m3x3_product(self)
    }

    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// let m = Matrix3x3f32::new([  2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// let t = m.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 *11.0 + 41.0 * 41.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T {
        T::m3x3_trace_sum_squares(self)
    }

    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::Zero;
    /// let z = Matrix3x3f32::zero();
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
    /// # use vqm::Matrix3x3f32;
    /// # use num_traits::One;
    /// let i = Matrix3x3f32::one();
    /// assert!(i.is_near_identity());
    /// ```
    pub fn is_near_identity(self) -> bool {
        if self.a[1].abs() > T::EPSILON
            || self.a[2].abs() > T::EPSILON
            || self.a[3].abs() > T::EPSILON
            || self.a[5].abs() > T::EPSILON
            || self.a[6].abs() > T::EPSILON
        {
            return false;
        }
        if (self.a[0] - T::one()).abs() > T::EPSILON
            || (self.a[4] - T::one()).abs() > T::EPSILON
            || (self.a[8] - T::one()).abs() > T::EPSILON
        {
            return false;
        }
        true
    }
}

// **** Iterators ****

impl<T> Matrix3x3<T> {
    /// Returns an iterator over the rows of the matrix as slices of 3 elements.
    #[inline]
    pub fn rows(&self) -> ChunksExact<'_, T> {
        self.chunks_exact(3)
    }
}

impl<T> Matrix3x3<T> {
    /// Returns an iterator over the rows of the matrix as mutable slices of 3 elements.
    #[inline]
    pub fn rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.chunks_exact_mut(3)
    }
}

impl<T> Matrix3x3<T> {
    /// Consumes the matrix and returns an array of its 3 rows.
    #[allow(clippy::many_single_char_names)]
    #[inline]
    pub fn into_rows(self) -> [[T; 3]; 3] {
        let [a, b, c, d, e, f, g, h, i] = self.a;
        [[a, b, c], [d, e, f], [g, h, i]]
    }
}

impl<T> Matrix3x3<T>
where
    T: Copy,
{
    /// Returns an iterator over the columns of the matrix as owned 3-element arrays.
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = [T; 3]> {
        // Create an iterator over the column indices (0, 1, 2)
        (0..3).map(|c| {
            // Collect the strided elements for the current column
            [self.a[c], self.a[c + 3], self.a[c + 6]]
        })
    }
}

impl<'a, T> IntoIterator for &'a Matrix3x3<T> {
    type Item = &'a [T];
    type IntoIter = ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the Deref trait automatically to get slice chunks
        self.chunks_exact(3)
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix3x3<T> {
    type Item = &'a mut [T];
    type IntoIter = ChunksExactMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the DerefMut trait automatically
        self.chunks_exact_mut(3)
    }
}

impl<T> IntoIterator for Matrix3x3<T> {
    type Item = [T; 3];
    type IntoIter = core::array::IntoIter<[T; 3], 3>;

    #[allow(clippy::many_single_char_names)]
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let [a, b, c, d, e, f, g, h, i] = self.a;
        // Construct an array of rows, then convert that array into an iterator
        [[a, b, c], [d, e, f], [g, h, i]].into_iter()
    }
}

// **** From ****

// **** From Matrix ****

impl<T> From<Matrix2x2<T>> for Matrix3x3<T>
where
    T: Copy + Zero,
{
    /// Matrix3x3 from Matrix2x2.
    /// ```
    /// # use vqm::{Matrix2x2f32,Matrix3x3f32};
    /// let m2 = Matrix2x2f32::new([ 2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let n2 = Matrix2x2f32::new([ 3.0, 19.0,
    ///                              7.0, 13.0]);
    /// let m3: Matrix3x3f32 = m2.into();
    /// let n3 = Matrix3x3f32::from(m2);
    ///
    /// assert_eq!(m3, Matrix3x3f32::new([ 2.0, 17.0, 0.0,
    ///                                    5.0, 11.0, 0.0,
    ///                                    0.0,  0.0, 0.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix2x2<T>) -> Self {
        Self { a: [
            m[0],      m[1],      T::zero(),
            m[2],      m[3],      T::zero(),
            T::zero(), T::zero(), T::zero()
        ] }
    }
}

impl<T> From<Matrix3x3<T>> for Matrix2x2<T>
where
    T: Copy,
{
    /// Matrix2x2 from Matrix3x3. Takes top left of m3x3, discarding other values.
    /// ```
    /// # use vqm::{Matrix2x2f32,Matrix3x3f32};
    /// let m2 = Matrix2x2f32::new([ 2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let m3 = Matrix3x3f32::new([ 2.0, 17.0, 59.0,
    ///                              5.0, 11.0, 47.0,
    ///                             23.0, 31.0, 41.0]);
    /// assert_eq!(m2, Matrix2x2f32::from(m3));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix3x3<T>) -> Self {
        Self { a: [
            m.a[0], m.a[1],
            m.a[3], m.a[4]
        ] }
    }
}

// **** From Quaternion ****

impl<T> From<Quaternion<T>> for Matrix3x3<T>
where
    T: Copy + Zero + One + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Create rotation matrix from quaternion.
    ///
    /// see [Quaternion-derived rotation matrix](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix),
    /// uses Hamilton convention.
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        let two = T::one() + T::one();
        Self {
            a: [
                T::one() - (q.y * q.y + q.z * q.z) * two,
                (q.x * q.y - q.w * q.z) * two,
                (q.w * q.y + q.x * q.z) * two,
                (q.w * q.z + q.x * q.y) * two,
                T::one() - (q.x * q.x + q.z * q.z) * two,
                (q.y * q.z - q.w * q.x) * two,
                (q.x * q.z - q.w * q.y) * two,
                (q.w * q.x + q.y * q.z) * two,
                T::one() - (q.x * q.x + q.y * q.y) * two,
            ],
        }
    }
}

impl<T> From<Matrix3x3<T>> for Quaternion<T>
where
    T: Copy + One + FloatCore + SqrtMethods + QuaternionMath,
{
    /// Create quaternion from a rotation matrix.
    ///
    /// Adapted from [Converting a Rotation Matrix to a Quaternion](https://d3cw3dd2w33x3b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf) by Mike Day.
    /// Note that Day's paper uses the [Shuster multiplication convention](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Alternative_conventions),
    /// rather than the Hamilton multiplication convention used by the Quaternion class.
    fn from(m: Matrix3x3<T>) -> Self {
        let half = T::one() / (T::one() + T::one());
        // Choose largest scale factor from 4w, 4x, 4y, and 4z, to avoid a scale factor of zero, or numerical instabilities caused by division of a small scale factor.
        if m.a[8] < T::zero() {
            // |(x,y)| is bigger than |(z,w)|?
            if m.a[0] > m.a[4] {
                // |x| bigger than |y|, so use x-form
                let t = T::one() + (m.a[0] - m.a[4]) - m.a[8]; // 1 + 2(xx - yy) - 1 + 2(xx + yy) = 4xx
                let q = Self { w: m.a[7] - m.a[5], x: t, y: m.a[1] + m.a[3], z: m.a[6] + m.a[2] };
                return q * t.sqrt_reciprocal() * half;
            }
            // |y| bigger than |x|, so use y-form
            let t = T::one() - (m.a[0] - m.a[4]) - m.a[8]; // 1 - 2(xx - yy) - 1 + 2(xx + yy) = 4yy
            let q = Self { w: m.a[2] - m.a[6], x: m.a[1] + m.a[3], y: t, z: m.a[5] + m.a[7] };
            return q * t.sqrt_reciprocal() * half;
        }

        // |(z,w)| bigger than |(x,y)|
        if m.a[0] < -m.a[4] {
            // |z| bigger than |w|, so use z-form
            let t = T::one() - m.a[0] - (m.a[4] - m.a[8]); // 1 - (1 - 2*(yy + zz)) - (2(yy - zz)) = 4zz
            let q = Self { w: m.a[3] - m.a[1], x: m.a[2] + m.a[6], y: m.a[5] + m.a[7], z: t };
            return q * t.sqrt_reciprocal() * half;
        }

        // |w| bigger than |z|, so use w-form
        // ww + xx + yy + zz = 1, since unit quaternion, so xx + yy + zz =  1 - ww
        let t = T::one() + m.a[0] + m.a[4] + m.a[8]; // 1 + 1 - 2*(yy + zz) + 1 - 2(xx + zz) + 1 - 2(xx + yy) =  4 - 4(xx + yy + zz) = 4 - 4(1 - ww) = 4ww
        let q = Self { w: t, x: m.a[7] - m.a[5], y: m.a[2] - m.a[6], z: m.a[3] - m.a[1] };
        q * t.sqrt_reciprocal() * half
    }
}
