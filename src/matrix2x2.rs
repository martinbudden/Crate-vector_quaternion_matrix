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

use crate::{MathConstants, Matrix2x2Math, Vector2d};

/// 2x2 matrix of `f32` values<br>
pub type Matrix2x2f32 = Matrix2x2<f32>;
/// 2x2 matrix of `f64` values<br><br>
pub type Matrix2x2f64 = Matrix2x2<f64>;

// **** Define ****

/// `Matrix2x2<T>`: 2x2 Matrix of type `T`.<br>
/// Aliases `Matrix2x2f32` and `Matrix2x2f64` are provided.<br>
/// `Matrix2x2f32` uses **SIMD** accelerations implemented in `Matrix2x2Math`.<br>
/// Internal implementation is using a flattened 1-dimensional array: an array of 4 elements stored in row-major order.
/// That is the element `m[row][col]` is at array position `[row * 2 + col]`, so element `m01` is at `a[1]`.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, align(16))]
pub struct Matrix2x2<T> {
    // Flattened 2x2 matrix: 4 elements in row-major order
    pub(crate) a: [T; 4],
}

#[cfg(feature = "serde")]
impl<T> PostcardValue<'_> for Matrix2x2<T> where T: serde::Serialize + for<'de> serde::Deserialize<'de> {}

/// Constants to index matrix elements.
impl<T> Matrix2x2<T> {
    pub const SIZE: usize = 4;
    pub const ROW_COUNT: usize = 2;
    pub const COL_COUNT: usize = 2;
    // Row 1
    pub const M11: usize = 0;
    pub const M12: usize = 1;
    // Row 2
    pub const M21: usize = 2;
    pub const M22: usize = 3;
}

// **** New ****

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Create a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::new([  2.0,  17.0,
    ///                              5.0,  11.0]);
    /// assert_eq!(m, Matrix2x2f32::from([  2.0,  17.0,
    ///                                     5.0,  11.0]));
    /// ```
    #[inline]
    pub const fn new(input: [T; 4]) -> Self {
        Self { a: input }
    }
}

// **** Zero ****

impl<T> Zero for Matrix2x2<T>
where
    T: Copy + Zero + PartialEq + Matrix2x2Math,
{
    /// Zero matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::{Zero,zero};
    /// let z = Matrix2x2f32::zero();
    ///
    /// assert_eq!(z, Matrix2x2f32::from([ 0.0, 0.0,
    ///                                    0.0, 0.0]));
    /// assert!(z.is_zero());
    /// ```
    #[inline]
    fn zero() -> Self {
        Self { a: [T::zero(); 4] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T> ConstZero for Matrix2x2<T>
where
    T: Copy + ConstZero + PartialEq + Matrix2x2Math,
{
    /// Const zero matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::{zero,Zero,ConstZero};
    /// let m = Matrix2x2f32::ZERO;
    /// assert!(m.is_zero());
    /// ```
    const ZERO: Self = Self { a: [T::ZERO; 4] };
}

// **** One ****

impl<T> One for Matrix2x2<T>
where
    T: Copy + Zero + One + PartialEq + Matrix2x2Math,
{
    /// Identity matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::One;
    /// let i = Matrix2x2f32::one();
    ///
    /// assert_eq!(i, Matrix2x2f32::from([ 1.0, 0.0,
    ///                                    0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    fn one() -> Self {
        Self {
            a: [
                T::one(),  T::zero(),
                T::zero(), T::one(),
            ],
        }
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

impl<T> ConstOne for Matrix2x2<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix2x2Math,
{
    /// Const identity matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::ConstOne;
    /// let i = Matrix2x2f32::ONE;
    ///
    /// assert_eq!(i, Matrix2x2f32::from([ 1.0, 0.0,
    ///                                    0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    const ONE: Self = Self {
        a: [
            T::ONE,  T::ZERO,
            T::ZERO, T::ONE,
        ]
    };
}

impl<T> Matrix2x2<T>
where
    T: Copy + Zero + One,
{
    /// Identity matrix.
    /// Alias for `one()` that does not require `num_traits::One`.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let i = Matrix2x2f32::identity();
    ///
    /// assert_eq!(i, Matrix2x2f32::from([ 1.0, 0.0,
    ///                                    0.0, 1.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn identity() -> Self {
        Self {
            a: [
                T::one(),  T::zero(),
                T::zero(), T::one(),
            ],
        }
    }
}

// **** Neg ****

impl<T> Neg for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Negate matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// m = - m;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([ -2.0, -17.0,
    ///                                    -5.0, -11.0]));
    /// ```
    #[inline]
    fn neg(self) -> Self {
        T::m2x2_neg(self)
    }
}

// **** Add ****

impl<T> Add for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Add two matrices.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = Matrix2x2f32::from([  3.0, 19.0,
    ///                               7.0, 13.0]);
    /// let r = m + n;
    ///
    /// assert_eq!(r, Matrix2x2f32::from([  5.0, 36.0,
    ///                                    12.0, 24.0]));
    ///
    /// # use num_traits::Zero;
    ///
    /// let z = Matrix2x2f32::zero();
    /// let r2 = m + z;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        T::m2x2_add(self, other)
    }
}

// **** AddAssign ****

impl<T> AddAssign for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Add one matrix to another.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0,  17.0,
    ///                                   5.0,  11.0]);
    /// let n = Matrix2x2f32::from([  3.0,  19.0,
    ///                               7.0,  13.0]);
    /// m += n;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  5.0, 36.0,
    ///                                    12.0, 24.0]));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Multiply vector by constant and add another vector.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::MulAdd;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = Matrix2x2f32::from([  3.0, 19.0,
    ///                               7.0, 13.0]);
    /// let k = 137.0;
    /// let r = m.mul_add(k, n);
    ///
    /// assert_eq!(r, Matrix2x2f32::from([  277.0, 2348.0,
    ///                                     692.0, 1520.0]));
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m2x2_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Multiply vector by constant and add another vector in place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::MulAddAssign;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// let n = Matrix2x2f32::from([  3.0, 19.0,
    ///                               7.0, 13.0]);
    /// let k = 137.0;
    /// m.mul_add_assign(k, n);
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  277.0, 2348.0,
    ///                                     692.0, 1520.0]));
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Subtract two matrices.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = Matrix2x2f32::from([  3.0, 13.0,
    ///                               7.0, 19.0]);
    /// let r = m - n;
    ///
    /// assert_eq!(r, Matrix2x2f32::from([  -1.0,  4.0,
    ///                                     -2.0, -8.0]));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse our existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Subtract one matrix from another.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// let n = Matrix2x2f32::from([  3.0, 13.0,
    ///                               7.0, 19.0]);
    /// m -= n;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  -1.0,  4.0,
    ///                                     -2.0, -8.0]));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

impl Mul<Matrix2x2<f32>> for f32 {
    type Output = Matrix2x2<f32>;

    /// Pre-multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix2x2f32::from([  4.0, 34.0,
    ///                                    10.0, 22.0]));
    /// ```
    #[inline]
    fn mul(self, other: Matrix2x2<f32>) -> Matrix2x2<f32> {
        f32::m2x2_mul_scalar(other, self)
    }
}

impl Mul<Matrix2x2<f64>> for f64 {
    type Output = Matrix2x2<f64>;
    #[inline]
    fn mul(self, other: Matrix2x2<f64>) -> Matrix2x2<f64> {
        f64::m2x2_mul_scalar(other, self)
    }
}

// **** Mul ****

impl<T> Mul<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([ 2.0, 17.0,
    ///                              5.0, 11.0]);
    /// let r = m * 2.0;
    ///
    /// assert_eq!(r, Matrix2x2f32::from([  4.0, 34.0,
    ///                                    10.0, 22.0]));
    /// ```
    #[inline]
    fn mul(self, other: T) -> Self {
        T::m2x2_mul_scalar(self, other)
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// In-place multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// m *= 2.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  4.0, 34.0,
    ///                                    10.0, 22.0]));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T> Mul<Vector2d<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Vector2d<T>;

    /// Multiply a vector by a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2df32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// let v = Vector2df32{x:3.0, y:7.0};
    /// let r = m * v;
    ///
    /// assert_eq!(r, Vector2df32{x:2.0*3.0 + 17.0*7.0, y:5.0*3.0 + 11.0*7.0});
    /// ```
    #[inline]
    fn mul(self, other: Vector2d<T>) -> Vector2d<T> {
        T::m2x2_mul_vector(self, other)
    }
}

impl<T> Mul<Matrix2x2<T>> for Vector2d<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Pre-multiply a vector by a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2df32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// let v = Vector2df32{x:3.0, y:7.0};
    /// let r = v * m;
    ///
    /// assert_eq!(r, Vector2df32{x:3.0*2.0 + 7.0*5.0, y:3.0*17.0 + 7.0*11.0});
    /// ```
    #[inline]
    fn mul(self, other: Matrix2x2<T>) -> Self {
        T::m2x2_vector_mul(self, other)
    }
}

impl<T> Mul<Matrix2x2<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Multiply two matrices.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = Matrix2x2f32::from([  3.0, 13.0,
    ///                               7.0, 19.0]);
    /// let r = m * n;
    ///
    /// assert_eq!(r, Matrix2x2f32::from([
    ///    2.0 * 3.0 + 17.0 * 7.0,  2.0 * 13.0 + 17.0 * 19.0,
    ///    5.0 * 3.0 + 11.0 * 7.0,  5.0 * 13.0 + 11.0 * 19.0,
    /// ]));
    ///
    /// # use num_traits::One;
    ///
    /// let i = Matrix2x2f32::one();
    /// let r2 = m * i;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        T::m2x2_mul(self, other)
    }
}

impl<T> MulAssign<Matrix2x2<T>> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Multiply one matrix by another.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// let n = Matrix2x2f32::from([  3.0, 13.0,
    ///                               7.0, 19.0]);
    /// m *= n;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([
    ///    2.0 * 3.0 + 17.0 * 7.0,  2.0 * 13.0 + 17.0 * 19.0,
    ///    5.0 * 3.0 + 11.0 * 7.0,  5.0 * 13.0 + 11.0 * 19.0,
    /// ]));
    ///
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Matrix2x2<T>) {
        *self = *self * other;
    }
}

// **** Outer Product ****

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Calculates the outer product of a column vector and a row vector to give a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2df32;
    /// let row = Vector2df32{x:2.0, y:5.0};
    /// let col = Vector2df32{x:3.0, y:7.0};
    /// let m = Matrix2x2f32::outer_product(col, row);
    /// assert_eq!(m, Matrix2x2f32::from([ 6.0,  15.0,
    ///                                   14.0,  35.0]));
    ///```
    #[inline]
    pub fn outer_product(col: Vector2d<T>, row: Vector2d<T>) -> Self {
        T::m2x2_vector_outer_product(col, row)
    }
}

impl<T> Vector2d<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Calculates the outer product with another vector to give a matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use vqm::Vector2df32;
    /// let row = Vector2df32{x:2.0, y:5.0};
    /// let col = Vector2df32{x:3.0, y:7.0};
    /// let m = col.outer_product(row);
    /// assert_eq!(m, Matrix2x2f32::from([ 6.0,  15.0,
    ///                                   14.0,  35.0]));
    ///```
    #[inline]
    pub fn outer_product(&self, row: Vector2d<T>) -> Matrix2x2<T> {
        T::m2x2_vector_outer_product(*self, row)
    }
}

// **** Div ****

impl<T> Div<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    type Output = Self;

    /// Divide a matrix by a constant.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let r = m / 2.0;
    ///
    /// assert_eq!(r, Matrix2x2f32::from([ 1.0,  8.5,
    ///                                    2.5,  5.5]));
    /// ```
    #[inline]
    fn div(self, other: T) -> Self {
        T::m2x2_div_scalar(self, other)
    }
}

// **** DivAssign ****

impl<T> DivAssign<T> for Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// In-place divide a matrix by a constant.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// m /= 2.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([ 1.0,  8.5,
    ///                                    2.5,  5.5]));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** AsRef ****

impl<T> AsRef<[T; 4]> for Matrix2x2<T> {
    #[inline]
    /// Immutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let a: &[f32; 4] = m.as_ref();
    /// assert_eq!(5.0, a[2]);
    /// ```
    fn as_ref(&self) -> &[T; 4] {
        &self.a
    }
}

impl<T> AsMut<[T; 4]> for Matrix2x2<T> {
    /// Immutable reference to the raw array.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// let a: &mut [f32; 4] = m.as_mut();
    /// a[2] = 7.0;
    /// assert_eq!(7.0, m[2]);
    /// ```
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 4] {
        &mut self.a
    }
}

// **** Deref ****

impl<T> Deref for Matrix2x2<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.a
    }
}

impl<T> DerefMut for Matrix2x2<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.a
    }
}

// **** Index ****

impl<T> Index<usize> for Matrix2x2<T> {
    type Output = T;

    /// Access matrix element by index.
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    ///
    /// assert_eq!(m[0], 2.0);
    /// assert_eq!(m[1], 17.0);
    /// assert_eq!(m[2], 5.0);
    /// assert_eq!(m[3], 11.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> Index<Range<usize>> for Matrix2x2<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<RangeFull> for Matrix2x2<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        &self.a
    }
}

impl<T> Index<RangeInclusive<usize>> for Matrix2x2<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix2x2<T> {
    type Output = T;

    /// Access matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    ///
    /// assert_eq!(m[(0,0)], 2.0);
    /// assert_eq!(m[(0,1)], 17.0);
    /// assert_eq!(m[(1,0)], 5.0);
    /// assert_eq!(m[(1,1)], 11.0);
    /// ```
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 2 && col < 2, "Matrix index out of bounds: row={row}, col={col}");
        &self.a[row * 2 + col]
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Matrix2x2<T> {
    /// Set matrix element by index.
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    ///
    /// m[0] = 3.0;
    /// m[1] = 19.0;
    /// m[2] = 7.0;
    /// m[3] = 13.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  3.0, 19.0,
    ///                                     7.0, 13.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

impl<T> IndexMut<Range<usize>> for Matrix2x2<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<RangeFull> for Matrix2x2<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        &mut self.a
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Matrix2x2<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix2x2<T> {
    /// Set matrix element by ordered pair (row, column).
    /// ```
    /// # use vqm::Matrix2x2f32;
    ///
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    ///
    /// m[(0,0)] = 3.0;
    /// m[(0,1)] = 19.0;
    /// m[(1,0)] = 7.0;
    /// m[(1,1)] = 13.0;
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  3.0, 19.0,
    ///                                     7.0, 13.0]));
    /// ```
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 2 && col < 2, "Matrix index out of bounds: row={row}, col={col}");
        &mut self.a[row * 2 + col]
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    pub fn set_row(&mut self, row: usize, value: Vector2d<T>) {
        // Get a mutable slice of exactly the requested row.
        // This will naturally panic with a clean message if 'row >= 2'.
        let row_slice = &mut self.a[row * 2..(row + 1) * 2];
        row_slice.copy_from_slice(&[value.x, value.y]);
    }

    /// Return matrix row as a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2df32};
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let v = m.row(0);
    ///
    /// assert_eq!(v, Vector2df32{ x: 2.0, y: 17.0 });
    /// assert_eq!(m.row(1), Vector2df32{ x: 5.0, y: 11.0 });
    /// ```
    pub fn row(self, row: usize) -> Vector2d<T> {
        match row {
            0 => Vector2d::<T> { x: self.a[0], y: self.a[1] },
            // default to row 1 if row out of range
            _ => Vector2d::<T> { x: self.a[2], y: self.a[3] },
        }
    }

    pub fn set_column(&mut self, column: usize, value: Vector2d<T>) {
        if column == 0 {
            self.a[0] = value.x;
            self.a[3] = value.y;
        } else {
            self.a[1] = value.x;
            self.a[2] = value.y;
        }
    }

    /// Return matrix column as a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2df32};
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let v = m.column(0);
    ///
    /// assert_eq!(v, Vector2df32{ x: 2.0, y: 5.0 });
    /// assert_eq!(m.column(1), Vector2df32{ x: 17.0, y: 11.0 });
    /// ```
    pub fn column(self, column: usize) -> Vector2d<T> {
        match column {
            0 => Vector2d::<T> { x: self.a[0], y: self.a[2] },
            // default to column 1 if column out of range
            _ => Vector2d::<T> { x: self.a[1], y: self.a[3] },
        }
    }

    /// Return matrix diagonal as a vector.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2df32};
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let v = m.diagonal();
    ///
    /// assert_eq!(v, Vector2df32{ x: 2.0, y: 11.0 });
    /// ```
    pub fn diagonal(self) -> Vector2d<T> {
        Vector2d::<T> { x: self.a[0], y: self.a[3] }
    }
}

// **** abs ****

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Return a copy of the matrix with all elements set to their absolute values.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, -17.0,
    ///                               5.0, -11.0]);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix2x2f32::from([  2.0, 17.0,
    ///                                     5.0, 11.0]));
    /// ```
    #[inline]
    pub fn abs(self) -> Self {
        T::m2x2_abs(self)
    }

    /// Set all elements of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, -17.0,
    ///                                   5.0, -11.0]);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  2.0, 17.0,
    ///                                     5.0, 11.0]));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m2x2_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix2x2<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all elements clamped to the specified range.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = m.clamp(2.5, 7.5);
    ///
    /// assert_eq!(n, Matrix2x2f32::from([ 2.5, 7.5,
    ///                                    5.0, 7.5]));
    /// ```
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut a = self.a;
        for it in &mut a {
            *it = it.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all elements of the matrix to the specified range.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0,  3.0,
    ///                                  7.0, 11.0]);
    /// m.clamp_in_place(2.5, 7.5);
    ///
    /// assert_eq!(m, Matrix2x2f32::from([ 2.5, 3.0,
    ///                                    7.0, 7.5]));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix2x2f32::from([  2.0,  5.0,
    ///                                    17.0, 11.0]));
    /// ```
    #[inline]
    pub fn transpose(self) -> Self {
        Self { a: [self.a[0], self.a[2], self.a[1], self.a[3]] }
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix2x2f32::from([  2.0,  5.0,
    ///                                    17.0, 11.0]));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Matrix2x2Math,
{
    /// Return the adjugate of this matrix, ie the transpose of the cofactor matrix.
    /// Equivalent to the inverse but without dividing by the determinant of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = m.adjugate();
    ///
    /// assert!((n*m/m.determinant()).is_near_identity());
    /// ```
    #[inline]
    pub fn adjugate(self) -> Self {
        T::m2x2_adjugate(self)
    }

    /// Adjugate matrix, in-place.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let mut n = m;
    /// n.adjugate_in_place();
    ///
    /// assert_eq!(m.adjugate(), n);
    /// ```
    #[inline]
    pub fn adjugate_in_place(&mut self) -> &mut Self {
        *self = self.adjugate();
        self
    }
    /// Return the inverse of this matrix. Does not check if the determinant is non-zero before inverting.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let n = m.inverse();
    ///
    /// ```
    #[inline]
    pub fn inverse(self) -> Self {
        let adjugate = self.adjugate();
        let determinant = self.determinant();
        adjugate / determinant
    }

    /// Invert this matrix, in-place. Does not check if the determinant is non-zero before inverting.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let mut m = Matrix2x2f32::from([  2.0, 17.0,
    ///                                   5.0, 11.0]);
    /// m.invert_in_place();
    /// ```
    #[inline]
    pub fn invert_in_place(&mut self) -> &mut Self {
        let adjugate = self.adjugate();
        let determinant = self.determinant();
        *self = adjugate / determinant;
        self
    }

    /// Matrix determinant.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let d = m.determinant();
    ///
    /// assert_eq!(2.0*11.0 - 17.0*5.0, d);
    ///
    /// ```
    #[inline]
    pub fn determinant(self) -> T {
        T::m2x2_determinant(self)
    }

    /// Return trace of matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let trace = m.trace();
    ///
    /// assert_eq!(trace, 13.0);
    /// ```
    #[inline]
    pub fn trace(self) -> T {
        T::m2x2_trace(self)
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy + Zero + One + Matrix2x2Math + MathConstants + PartialOrd + Signed,
{
    /// Return inverse of matrix or `T::zero()` if not invertible.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::Zero;
    /// let m = Matrix2x2f32::from([  2.0,  3.0,
    ///                               7.0, 10.5]);
    /// let n = m.inverse_or_zero();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(Matrix2x2f32::zero(), n);
    ///
    /// ```
    pub fn inverse_or_zero(self) -> Self {
        let determinant = self.determinant();
        if determinant.abs() < T::EPSILON {
            return Self::zero();
        }
        let adjugate = self.adjugate();
        adjugate / determinant
    }

    /// Return inverse of matrix or `None` if not invertible.
    /// ```
    /// # use vqm::{Matrix2x2f32};
    /// let m = Matrix2x2f32::from([  2.0,  3.0,
    ///                               7.0, 10.5]);
    /// let n = m.try_inverse();
    ///
    /// assert_eq!(0.0, m.determinant());
    /// assert_eq!(None, n);
    ///
    /// ```
    pub fn try_inverse(self) -> Option<Self> {
        let determinant = self.determinant();
        if determinant.abs() < T::EPSILON {
            return None;
        }
        let adjugate = self.adjugate();
        Some(adjugate / determinant)
    }

    /// Return the sum of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let s = m.sum();
    ///
    /// assert_eq!(s, 35.0);
    /// ```
    #[inline]
    pub fn sum(self) -> T {
        T::m2x2_sum(self)
    }

    /// Return the mean of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let mean = m.mean();
    ///
    /// assert_eq!(mean, 35.0 / 4.0);
    /// ```
    #[inline]
    pub fn mean(self) -> T {
        T::m2x2_mean(self)
    }

    /// Return the product of all elements of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let product = m.product();
    ///
    /// assert_eq!(product, 1870.0);
    /// ```
    #[inline]
    pub fn product(self) -> T {
        T::m2x2_product(self)
    }

    /// Return the sum of the squares of the trace of the matrix.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// let t = m.trace_sum_squares();
    ///
    /// assert_eq!(t, 2.0 * 2.0 + 11.0 * 11.0);
    /// ```
    #[inline]
    pub fn trace_sum_squares(self) -> T {
        T::m2x2_trace_sum_squares(self)
    }

    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::Zero;
    /// let z = Matrix2x2f32::zero();
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
    /// # use vqm::Matrix2x2f32;
    /// # use num_traits::One;
    /// let i = Matrix2x2f32::one();
    /// assert!(i.is_near_identity());
    /// ```
    pub fn is_near_identity(self) -> bool {
        if self.a[1].abs() > T::EPSILON || self.a[2].abs() > T::EPSILON {
            return false;
        }
        if (self.a[0] - T::one()).abs() > T::EPSILON || (self.a[3] - T::one()).abs() > T::EPSILON {
            return false;
        }
        true
    }
}

// **** Iterators ****

impl<T> Matrix2x2<T> {
    /// Returns an iterator over the rows of the matrix as slices of 2 elements.
    #[inline]
    pub fn rows(&self) -> ChunksExact<'_, T> {
        self.chunks_exact(2)
    }
}

impl<T> Matrix2x2<T> {
    /// Returns an iterator over the rows of the matrix as mutable slices of 2 elements.
    #[inline]
    pub fn rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.chunks_exact_mut(2)
    }
}

impl<T> Matrix2x2<T> {
    /// Consumes the matrix and returns an array of its 2 rows.
    #[allow(clippy::many_single_char_names)]
    #[inline]
    pub fn into_rows(self) -> [[T; 2]; 2] {
        let [a, b, c, d] = self.a;
        [[a, b], [c, d]]
    }
}

impl<T> Matrix2x2<T>
where
    T: Copy,
{
    /// Returns an iterator over the columns of the matrix as owned 2-element arrays.
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = [T; 2]> {
        // Create an iterator over the column indices (0, 1)
        (0..2).map(|c| {
            // Collect the strided elements for the current column
            [self.a[c], self.a[c + 2]]
        })
    }
}

impl<'a, T> IntoIterator for &'a Matrix2x2<T> {
    type Item = &'a [T];
    type IntoIter = ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages your Deref trait automatically to get slice chunks
        self.chunks_exact(2)
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix2x2<T> {
    type Item = &'a mut [T];
    type IntoIter = ChunksExactMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages your DerefMut trait automatically
        self.chunks_exact_mut(2)
    }
}

impl<T> IntoIterator for Matrix2x2<T> {
    type Item = [T; 2];
    type IntoIter = core::array::IntoIter<[T; 2], 2>;

    #[allow(clippy::many_single_char_names)]
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let [a, b, c, d] = self.a;
        // Construct an array of rows, then convert that array into an iterator
        [[a, b], [c, d]].into_iter()
    }
}

// **** From ****

// **** From Array ****

impl<T> From<[T; 4]> for Matrix2x2<T>
where
    T: Copy,
{
    /// Matrix from 1D array.
    /// ```
    /// # use vqm::{Matrix2x2f32};
    /// let m = Matrix2x2f32::from([  2.0, 17.0,
    ///                               5.0, 11.0]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0 ]));
    /// ```
    #[inline]
    fn from(input: [T; 4]) -> Self {
        Self { a: input }
    }
}

impl<T> From<[[T; 2]; 2]> for Matrix2x2<T>
where
    T: Copy,
{
    /// Matrix from 2D array.
    /// ```
    /// # use vqm::{Matrix2x2f32};
    /// let m = Matrix2x2f32::from([[  2.0, 17.0],
    ///                             [  5.0, 11.0]]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0 ]));
    /// ```
    #[inline]
    fn from(a: [[T; 2]; 2]) -> Self {
        Self {
            a: [
                a[0][0], a[0][1], //
                a[1][0], a[1][1], //
            ],
        }
    }
}

impl<T> From<[Vector2d<T>; 2]> for Matrix2x2<T>
where
    T: Copy,
{
    /// Matrix from array of vectors.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2df32};
    /// let m = Matrix2x2f32::from([ Vector2df32::new(2.0, 17.0),
    ///                              Vector2df32::new(5.0, 11.0) ]);
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0 ]));
    /// ```
    #[inline]
    fn from(v: [Vector2d<T>; 2]) -> Self {
        Self {
            a: [
                v[0].x, v[0].y, //
                v[1].x, v[1].y, //
            ],
        }
    }
}

impl<T> From<(Vector2d<T>, Vector2d<T>)> for Matrix2x2<T> {
    /// Matrix from tuple of vectors.
    /// ```
    /// # use vqm::{Matrix2x2f32,Vector2df32};
    /// let m = Matrix2x2f32::from(( Vector2df32::new(2.0, 17.0),
    ///                              Vector2df32::new(5.0, 11.0) ));
    /// assert_eq!(m, Matrix2x2f32::new([  2.0, 17.0,
    ///                                    5.0, 11.0 ]));
    /// ```
    #[inline]
    fn from(v: (Vector2d<T>, Vector2d<T>)) -> Self {
        Self {
            a: [
                v.0.x, v.0.y, //
                v.1.x, v.1.y, //
            ],
        }
    }
}
