use core::fmt;
use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    RangeInclusive, Sub, SubAssign,
};
use num_traits::{ConstOne, ConstZero, MulAdd, MulAddAssign, One, Zero, float::FloatCore};

use crate::{MathConstants, Matrix2x2, Matrix3x3, Matrix3x3Math, Matrix4x4, Matrix9x9, Matrix9x9Math, Vector3};

/// 9x9 matrix of `f32` values<br>
pub type Matrix9f32 = Matrix9<f32>;
/// 9x9 matrix of `f64` values<br><br>
pub type Matrix9f64 = Matrix9<f64>;

// **** Define ****

/// `Matrix9<T>`: 9x9 Matrix of type `T`.<br>
/// Provided to support Kalman filter matrix math and so not all functions are provided.<br>
/// In particular matrix by matrix multiply, determinant, adjugate, and inverse are not provided.<br>
/// Functions to extract and utilize 3x3 sub-matrices are provided.<br>
/// Aliases `Matrix9f32` and `Matrix9f64` are provided.<br>
/// Internal implementation is a flat array of `Matrix3x3`.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Matrix9<T> {
    // Flattened array of Matrix3x3: 9 elements in column-major order
    pub(crate) a: [Matrix3x3<T>; 9],
}

impl<T> Default for Matrix9<T>
where
    T: Copy + Zero,
    Matrix3x3<T>: Zero,
{
    fn default() -> Self {
        Self { a: [Matrix3x3::<T>::zero(); 9] }
    }
}

impl<T: fmt::Debug> fmt::Debug for Matrix9<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix9x9 [")?; // Start the struct block wrapper
        // Loop over rows using Deref slice chunking behavior.
        for row in self.chunks_exact(9) {
            write!(f, "    ")?; // Print 4 spaces of indentation for clean alignment
            fmt::Debug::fmt(row, f)?; // Format the row elements neatly as a standard array slice
            writeln!(f, ",")?;
        }
        write!(f, "]") // Close the struct block wrapper
    }
}

/// Constants to index matrix elements.
#[allow(missing_docs)]
impl<T> Matrix9<T> {
    pub const SIZE: usize = 9;
    pub const ROW_COUNT: usize = 3;
    pub const COL_COUNT: usize = 3;
    // Column 1
    pub const M11: usize = 0;
    pub const M21: usize = 1;
    pub const M31: usize = 2;
    // Column 2
    pub const M12: usize = 3;
    pub const M22: usize = 4;
    pub const M32: usize = 5;
    // Column 3
    pub const M13: usize = 6;
    pub const M23: usize = 7;
    pub const M33: usize = 8;
}

// **** New ****

impl<T> Matrix9<T>
where
    T: Copy,
{
    /// Constructor.
    #[inline]
    pub fn new(a: [T; 81]) -> Self {
        let m9x9 = Matrix9x9::new(a);
        Matrix9::from(m9x9)
    }
}

// **** Other constructors ****

impl<T> Matrix9<T>
where
    T: Copy,
{
    /// Create a matrix with all its elements set to a single value.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(2.0);
    /// assert_eq!(2.0, m[2][3]);
    /// ```
    pub fn from_element(value: T) -> Self {
        Self { a: [Matrix3x3::<T>::from_element(value); 9] }
    }
    /// Matrix from 1D row array.
    #[inline]
    pub const fn from_row_array(a: [Matrix3x3<T>; 9]) -> Self {
        Self {
            a: [
                a[0], a[3], a[6], //
                a[1], a[4], a[7], //
                a[2], a[5], a[8], //
            ],
        }
    }

    /// Matrix from 1D column array.
    #[inline]
    pub const fn from_column_array(a: [Matrix3x3<T>; 9]) -> Self {
        Self { a }
    }

    /// Matrix from 2D row array.
    #[inline]
    pub const fn from_2d_row_array(a: [[Matrix3x3<T>; 3]; 3]) -> Self {
        Self {
            a: [
                a[0][0], a[1][0], a[2][0], //
                a[0][1], a[1][1], a[2][1], //
                a[0][2], a[1][2], a[2][2], //
            ],
        }
    }
}

impl<T> Matrix9<T>
where
    T: Copy + ConstZero,
{
    /// Create a matrix with the diagonal set to a single value.
    /// ```
    /// # use vqm::{Matrix9f32, Matrix3x3f32};
    /// let m = Matrix9f32::from_diagonal_element(2.0);
    /// assert_eq!(m[Matrix3x3f32::M22], Matrix3x3f32::new([ 2.0, 0.0, 0.0,
    ///                                                      0.0, 2.0, 0.0,
    ///                                                      0.0, 0.0, 2.0]));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn from_diagonal_element(value: T) -> Self {
        Self { a: [Matrix3x3::<T>::from_diagonal_element(value); 9] }
    }
}

// **** Zero ****

impl<T> Zero for Matrix9<T>
where
    T: Copy + Zero + PartialEq + Matrix9x9Math,
    Matrix3x3<T>: Zero,
{
    /// Zero matrix.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::Zero;
    /// let z = Matrix9f32::zero();
    /// assert!(z.is_zero());
    /// ```
    #[inline]
    fn zero() -> Self {
        Self { a: [Matrix3x3::<T>::zero(); 9] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T> ConstZero for Matrix9<T>
where
    T: Copy + ConstZero + PartialEq + Matrix9x9Math,
    Matrix3x3<T>: Zero + ConstZero,
{
    /// Const zero matrix.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::{zero,Zero,ConstZero};
    /// let m = Matrix9f32::ZERO;
    /// assert!(m.is_zero());
    /// ```
    const ZERO: Self = Self { a: [Matrix3x3::<T>::ZERO; 9] };
}

impl<T> Matrix9<T>
where
    T: Copy + FloatCore,
{
    /// Return true if matrix is near zero.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::Zero;
    /// let z = Matrix9f32::zero();
    /// assert!(z.is_near_zero(1e-5));
    /// ```
    pub fn is_near_zero(self, epsilon: T) -> bool {
        for a in &self.a {
            if !a.is_near_zero(epsilon) {
                return false;
            }
        }
        true
    }
}

// **** One ****

impl<T> One for Matrix9<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix9x9Math,
    Matrix3x3<T>: ConstOne + ConstZero,
{
    /// Identity matrix.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::One;
    /// let i = Matrix9f32::one();
    ///
    /// assert!(i.is_one());
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

impl<T> ConstOne for Matrix9<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix9x9Math,
    Matrix3x3<T>: ConstOne + ConstZero,
{
    /// Const identity matrix.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::{ConstOne, One};
    /// let i = Matrix9f32::ONE;
    ///
    /// assert!(i.is_one());
    /// ```
    #[rustfmt::skip]
    const ONE: Self = Self {
        a: [
            Matrix3x3::<T>::ONE,  Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ZERO,
            Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ONE,  Matrix3x3::<T>::ZERO,
            Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ONE,
        ]
    };
}

impl<T> Matrix9<T>
where
    T: Copy + Zero + One,
    Matrix3x3<T>: ConstOne + ConstZero,
{
    /// Identity matrix.
    /// Alias for `one()` that does not require `num_traits::One`.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let i = Matrix9f32::identity();
    /// ```
    #[rustfmt::skip]
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
    Self {
        a: [
            Matrix3x3::<T>::ONE,  Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ZERO,
            Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ONE,  Matrix3x3::<T>::ZERO,
            Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ZERO, Matrix3x3::<T>::ONE,
        ],
    }
    }
}

impl<T> Matrix9<T>
where
    T: Copy + FloatCore,
    Matrix9<T>: One + Sub<Output = Matrix9<T>>,
{
    /// Return true if matrix is near identity.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::One;
    /// let i = Matrix9f32::one();
    /// assert!(i.is_near_identity(1e-5));
    /// ```
    pub fn is_near_identity(self, epsilon: T) -> bool {
        (self - Matrix9::<T>::one()).is_near_zero(epsilon)
    }
}

// **** Neg ****

impl<T> Neg for Matrix9<T>
where
    T: Copy,
    Matrix3x3<T>: Neg<Output = Matrix3x3<T>>,
{
    type Output = Self;

    /// Negate matrix.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// m = - m;
    /// ```
    #[inline]
    fn neg(self) -> Self {
        let mut ret = self;
        for ii in 0..9 {
            ret.a[ii] = -ret.a[ii];
        }
        ret
    }
}

// **** Add ****

impl<T> Add for Matrix9<T>
where
    T: Copy,
    Matrix3x3<T>: Copy + Add<Output = Matrix3x3<T>>,
{
    type Output = Self;

    /// Add two matrices.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(2.0);
    /// let n = Matrix9f32::from_element(3.0);
    /// let r = m + n;
    ///
    ///
    /// # use num_traits::Zero;
    ///
    /// let z = Matrix9f32::zero();
    /// let r2 = m + z;
    ///
    /// assert_eq!(r2, m);
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        let mut ret = self;
        for ii in 0..9 {
            ret.a[ii] = ret.a[ii] + other.a[ii];
        }
        ret
    }
}

// **** AddAssign ****

impl<T> AddAssign for Matrix9<T>
where
    T: Copy + Matrix9x9Math,
    Matrix3x3<T>: Add<Output = Matrix3x3<T>> + Copy,
{
    /// Add one matrix to another.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// let n = Matrix9f32::from_element(3.0);
    /// m += n;
    ///
    /// assert_eq!(m, Matrix9f32::from_element(5.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        for ii in 0..9 {
            self.a[ii] = self.a[ii] + other.a[ii];
        }
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Matrix9<T>
where
    T: Copy + Matrix3x3Math,
    Matrix3x3<T>: Neg,
{
    type Output = Self;

    /// Multiply matrix by constant and add another matrix.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::MulAdd;
    /// let m = Matrix9f32::from_element(2.0);
    /// let n = Matrix9f32::from_element(3.0);
    /// let k = 5.0;
    /// let r = m.mul_add(k, n);
    ///
    /// assert_eq!(r, Matrix9f32::from_element(13.0));
    /// ```
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        let mut ret = self;
        for ii in 0..9 {
            ret.a[ii] = ret.a[ii] * k + other.a[ii];
        }
        ret
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Matrix9<T>
where
    T: Copy + Matrix9x9Math,
    Matrix3x3<T>: Mul<T, Output = Matrix3x3<T>> + Add<Matrix3x3<T>, Output = Matrix3x3<T>>,
{
    /// Multiply matrix by constant and add another matrix in place.
    /// ```
    /// # use vqm::Matrix9f32;
    /// # use num_traits::MulAddAssign;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// let n = Matrix9f32::from_element(3.0);
    /// let k = 5.0;
    /// m.mul_add_assign(k, n);
    ///
    /// assert_eq!(m, Matrix9f32::from_element(13.0));
    /// ```
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        for ii in 0..9 {
            self.a[ii] = self.a[ii] * k + other.a[ii];
        }
    }
}

// **** Sub ****

impl<T> Sub for Matrix9<T>
where
    T: Copy,
    Matrix3x3<T>: Neg<Output = Matrix3x3<T>>,
    Matrix9<T>: Add<Output = Matrix9<T>>,
{
    type Output = Self;

    /// Subtract two matrices.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(2.0);
    /// let n = Matrix9f32::from_element(3.0);
    /// let r = m - n;
    ///
    /// assert_eq!(r, Matrix9f32::from_element(-1.0));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix9<T>
where
    T: Copy,
    Matrix9<T>: Sub<Output = Matrix9<T>>,
{
    /// Subtract one matrix from another.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// let n = Matrix9f32::from_element(3.0);
    /// m -= n;
    ///
    /// assert_eq!(m, Matrix9f32::from_element(-1.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(other);
    }
}

// **** Pre-multiply ****

impl Mul<Matrix9<f32>> for f32 {
    type Output = Matrix9<f32>;

    /// Pre-multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(2.0);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul(self, other: Matrix9<f32>) -> Matrix9<f32> {
        let mut ret = other;
        for ii in 0..9 {
            ret.a[ii] *= self;
        }
        ret
    }
}

impl Mul<Matrix9<f64>> for f64 {
    type Output = Matrix9<f64>;

    /// Pre-multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(2.0);
    /// let r = 2.0 * m;
    ///
    /// assert_eq!(r, Matrix9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul(self, other: Matrix9<f64>) -> Matrix9<f64> {
        let mut ret = other;
        for ii in 0..9 {
            ret.a[ii] *= self;
        }
        ret
    }
}

// **** Mul ****

impl<T> Mul<T> for Matrix9<T>
where
    T: Copy + Matrix9x9Math,
    Matrix3x3<T>: Mul<T, Output = Matrix3x3<T>>,
{
    type Output = Self;

    /// Multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(2.0);
    /// let r = m * 2.0;
    ///
    /// assert_eq!(r, Matrix9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul(self, other: T) -> Self {
        let mut ret = self;
        for ii in 0..9 {
            ret.a[ii] = ret.a[ii] * other;
        }
        ret
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Matrix9<T>
where
    T: Copy + Matrix9x9Math,
    Matrix3x3<T>: Mul<T, Output = Matrix3x3<T>>,
{
    /// In-place multiply a matrix by a constant.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// m *= 2.0;
    ///
    /// assert_eq!(m, Matrix9f32::from_element(4.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T> Mul<Matrix9<T>> for Matrix9<T>
where
    T: Copy + Matrix9x9Math,
    Matrix3x3<T>: Add<Output = Matrix3x3<T>> + Mul<Output = Matrix3x3<T>>,
    Matrix9<T>: Zero,
{
    type Output = Self;

    /// Multiply two matrices.
    #[inline]
    fn mul(self, other: Self) -> Self {
        let mut result = Self::zero();

        for col in 0..Self::COL_COUNT {
            let c = col * Self::ROW_COUNT;

            for r in 0..Self::ROW_COUNT {
                result.a[c + r] =
                    self.a[r] * other.a[c] + self.a[r + 3] * other.a[c + 1] + self.a[r + 6] * other.a[c + 2];
            }
        }
        result
    }
}

// **** Outer Product ****

impl<T> Matrix9<T>
where
    T: Copy + Zero + PartialEq + Mul<Output = T> + Matrix9x9Math,
    Matrix9<T>: Zero,
{
    /// Calculates the outer product of two 9-element states for COLUMN-MAJOR matrices (Cortex-M Edition).
    #[inline]
    pub fn outer_product(
        col_a: Vector3<T>,
        col_b: Vector3<T>,
        col_c: Vector3<T>,
        row_a: Vector3<T>,
        row_b: Vector3<T>,
        row_c: Vector3<T>,
    ) -> Matrix9<T> {
        let m9x9 = Matrix9x9::outer_product(col_a, col_b, col_c, row_a, row_b, row_c);
        Matrix9::from(m9x9)
    }
}

// **** Div ****

impl<T> Div<T> for Matrix9<T>
where
    T: Copy + One + Div<Output = T>,
    Matrix9<T>: Mul<T, Output = Matrix9<T>>,
{
    type Output = Self;

    /// Divide a matrix by a constant.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(2.0);
    /// let r = m / 2.0;
    ///
    /// assert_eq!(r, Matrix9f32::from_element(1.0));
    /// ```
    #[inline]
    fn div(self, other: T) -> Self {
        let r = T::one() / other;
        self * r
    }
}

// **** DivAssign ****

impl<T> DivAssign<T> for Matrix9<T>
where
    T: Copy + One + Div,
    Matrix9<T>: Div<T, Output = Matrix9<T>>,
{
    /// In-place divide a matrix by a constant.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// m /= 2.0;
    ///
    /// assert_eq!(m, Matrix9f32::from_element(1.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** AsRef ****

impl<T> AsRef<[Matrix3x3<T>; 9]> for Matrix9<T> {
    /// Immutable reference to the raw array.
    /// ```
    /// # use vqm::{Matrix9f32, Matrix3x3f32};
    /// let mut m = Matrix9f32::from_element(2.0);
    /// let a: &[Matrix3x3f32; 9] = m.as_ref();
    /// assert_eq!(2.0, a[Matrix9f32::M21][Matrix9f32::M11]);
    /// ```
    #[inline]
    fn as_ref(&self) -> &[Matrix3x3<T>; 9] {
        &self.a
    }
}

impl<T> AsMut<[Matrix3x3<T>; 9]> for Matrix9<T> {
    /// Mutable reference to the raw array.
    /// ```
    /// # use vqm::{Matrix9f32, Matrix3x3f32};
    /// let mut m = Matrix9f32::from_element(2.0);
    /// let a: &mut [Matrix3x3f32; 9] = m.as_mut();
    /// a[4][3] = 7.0;
    /// assert_eq!(7.0, m[4][3]);
    /// ```
    #[inline]
    fn as_mut(&mut self) -> &mut [Matrix3x3<T>; 9] {
        &mut self.a
    }
}

// **** Deref ****

impl<T> Deref for Matrix9<T> {
    type Target = [Matrix3x3<T>];

    #[inline]
    fn deref(&self) -> &[Matrix3x3<T>] {
        &self.a
    }
}

impl<T> DerefMut for Matrix9<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Matrix3x3<T>] {
        &mut self.a
    }
}

// **** Index ****

impl<T> Index<usize> for Matrix9<T> {
    type Output = Matrix3x3<T>;

    /// Access matrix element by index.
    #[inline]
    fn index(&self, index: usize) -> &Matrix3x3<T> {
        &self.a[index]
    }
}

impl<T> Index<Range<usize>> for Matrix9<T> {
    type Output = [Matrix3x3<T>];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[Matrix3x3<T>] {
        &self.a[index]
    }
}

impl<T> Index<RangeFull> for Matrix9<T> {
    type Output = [Matrix3x3<T>];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[Matrix3x3<T>] {
        &self.a
    }
}

impl<T> Index<RangeInclusive<usize>> for Matrix9<T> {
    type Output = [Matrix3x3<T>];

    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &[Matrix3x3<T>] {
        &self.a[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix9<T> {
    type Output = Matrix3x3<T>;

    /// Access matrix element by ordered pair (row, column).
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 3 && col < 3, "Matrix index out of bounds: row={row}, col={col}");
        &self.a[col * 3 + row]
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Matrix9<T> {
    /// Set matrix element by index.
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Matrix3x3<T> {
        &mut self.a[index]
    }
}

impl<T> IndexMut<Range<usize>> for Matrix9<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [Matrix3x3<T>] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<RangeFull> for Matrix9<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [Matrix3x3<T>] {
        &mut self.a
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Matrix9<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut [Matrix3x3<T>] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix9<T> {
    /// Set matrix element by ordered pair (row, column).
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Matrix3x3<T> {
        assert!(row < 3 && col < 3, "Matrix index out of bounds: row={row}, col={col}");
        &mut self.a[col * 3 + row]
    }
}

impl<T> Matrix9<T>
where
    T: Copy,
{
    /// Return matrix diagonal as an array.
    /// ```
    /// # use vqm::Matrix9f32;
    ///
    /// let m = Matrix9f32::from_element(2.0);
    /// let a = m.diagonal_as_array();
    ///
    /// assert_eq!(2.0, a[Matrix9f32::M11][Matrix9f32::M22]);
    /// ```
    pub fn diagonal_as_array(self) -> [Matrix3x3<T>; 3] {
        [
            Matrix3x3 { a: self.a[Self::M11].a },
            Matrix3x3 { a: self.a[Self::M22].a },
            Matrix3x3 { a: self.a[Self::M33].a },
        ]
    }
}

// **** abs ****

impl<T> Matrix9<T>
where
    T: Copy + Matrix3x3Math,
    Matrix3x3<T>: Copy + Neg,
    Matrix9<T>: Copy + Neg,
{
    /// Return a copy of the matrix with all elements set to their absolute values.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(-2.0);
    /// let n = m.abs();
    ///
    /// assert_eq!(n, Matrix9f32::from_element(2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        let mut ret = self;
        for ii in 0..9 {
            ret.a[ii] = ret.a[ii].abs();
        }
        ret
    }

    /// Set all elements of the matrix to their absolute values.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(-2.0);
    /// m.abs_in_place();
    ///
    /// assert_eq!(m, Matrix9f32::from_element(2.0));
    /// ```
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = self.abs();
        self
    }
}

// **** clamp ****

impl<T> Matrix9<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all elements clamped to the specified range.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let m = Matrix9f32::from_element(-2.0);
    ///
    /// let n = m.clamp(7.0, 17.0);
    ///
    /// assert_eq!(n, Matrix9f32::from_element(7.0));
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
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(-2.0);
    /// m.clamp_in_place(7.0, 17.0);
    ///
    /// assert_eq!(m, Matrix9f32::from_element(7.0));
    /// ```
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix9<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// let n = m.transpose();
    ///
    /// assert_eq!(n, Matrix9f32::from_element(2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self { a: [self.a[0], self.a[3], self.a[6], self.a[1], self.a[4], self.a[7], self.a[2], self.a[5], self.a[8]] }
    }

    /// Transpose matrix, in-place.
    /// ```
    /// # use vqm::Matrix9f32;
    /// let mut m = Matrix9f32::from_element(2.0);
    /// m.transpose_in_place();
    ///
    /// assert_eq!(m, Matrix9f32::from_element(2.0));
    /// ```
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix9<T>
where
    T: Copy + Add<Output = T> + Matrix3x3Math,
{
    /// Return trace of matrix.
    #[inline]
    pub fn trace(self) -> T {
        self.a[Self::M11].trace() + self.a[Self::M22].trace() + self.a[Self::M33].trace()
    }
}

impl<T> Matrix9<T>
where
    T: Copy + Zero + One + Matrix3x3Math + MathConstants + PartialOrd + FloatCore,
{
    /// Return the sum of all elements of the matrix.
    #[inline]
    pub fn sum(self) -> T {
        let mut sum: T = T::zero();
        for a in self.a {
            sum = sum + a.sum();
        }
        sum
    }

    /// Return the product of all elements of the matrix.
    #[inline]
    pub fn product(self) -> T {
        let mut product: T = T::one();
        for a in self.a {
            product = product * a.product();
        }
        product
    }
}

// **** Symmetry ****

impl<T> Matrix9<T>
where
    T: Copy + One + Add<Output = T> + Div<Output = T> + Matrix3x3Math,
    Matrix3x3<T>: Add<Output = Matrix3x3<T>>,
{
    /// Enforces strict mathematical symmetry on the matrix in-place.
    /// Used so that rounding errors do not erode the symmetry of the matrix.
    /// Formula: `M = (M + Mᵀ) / 2`.
    pub fn enforce_symmetry(&mut self) {
        let half = T::one() / (T::one() + T::one());

        self.a[Self::M12] = (self.a[Self::M12] + self.a[Self::M21]) * half;
        self.a[Self::M21] = self.a[Self::M12];

        self.a[Self::M13] = (self.a[Self::M13] + self.a[Self::M31]) * half;
        self.a[Self::M31] = self.a[Self::M13];

        self.a[Self::M23] = (self.a[Self::M23] + self.a[Self::M32]) * half;
        self.a[Self::M32] = self.a[Self::M23];
    }
}

// **** Iterators ****

impl<T> Matrix9<T>
where
    T: Copy,
{
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = [T; 9]> {
        (0..9).map(|col| self.column(col))
    }

    #[inline]
    pub fn column(&self, col: usize) -> [T; 9] {
        debug_assert!(col < 9);
        let block_col = col / 3;
        let local_col = col % 3;
        [
            self.a[block_col * 3][local_col * 3],
            self.a[block_col * 3][local_col * 3 + 1],
            self.a[block_col * 3][local_col * 3 + 2],
            self.a[block_col * 3 + 1][local_col * 3],
            self.a[block_col * 3 + 1][local_col * 3 + 1],
            self.a[block_col * 3 + 1][local_col * 3 + 2],
            self.a[block_col * 3 + 2][local_col * 3],
            self.a[block_col * 3 + 2][local_col * 3 + 1],
            self.a[block_col * 3 + 2][local_col * 3 + 2],
        ]
    }
}

// **** Column Iterators ****

impl<T> Matrix9<T> {
    #[inline]
    pub fn iter_columns(&self) -> Matrix9Columns<'_, T> {
        Matrix9Columns::new(self)
    }
}

// **** Iterator Pairs ****

/// A custom iterator over the read-only columns of a 9x9 matrix.
#[derive(Debug)]
pub struct Matrix9Columns<'a, T> {
    matrix: &'a Matrix9<T>,
    index: usize,
}

impl<'a, T> Matrix9Columns<'a, T> {
    #[inline]
    fn new(matrix: &'a Matrix9<T>) -> Self {
        Self { matrix, index: 0 }
    }
}

impl<T: Copy> Iterator for Matrix9Columns<'_, T> {
    type Item = [T; 9];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= 9 {
            return None;
        }
        let c = self.index;
        self.index += 1;
        let block_col = c / 3;
        let col = c % 3;
        let top = &self.matrix.a[block_col * 3];
        let middle = &self.matrix.a[block_col * 3 + 1];
        let bottom = &self.matrix.a[block_col * 3 + 2];
        Some([
            top[col],
            top[col + 3],
            top[col + 6],
            middle[col],
            middle[col + 3],
            middle[col + 6],
            bottom[col],
            bottom[col + 3],
            bottom[col + 6],
        ])
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = 9 - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for Matrix9Columns<'_, T> {}

impl<T: Copy> core::iter::FusedIterator for Matrix9Columns<'_, T> {}

// **** From ****

// **** From Matrix ****

impl<T: Copy> From<Matrix9<T>> for Matrix2x2<T> {
    /// Matrix2x2 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9<T>) -> Self {
        Self { a: [
            m.a[0].a[0], m.a[0].a[3],
            m.a[0].a[1], m.a[0].a[4],
        ] }
    }
}

impl<T: Copy> From<Matrix9<T>> for Matrix3x3<T> {
    /// Matrix3x3 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9<T>) -> Self {
        Self { a: [
            m.a[0].a[0], m.a[0].a[3], m.a[0].a[6],
            m.a[0].a[1], m.a[0].a[4], m.a[0].a[7],
            m.a[0].a[2], m.a[0].a[5], m.a[0].a[8],
        ] }
    }
}

impl<T: Copy> From<Matrix9<T>> for Matrix4x4<T> {
    /// Matrix4x4 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9<T>) -> Self {
        Self { a: [
            m.a[0].a[0], m.a[0].a[3], m.a[0].a[6], m.a[3].a[0],
            m.a[0].a[1], m.a[0].a[4], m.a[0].a[7], m.a[3].a[1],
            m.a[0].a[2], m.a[0].a[5], m.a[0].a[8], m.a[3].a[2],
            m.a[1].a[0], m.a[1].a[3], m.a[1].a[6], m.a[4].a[0],
        ] }
    }
}

impl<T: Copy + Zero> From<Matrix9<T>> for Matrix9x9<T> {
    fn from(src: Matrix9<T>) -> Self {
        let mut ret = Self::default();
        for block_col in 0..3 {
            for block_row in 0..3 {
                let block = src[block_col * 3 + block_row];
                for local_col in 0..3 {
                    for local_row in 0..3 {
                        let row = block_row * 3 + local_row;
                        let col = block_col * 3 + local_col;
                        ret[col * 9 + row] = block[local_col * 3 + local_row];
                    }
                }
            }
        }
        ret
    }
}

impl<T: Copy> From<Matrix9x9<T>> for Matrix9<T> {
    fn from(src: Matrix9x9<T>) -> Self {
        Self {
            a: core::array::from_fn(|block| {
                let block_row = block % 3;
                let block_col = block / 3;
                Matrix3x3 {
                    a: core::array::from_fn(|i| {
                        let local_row = i % 3;
                        let local_col = i / 3;
                        let row = block_row * 3 + local_row;
                        let col = block_col * 3 + local_col;
                        src[col * 9 + row]
                    }),
                }
            }),
        }
    }
}

impl<T: Copy> From<[Matrix3x3<T>; 9]> for Matrix9<T> {
    /// Matrix4x4 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: [Matrix3x3<T>;9]) -> Self {
        Self { a: [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        ] }
    }
}
