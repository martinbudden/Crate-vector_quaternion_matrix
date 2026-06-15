use core::fmt;
use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    RangeInclusive, Sub, SubAssign,
};
use core::slice::{ChunksExact, ChunksExactMut};
use num_traits::{ConstZero, MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};
#[cfg(feature = "serde")]
use {
    sequential_storage::map::PostcardValue,
    serde::{Deserialize, Serialize},
};

use crate::{MathConstants, Matrix2x2, Matrix3x3, Matrix4x4, Matrix9x9Math, Vector3d};

/// 9x9 matrix of `f32` values<br>
pub type Matrix9x9f32 = Matrix9x9<f32>;
/// 9x9 matrix of `f64` values<br><br>
pub type Matrix9x9f64 = Matrix9x9<f64>;

// **** Define ****

/// `Matrix9x9<T>`: 9x9 Matrix of type `T`.<br>
/// Provided to support Kalman filter matrix math and so not all functions are provided.<br>
/// In particular matrix by matrix multiply, determinant, adjugate, and inverse are not provided.<br>
/// Functions to extract and utilize 3x3 sub-matrices are provided.<br>
/// Aliases `Matrix9x9f32` and `Matrix9x9f64` are provided.<br>
/// Internal implementation is a flattened 9x9 matrix: an array of 9 elements stored in row-major order.
/// That is the element `m[row][col]` is at array position `[row * 3 + col]`, so element `m12` is at `a[5]`.<br><br>
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Matrix9x9<T> {
    // Flattened 9x9 matrix: 81 elements in row-major order
    pub(crate) a: [T; 81],
}

#[cfg(feature = "serde")]
impl<T> PostcardValue<'_> for Matrix4x4<T> where T: serde::Serialize + for<'de> serde::Deserialize<'de> {}

impl<T> Default for Matrix9x9<T>
where
    T: Copy + Zero,
{
    fn default() -> Self {
        Self { a: [T::zero(); 81] }
    }
}

impl<T> fmt::Debug for Matrix9x9<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Start the struct block wrapper
        writeln!(f, "Matrix9x9 [")?;

        // Loop over rows using the Deref slice chunking behavior we added earlier
        for row in self.chunks_exact(9) {
            // Print 4 spaces of indentation for clean alignment
            write!(f, "    ")?;

            // Format the row elements neatly as a standard array slice
            fmt::Debug::fmt(row, f)?;
            writeln!(f, ",")?;
        }

        // Close the struct block wrapper
        write!(f, "]")
    }
}

/// Constants to index matrix elements.
impl<T> Matrix9x9<T> {
    pub const SIZE: usize = 81;
    pub const ROW_COUNT: usize = 9;
    pub const COL_COUNT: usize = 9;
    // Row 1
    pub const M11: usize = 0;
    pub const M12: usize = 1;
    pub const M13: usize = 2;
    pub const M14: usize = 3;
    pub const M15: usize = 4;
    pub const M16: usize = 5;
    pub const M17: usize = 6;
    pub const M18: usize = 7;
    pub const M19: usize = 8;
    // Row 2
    pub const M21: usize = 9;
    pub const M22: usize = 10;
    pub const M23: usize = 11;
    pub const M24: usize = 12;
    pub const M25: usize = 13;
    pub const M26: usize = 14;
    pub const M27: usize = 15;
    pub const M28: usize = 16;
    pub const M29: usize = 17;
    // Row 3
    pub const M31: usize = 18;
    pub const M32: usize = 19;
    pub const M33: usize = 20;
    pub const M34: usize = 21;
    pub const M35: usize = 22;
    pub const M36: usize = 23;
    pub const M37: usize = 24;
    pub const M38: usize = 25;
    pub const M39: usize = 26;
    // Row 4
    pub const M41: usize = 27;
    pub const M42: usize = 28;
    pub const M43: usize = 29;
    pub const M44: usize = 30;
    pub const M45: usize = 31;
    pub const M46: usize = 32;
    pub const M47: usize = 33;
    pub const M48: usize = 34;
    pub const M49: usize = 35;
    // Row 5
    pub const M51: usize = 36;
    pub const M52: usize = 37;
    pub const M53: usize = 38;
    pub const M54: usize = 39;
    pub const M55: usize = 40;
    pub const M56: usize = 41;
    pub const M57: usize = 42;
    pub const M58: usize = 43;
    pub const M59: usize = 44;
    // Row 6
    pub const M61: usize = 45;
    pub const M62: usize = 46;
    pub const M63: usize = 47;
    pub const M64: usize = 48;
    pub const M65: usize = 49;
    pub const M66: usize = 50;
    pub const M67: usize = 51;
    pub const M68: usize = 52;
    pub const M69: usize = 53;
    // Row 7
    pub const M71: usize = 54;
    pub const M72: usize = 55;
    pub const M73: usize = 56;
    pub const M74: usize = 57;
    pub const M75: usize = 58;
    pub const M76: usize = 59;
    pub const M77: usize = 60;
    pub const M78: usize = 61;
    pub const M79: usize = 62;
    // Row 8
    pub const M81: usize = 63;
    pub const M82: usize = 64;
    pub const M83: usize = 65;
    pub const M84: usize = 66;
    pub const M85: usize = 67;
    pub const M86: usize = 68;
    pub const M87: usize = 69;
    pub const M88: usize = 70;
    pub const M89: usize = 71;
    // Row 9
    pub const M91: usize = 72;
    pub const M92: usize = 73;
    pub const M93: usize = 74;
    pub const M94: usize = 75;
    pub const M95: usize = 76;
    pub const M96: usize = 77;
    pub const M97: usize = 78;
    pub const M98: usize = 79;
    pub const M99: usize = 80;
}

// **** New ****

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Create a matrix.
    #[inline]
    pub const fn new(input: [T; 81]) -> Self {
        Self { a: input }
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Create a matrix filled with a single value.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let m = Matrix9x9f32::fill(2.0);
    /// assert_eq!(2.0, m[9]);
    /// ```
    pub fn fill(value: T) -> Self {
        Self { a: [value; 81] }
    }

    /// Try to create a matrix from a slice.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// let valid_data = [2.0; 81];
    /// let invalid_data = [2.0; 3];
    /// let Some(m) = Matrix9x9f32::try_from_slice(&valid_data) else {
    ///     panic!("Expected Some(Matrix9x9), but got None");
    /// };
    /// assert_eq!(2.0, m[0]);
    /// let None = Matrix9x9f32::try_from_slice(&invalid_data) else {
    ///     panic!("Expected None for invalid data, but got Some");
    /// };
    /// ```
    pub fn try_from_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != 81 {
            return None;
        }
        let mut a = [slice[0]; 81];
        a.copy_from_slice(&slice[0..81]);
        Some(Self { a })
    }
}

// **** Zero ****

impl<T> Zero for Matrix9x9<T>
where
    T: Copy + Zero + PartialEq + Matrix9x9Math,
{
    /// Zero matrix.
    #[inline]
    fn zero() -> Self {
        Self { a: [T::zero(); 81] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<T> ConstZero for Matrix9x9<T>
where
    T: Copy + ConstZero + PartialEq + Matrix9x9Math,
{
    /// Const zero matrix.
    /// ```
    /// # use vqm::Matrix9x9f32;
    /// # use num_traits::{zero,Zero,ConstZero};
    /// let m = Matrix9x9f32::ZERO;
    /// assert!(m.is_zero());
    /// ```
    const ZERO: Self = Self { a: [T::ZERO; 81] };
}

// **** One ****

impl<T> Matrix9x9<T>
where
    T: Copy + Zero + One,
{
    /// Identity matrix.
    /// Alias for `one()` that does not require `num_traits::One`.
    #[inline]
    pub fn identity() -> Self {
        let mut m = Self { a: [T::zero(); 81] };
        for ii in 0..=8 {
            m.a[ii * 9 + ii] = T::one();
        }
        m
    }
}

// **** Neg ****

impl<T> Neg for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Negate matrix.
    #[inline]
    fn neg(self) -> Self {
        T::m9x9_neg(self)
    }
}

// **** Add ****

impl<T> Add for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Add two matrices.
    #[inline]
    fn add(self, other: Self) -> Self {
        T::m9x9_add(self, other)
    }
}

// **** AddAssign ****

impl<T> AddAssign for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Add one matrix to another.
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

// **** MulAdd ****

impl<T> MulAdd<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Multiply matrix by constant and add another matrix.
    #[inline]
    fn mul_add(self, k: T, other: Self) -> Self {
        T::m9x9_mul_add(self, k, other)
    }
}

// **** MulAddAssign ****

impl<T> MulAddAssign<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Multiply matrix by constant and add another matrix in place.
    #[inline]
    fn mul_add_assign(&mut self, k: T, other: Self) {
        *self = self.mul_add(k, other);
    }
}

// **** Sub ****

impl<T> Sub for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Subtract two matrices.
    #[inline]
    fn sub(self, other: Self) -> Self {
        // Reuse SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Subtract one matrix from another.
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

// **** Pre-multiply ****

impl Mul<Matrix9x9<f32>> for f32 {
    type Output = Matrix9x9<f32>;

    /// Pre-multiply a matrix by a constant.
    #[inline]
    fn mul(self, other: Matrix9x9<f32>) -> Matrix9x9<f32> {
        f32::m9x9_mul_scalar(other, self)
    }
}

impl Mul<Matrix9x9<f64>> for f64 {
    type Output = Matrix9x9<f64>;
    #[inline]
    fn mul(self, other: Matrix9x9<f64>) -> Matrix9x9<f64> {
        f64::m9x9_mul_scalar(other, self)
    }
}

// **** Mul ****

impl<T> Mul<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Multiply a matrix by a constant.
    #[inline]
    fn mul(self, other: T) -> Self {
        T::m9x9_mul_scalar(self, other)
    }
}

// **** MulAssign ****

impl<T> MulAssign<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// In-place multiply a matrix by a constant.
    #[inline]
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + Zero + Matrix9x9Math + Mul<T, Output = T>,
{
    // Extract the first 3 columns of the 9x9 matrix as an array of 27 elements.
    pub fn extract_9x3_array(&self) -> [T; 27] {
        let mut ret = [T::zero(); 27];
        for r in 0..9 {
            let offset9 = r * 9;
            let offset3 = r * 3;
            ret[offset3] = self.a[offset9]; // Column 1
            ret[offset3 + 1] = self.a[offset9 + 1]; // Column 2
            ret[offset3 + 2] = self.a[offset9 + 2]; // Column 3
        }
        ret
    }

    /// Multiplies the first 3 columns of lhs (a 9x3 sub-matrix) by rhs.
    /// Returns a tuple of three 3x3 matrices.
    #[inline]
    pub fn multiply_9x3_array_by_3x3(lhs: [T; 27], rhs: Matrix3x3<T>) -> (Matrix3x3<T>, Matrix3x3<T>, Matrix3x3<T>) {
        // Helper closure to calculate a single 3x3 block from a specific row slice.
        #[rustfmt::skip]
        let multiply_block = |start_row: usize| -> Matrix3x3<T> {
            let mut ret = [T::zero(); 9];
            for r in 0..3 {
                let l_offset = (start_row + r) * 3;
                let l1 = lhs[l_offset];
                let l2 = lhs[l_offset + 1];
                let l3 = lhs[l_offset + 2];

                // Calculates row dot products with the 3 columns of `rhs`
                let ret_offset = r * 3;
                ret[ret_offset] =     l1 * rhs.a[0] + l2 * rhs.a[3] + l3 * rhs.a[6];
                ret[ret_offset + 1] = l1 * rhs.a[1] + l2 * rhs.a[4] + l3 * rhs.a[7];
                ret[ret_offset + 2] = l1 * rhs.a[2] + l2 * rhs.a[5] + l3 * rhs.a[8];
            }
            Matrix3x3 { a: ret }
        };

        // Separate and calculate the three blocks (rows 0-2, rows 3-5, rows 6-8)
        (multiply_block(0), multiply_block(3), multiply_block(6))
    }

    #[inline]
    pub fn multiply_9x3_by_3x3(&self, rhs: Matrix3x3<T>) -> (Matrix3x3<T>, Matrix3x3<T>, Matrix3x3<T>) {
        let lhs = Self::extract_9x3_array(self);
        Self::multiply_9x3_array_by_3x3(lhs, rhs)
    }
}

// **** Div ****

impl<T> Div<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Divide a matrix by a constant.
    #[inline]
    fn div(self, other: T) -> Self {
        T::m9x9_div_scalar(self, other)
    }
}

// **** DivAssign ****

impl<T> DivAssign<T> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// In-place divide a matrix by a constant.
    #[inline]
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

// **** AsRef ****

impl<T> AsRef<[T; 81]> for Matrix9x9<T> {
    /// Immutable reference to the raw array.
    #[inline]
    fn as_ref(&self) -> &[T; 81] {
        &self.a
    }
}

impl<T> AsMut<[T; 81]> for Matrix9x9<T> {
    /// Mutable reference to the raw array.
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 81] {
        &mut self.a
    }
}

// **** Deref ****

impl<T> Deref for Matrix9x9<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.a
    }
}

impl<T> DerefMut for Matrix9x9<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.a
    }
}

// **** Index ****

impl<T> Index<usize> for Matrix9x9<T> {
    type Output = T;

    /// Access matrix element by index.
    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.a[index]
    }
}

impl<T> Index<Range<usize>> for Matrix9x9<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<RangeFull> for Matrix9x9<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _index: RangeFull) -> &[T] {
        &self.a
    }
}

impl<T> Index<RangeInclusive<usize>> for Matrix9x9<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeInclusive<usize>) -> &[T] {
        &self.a[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix9x9<T> {
    type Output = T;

    /// Access matrix element by ordered pair (row, column).
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < 9 && col < 9, "Matrix index out of bounds: row={row}, col={col}");
        &self.a[row * 9 + col]
    }
}

// **** IndexMut ****

impl<T> IndexMut<usize> for Matrix9x9<T> {
    #[inline]
    /// Set matrix element by index.
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.a[index]
    }
}

impl<T> IndexMut<Range<usize>> for Matrix9x9<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<RangeFull> for Matrix9x9<T> {
    #[inline]
    fn index_mut(&mut self, _index: RangeFull) -> &mut [T] {
        &mut self.a
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Matrix9x9<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeInclusive<usize>) -> &mut [T] {
        &mut self.a[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix9x9<T> {
    /// Set matrix element by ordered pair (row, column).
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < 9 && col < 9, "Matrix index out of bounds: row={row}, col={col}");
        &mut self.a[row * 9 + col]
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Returns a row as a Vector3d 3-tuple.
    #[inline]
    pub fn row_tuple3d(&self, row_index: usize) -> (Vector3d<T>, Vector3d<T>, Vector3d<T>) {
        let offset = row_index * 9;
        (
            Vector3d { x: self.a[offset], y: self.a[offset + 1], z: self.a[offset + 2] },
            Vector3d { x: self.a[offset + 3], y: self.a[offset + 4], z: self.a[offset + 5] },
            Vector3d { x: self.a[offset + 6], y: self.a[offset + 7], z: self.a[offset + 8] },
        )
    }

    /// Returns a column as a Vector3d 3-tuple.
    #[inline]
    pub fn column_tuple3d(&self, col_index: usize) -> (Vector3d<T>, Vector3d<T>, Vector3d<T>) {
        let c = col_index;
        (
            Vector3d { x: self.a[c], y: self.a[c + 9], z: self.a[c + 18] },
            Vector3d { x: self.a[c + 27], y: self.a[c + 36], z: self.a[c + 45] },
            Vector3d { x: self.a[c + 54], y: self.a[c + 63], z: self.a[c + 72] },
        )
    }
}

// **** abs ****

impl<T> Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Return a copy of the matrix with all elements set to their absolute values.
    #[inline]
    pub fn abs(self) -> Self {
        T::m9x9_abs(self)
    }

    /// Set all elements of the matrix to their absolute values.
    #[inline]
    pub fn abs_in_place(&mut self) -> &mut Self {
        *self = T::m9x9_abs(*self);
        self
    }
}

// **** clamp ****

impl<T> Matrix9x9<T>
where
    T: Copy + FloatCore,
{
    /// Return a copy of the matrix with all elements clamped to the specified range.
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        let mut a = self.a;
        for it in &mut a {
            *it = it.clamp(min, max);
        }
        Self { a }
    }

    /// Clamp all elements of the matrix to the specified range.
    #[inline]
    pub fn clamp_in_place(&mut self, min: T, max: T) -> &mut Self {
        *self = self.clamp(min, max);
        self
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Return the transpose of this matrix.
    #[inline]
    pub fn transpose(&mut self) -> Self {
        // In-place transpose of the 8x8 submatrix
        // LLVM easily vectorizes this because the bounds and strides are power-of-two friendly
        for ii in 0..8 {
            for jj in (ii + 1)..8 {
                let idx_a = ii * 9 + jj;
                let idx_b = jj * 9 + ii;
                self.a.swap(idx_a, idx_b);
            }
        }

        // In-place swap of the 9th row and 9th column tail elements
        // (Excluding the very last corner element matrix[80] which stays put)
        for ii in 0..8 {
            let row_tail = ii * 9 + 8; // Element in the 9th column
            let col_tail = 8 * 9 + ii; // Element in the 9th row
            self.a.swap(row_tail, col_tail);
        }
        *self
    }

    /// Transpose matrix, in-place.
    #[inline]
    pub fn transpose_in_place(&mut self) -> &mut Self {
        *self = self.transpose();
        self
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    /// Return trace of matrix.
    #[inline]
    pub fn trace(self) -> T {
        T::m9x9_trace(self)
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + Zero + One + Matrix9x9Math + MathConstants + PartialOrd + Signed,
{
    /// Return the sum of all elements of the matrix.
    #[inline]
    pub fn sum(self) -> T {
        T::m9x9_sum(self)
    }

    /// Return the mean of all elements of the matrix.
    #[inline]
    pub fn mean(self) -> T {
        T::m9x9_mean(self)
    }

    /// Return the product of all elements of the matrix.
    #[inline]
    pub fn product(self) -> T {
        T::m9x9_product(self)
    }

    /// Return true if matrix is near zero.
    pub fn is_near_zero(self) -> bool {
        for a in &self.a {
            if a.abs() > T::EPSILON {
                return false;
            }
        }
        true
    }
}

// **** Iterators ****

impl<T> Matrix9x9<T> {
    /// Returns an iterator over the rows of the matrix as slices of 9 elements.
    #[inline]
    pub fn rows(&self) -> ChunksExact<'_, T> {
        self.chunks_exact(9)
    }
}

impl<T> Matrix9x9<T> {
    /// Returns an iterator over the rows of the matrix as mutable slices of 9 elements.
    #[inline]
    pub fn rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.chunks_exact_mut(9)
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Consumes the matrix and returns an array of its 9 rows.
    #[inline]
    pub fn into_rows(self) -> [[T; 9]; 9] {
        // Build the nested 2D array matrix safely in a single unrolled pass
        core::array::from_fn(|r| core::array::from_fn(|c| self.a[r * 9 + c]))
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Returns an iterator over the columns of the matrix as owned 4-element arrays.
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = [T; 9]> {
        // Create an iterator over the column indices (0, 1, 2, ..)
        (0..9).map(|c| {
            // Collect the strided elements for the current column
            [
                self.a[c],
                self.a[c + 4],
                self.a[c + 8],
                self.a[c + 12],
                self.a[c + 16],
                self.a[c + 20],
                self.a[c + 24],
                self.a[c + 28],
                self.a[c + 32],
            ]
        })
    }
}

impl<'a, T> IntoIterator for &'a Matrix9x9<T> {
    type Item = &'a [T];
    type IntoIter = ChunksExact<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the Deref trait automatically to get 9-element rows
        self.chunks_exact(9)
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix9x9<T> {
    type Item = &'a mut [T];
    type IntoIter = ChunksExactMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Leverages the DerefMut trait automatically.
        self.chunks_exact_mut(9)
    }
}

impl<T> IntoIterator for Matrix9x9<T>
where
    T: Copy,
{
    type Item = [T; 9];
    type IntoIter = core::array::IntoIter<[T; 9], 9>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Construct the 9 rows of 9 elements safely in one direct pass
        let rows = core::array::from_fn(|r| core::array::from_fn(|c| self.a[r * 9 + c]));
        rows.into_iter()
    }
}

// **** From ****

// **** From Array ****

impl<T> From<[T; 81]> for Matrix9x9<T>
where
    T: Copy,
{
    /// Matrix from 1D array.
    #[inline]
    fn from(input: [T; 81]) -> Self {
        Self { a: input }
    }
}

impl<T> From<Matrix9x9<T>> for Matrix2x2<T>
where
    T: Copy,
{
    /// Matrix2x2 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9x9<T>) -> Self {
        Self { a: [
            m.a[0],  m.a[1],
            m.a[9],  m.a[10],
        ] }
    }
}

impl<T> From<Matrix9x9<T>> for Matrix3x3<T>
where
    T: Copy,
{
    /// Matrix3x3 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9x9<T>) -> Self {
        Self { a: [
            m.a[0],  m.a[1],  m.a[2],
            m.a[9],  m.a[10], m.a[11],
            m.a[18], m.a[19], m.a[20]
        ] }
    }
}

impl<T> From<Matrix9x9<T>> for Matrix4x4<T>
where
    T: Copy,
{
    /// Matrix4x4 from Matrix9x9. Takes top left of m9x9, discarding other values.
    #[rustfmt::skip]
    #[inline]
    fn from(m: Matrix9x9<T>) -> Self {
        Self { a: [
            m.a[0],  m.a[1],  m.a[2],  m.a[3],
            m.a[9],  m.a[10], m.a[11], m.a[12],
            m.a[18], m.a[19], m.a[20], m.a[21],
            m.a[27], m.a[28], m.a[29], m.a[30],
        ] }
    }
}
