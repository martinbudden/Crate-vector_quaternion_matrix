use core::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFull,
    RangeInclusive, Sub, SubAssign,
};
use num_traits::{ConstOne, ConstZero, MulAdd, MulAddAssign, One, Signed, Zero, float::FloatCore};

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
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Matrix9x9<T> {
    // Flattened 9x9 matrix: 16 elements in row-major order
    pub(crate) a: [T; 81],
}

impl<T> Default for Matrix9x9<T>
where
    T: Copy + Zero,
{
    fn default() -> Self {
        Self { a: [T::zero(); 81] }
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

impl<T> One for Matrix9x9<T>
where
    T: Copy + Zero + One + PartialEq + Matrix9x9Math,
{
    /// Identity matrix.
    #[inline]
    fn one() -> Self {
        let mut z = [T::zero(); 81];
        for ii in 0..=8 {
            z[ii * 9 + ii] = T::one();
        }
        Self { a: z }
    }
    #[inline]
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

impl<T> ConstOne for Matrix9x9<T>
where
    T: Copy + ConstZero + ConstOne + PartialEq + Matrix9x9Math,
{
    /// Const identity matrix.
    #[rustfmt::skip]
    const ONE: Self = Self {
        a: [
            T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,  T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ZERO, T::ONE,
        ]
    };
}

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
        // Reuse existing SIMD-optimized Add and Neg implementations
        self + (-other)
    }
}

// **** SubAssign ****

impl<T> SubAssign for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
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
    // Extract the first 3 columns of the 9x9 matrix as and array of 27 elements.
    pub fn extract_9x3_array(&mut self) -> [T; 27] {
        let mut a27 = [T::zero(); 27];
        for r in 0..9 {
            let offset9 = r * 9;
            let offset3 = r * 3;
            a27[offset3] = self.a[offset9]; // Column 1
            a27[offset3 + 1] = self.a[offset9 + 1]; // Column 2
            a27[offset3 + 2] = self.a[offset9 + 2]; // Column 3
        }
        a27
    }

    /// Multiplies the first 3 columns of P (9x3 block) by the inverted 3x3 S matrix.
    /// Returns a tuple of three 3x3 sub-matrices mapping directly to the physical state domains.
    #[inline]
    pub fn multiply_9x3_by_3x3(
        p_cols_1to3: [f32; 27],
        s_inv: Matrix3x3<f32>,
    ) -> (Matrix3x3<f32>, Matrix3x3<f32>, Matrix3x3<f32>) {
        // Helper closure to compute a single 3x3 block from a specific row slice of p_cols_1to3
        let multiply_block = |start_row: usize| -> Matrix3x3<f32> {
            let mut out_data = [0.0; 9];
            for r in 0..3 {
                let p_offset = (start_row + r) * 3;
                let out_offset = r * 3;

                let p1 = p_cols_1to3[p_offset];
                let p2 = p_cols_1to3[p_offset + 1];
                let p3 = p_cols_1to3[p_offset + 2];

                // Compute row dot products with the 3 columns of s_inv
                out_data[out_offset] = p1 * s_inv.a[0] + p2 * s_inv.a[3] + p3 * s_inv.a[6];
                out_data[out_offset + 1] = p1 * s_inv.a[1] + p2 * s_inv.a[4] + p3 * s_inv.a[7];
                out_data[out_offset + 2] = p1 * s_inv.a[2] + p2 * s_inv.a[5] + p3 * s_inv.a[8];
            }
            Matrix3x3 { a: out_data }
        };

        // Separate and calculate the three physical state blocks (Pos rows 0-2, Vel rows 3-5, Bias rows 6-8)
        (
            multiply_block(0), // K_block_pos
            multiply_block(3), // K_block_vel
            multiply_block(6), // K_block_acc_bias
        )
    }
}

impl<T> Mul<Matrix9x9<T>> for Matrix9x9<T>
where
    T: Copy + Matrix9x9Math,
{
    type Output = Self;

    /// Multiply two matrices.
    /// Not implemented just returns self, but required for the One trait.
    #[inline]
    fn mul(self, _other: Self) -> Self {
        debug_assert!(false);
        self
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

// **** AsRef ****

// Immutable reference to the raw array: let array_ref: &[T; 81] = matrix.as_ref();
impl<T> AsRef<[T; 81]> for Matrix9x9<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 81] {
        &self.a
    }
}

// Mutable reference to the raw array: let array_mut: &mut [T; 81] = matrix.as_mut();
impl<T> AsMut<[T; 81]> for Matrix9x9<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; 81] {
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
    #[inline]
    /// Set matrix element by ordered pair (row, column).
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
