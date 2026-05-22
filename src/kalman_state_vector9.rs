use core::ops::Mul;

use num_traits::Zero;

use crate::{Matrix9x9, Matrix9x9Math, Vector3d};

/// Kalman state vector of `f32` values<br>
pub type KalmanStateVector9f32 = KalmanStateVector9<f32>;
/// Kalman tate vector of `f64` values<br><br>
pub type KalmanStateVector9f64 = KalmanStateVector9<f64>;

/// Flattened representation of a 9-element state vector for Kalman filter matrix math.<br><br>
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(64))]
pub struct KalmanStateVector9<T> {
    pub pos: Vector3d<T>,
    pub vel: Vector3d<T>,
    pub bias: Vector3d<T>,
}

impl<T> From<(Vector3d<T>, Vector3d<T>, Vector3d<T>)> for KalmanStateVector9<T> {
    #[inline]
    fn from(v: (Vector3d<T>, Vector3d<T>, Vector3d<T>)) -> Self {
        Self { pos: v.0, vel: v.1, bias: v.2 }
    }
}

/// Implement vector-by-scalar multiplication to scale the Kalman Gain.
impl Mul<f32> for KalmanStateVector9<f32> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self { pos: self.pos * rhs, vel: self.vel * rhs, bias: self.bias * rhs }
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy,
{
    /// Helper to map 0-based Row and Column indices to array positions.
    #[inline]
    pub const fn index0(row: usize, col: usize) -> usize {
        row * 9 + col
    }

    /// Helper to map 1-based Row and Column indices to array positions.
    #[inline]
    pub const fn index1(row: usize, col: usize) -> usize {
        (row - 1) * 9 + (col - 1)
    }
}

impl<T> Matrix9x9<T>
where
    T: Copy + Zero + Matrix9x9Math + Mul<T, Output = T>,
{
    /// Computes the outer product of two 9-element states in a compiler-friendly manner.
    #[inline]
    pub fn outer_product(col: KalmanStateVector9<T>, row: KalmanStateVector9<T>) -> Self {
        // Convert the 9 elements into three 4-lane logically padded chunks.
        // The 4th element is a trailing scratchpad to fulfill 128-bit SIMD constraints.
        let r1 = [row.pos.x, row.pos.y, row.pos.z, T::zero()];
        let r2 = [row.vel.x, row.vel.y, row.vel.z, T::zero()];
        let r3 = [row.bias.x, row.bias.y, row.bias.z, T::zero()];

        // We can flatten 'col' into an easily indexable array via registers
        let c = [col.pos.x, col.pos.y, col.pos.z, col.vel.x, col.vel.y, col.vel.z, col.bias.x, col.bias.y, col.bias.z];

        let mut out = Self { a: [T::zero(); 81] };

        // Process each row. LLVM sees fixed loops of 4 elements and easily emits
        // optimized parallel hardware vector instructions.
        #[allow(clippy::needless_range_loop)]
        for r_idx in 0..9 {
            let offset = r_idx * 9;
            let scalar = c[r_idx];

            let mut chunk1 = [T::zero(); 4];
            let mut chunk2 = [T::zero(); 4];
            let mut chunk3 = [T::zero(); 4];

            for ii in 0..4 {
                chunk1[ii] = scalar * r1[ii];
            }
            for ii in 0..4 {
                chunk2[ii] = scalar * r2[ii];
            }
            for ii in 0..4 {
                chunk3[ii] = scalar * r3[ii];
            }

            // Write back to the matrix, dropping the 4th padding lane of each chunk
            out.a[offset] = chunk1[0];
            out.a[offset + 1] = chunk1[1];
            out.a[offset + 2] = chunk1[2];
            out.a[offset + 3] = chunk2[0];
            out.a[offset + 4] = chunk2[1];
            out.a[offset + 5] = chunk2[2];
            out.a[offset + 6] = chunk3[0];
            out.a[offset + 7] = chunk3[1];
            out.a[offset + 8] = chunk3[2];
        }

        out
    }
}
