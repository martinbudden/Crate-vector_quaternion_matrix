#![allow(clippy::inline_always)]
#![allow(unused)]

use crate::Matrix9x9;

// **** Math ****

/// Math functions for Matrix9x9.<br><br>
pub trait Matrix9x9Math: Sized {
    fn m9x9_neg(this: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_abs(this: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_add(this: Matrix9x9<Self>, this: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_mul_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self>;
    fn m9x9_div_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self>;
    fn m9x9_mul_add(this: Matrix9x9<Self>, k: Self, other: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_mul(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self>;
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self;
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self;
    fn m9x9_sum(this: Matrix9x9<Self>) -> Self;
    fn m9x9_mean(this: Matrix9x9<Self>) -> Self;
    fn m9x9_product(this: Matrix9x9<Self>) -> Self;
}

impl Matrix9x9Math for f32 {
    #[inline(always)]
    fn m9x9_neg(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| -this.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_abs(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_add(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_mul_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_div_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        Self::m9x9_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m9x9_mul_add(this: Matrix9x9<Self>, k: Self, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        Self::m9x9_add(Self::m9x9_mul_scalar(this, k), other)
    }

    #[inline]
    fn m9x9_mul(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let mut ret = [0.0; 81];

        // Loop through the output matrix 9 elements at a time (row by row)
        // chunks_exact_mut(9) proves to the compiler that each block is exactly 9 elements,
        // which completely eliminates internal array bounds checks.
        for (r, row_slice) in ret.chunks_exact_mut(9).enumerate() {
            let row_offset = r * 9;

            // Cache the current row of the left matrix in local memory/registers.
            // This is a massive win for the CPU cache line.
            let this_row = &this.a[row_offset..row_offset + 9];

            // Calculate the dot product for each of the 9 columns in the right matrix
            for (c, out_element) in row_slice.iter_mut().enumerate() {
                // Initialize the accumulator with the first pair
                let mut sum = this_row[0] * other.a[c];

                // Unroll the remaining 8 elements of the dot product.
                // Using fixed index offsets (c + 9, c + 18, etc.) allows the compiler
                // to optimize this into pipelined, branchless fused multiply-add instructions.
                sum += this_row[1] * other.a[c + 9];
                sum += this_row[2] * other.a[c + 18];
                sum += this_row[3] * other.a[c + 27];
                sum += this_row[4] * other.a[c + 36];
                sum += this_row[5] * other.a[c + 45];
                sum += this_row[6] * other.a[c + 54];
                sum += this_row[7] * other.a[c + 63];
                sum += this_row[8] * other.a[c + 72];

                // Store the final computed dot product in the output row slice
                *out_element = sum;
            }
        }
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val * val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_sum(this: Matrix9x9<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m9x9_mean(this: Matrix9x9<Self>) -> Self {
        Self::m9x9_sum(this) / 16.0
    }

    #[inline(always)]
    fn m9x9_product(this: Matrix9x9<Self>) -> Self {
        this.a.iter().product()
    }
}

// **** f64 ****

impl Matrix9x9Math for f64 {
    #[inline(always)]
    fn m9x9_neg(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| -this.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_abs(this: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_add(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_mul_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_div_scalar(this: Matrix9x9<Self>, other: Self) -> Matrix9x9<Self> {
        Self::m9x9_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m9x9_mul_add(this: Matrix9x9<Self>, k: Self, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        Self::m9x9_add(Self::m9x9_mul_scalar(this, k), other)
    }

    #[inline]
    fn m9x9_mul(this: Matrix9x9<Self>, other: Matrix9x9<Self>) -> Matrix9x9<Self> {
        let mut ret = [0.0; 81];

        // Loop through the output matrix 9 elements at a time (row by row)
        // chunks_exact_mut(9) proves to the compiler that each block is exactly 9 elements,
        // which completely eliminates internal array bounds checks.
        for (r, row_slice) in ret.chunks_exact_mut(9).enumerate() {
            let row_offset = r * 9;

            // Cache the current row of the left matrix in local memory/registers.
            // This is a massive win for the CPU cache line.
            let this_row = &this.a[row_offset..row_offset + 9];

            // Calculate the dot product for each of the 9 columns in the right matrix
            for (c, out_element) in row_slice.iter_mut().enumerate() {
                // Initialize the accumulator with the first pair
                let mut sum = this_row[0] * other.a[c];

                // Unroll the remaining 8 elements of the dot product.
                // Using fixed index offsets (c + 9, c + 18, etc.) allows the compiler
                // to optimize this into pipelined, branchless fused multiply-add instructions.
                sum += this_row[1] * other.a[c + 9];
                sum += this_row[2] * other.a[c + 18];
                sum += this_row[3] * other.a[c + 27];
                sum += this_row[4] * other.a[c + 36];
                sum += this_row[5] * other.a[c + 45];
                sum += this_row[6] * other.a[c + 54];
                sum += this_row[7] * other.a[c + 63];
                sum += this_row[8] * other.a[c + 72];

                // Store the final computed dot product in the output row slice
                *out_element = sum;
            }
        }
        Matrix9x9::from(ret)
    }

    #[inline(always)]
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self {
        let mut sum = 0.0;
        for &val in this.a.iter().step_by(10) {
            sum += val * val;
        }
        sum
    }

    #[inline(always)]
    fn m9x9_sum(this: Matrix9x9<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m9x9_mean(this: Matrix9x9<Self>) -> Self {
        Self::m9x9_sum(this) / 16.0
    }

    #[inline(always)]
    fn m9x9_product(this: Matrix9x9<Self>) -> Self {
        this.a.iter().product()
    }
}
