#![allow(clippy::inline_always)]
#![allow(unused)]

use crate::{Matrix3x3, Matrix3x3Math, Matrix4x4, Vector4d};

// **** Math ****

/// Math functions for Matrix4x4, using **SIMD** accelerations for `f32`.<br><br>
pub trait Matrix4x4Math: Sized {
    fn m4x4_neg(this: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_abs(this: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_add(this: Matrix4x4<Self>, this: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_mul_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self>;
    fn m4x4_div_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self>;
    fn m4x4_mul_add(this: Matrix4x4<Self>, k: Self, other: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_mul_vector(this: Matrix4x4<Self>, other: Vector4d<Self>) -> Vector4d<Self>;
    fn m4x4_vector_mul(this: Vector4d<Self>, other: Matrix4x4<Self>) -> Vector4d<Self>;
    fn m4x4_mul(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self>;
    fn m4x4_determinant(this: Matrix4x4<Self>) -> Self;
    fn m4x4_top_right_determinant(this: Matrix4x4<Self>) -> Self;
    fn m4x4_top_right_sum_squares(this: Matrix4x4<Self>) -> Self;
    fn m4x4_trace(this: Matrix4x4<Self>) -> Self;
    fn m4x4_trace_sum_squares(this: Matrix4x4<Self>) -> Self;
    fn m4x4_sum(this: Matrix4x4<Self>) -> Self;
    fn m4x4_mean(this: Matrix4x4<Self>) -> Self;
    fn m4x4_product(this: Matrix4x4<Self>) -> Self;
    fn m4x4_adjugate(this: Matrix4x4<Self>) -> (Matrix4x4<Self>, Self);
}

impl Matrix4x4Math for f32 {
    #[inline(always)]
    fn m4x4_neg(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| -this.a[ii]);
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_abs(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_add(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_mul_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_div_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        Self::m4x4_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m4x4_mul_add(this: Matrix4x4<Self>, k: Self, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        Self::m4x4_add(Self::m4x4_mul_scalar(this, k), other)
    }

    #[rustfmt::skip]
    #[inline]
    fn m4x4_vector_mul(this: Vector4d<Self>, other: Matrix4x4<Self>) -> Vector4d<Self> {
        Vector4d {
            x: this.x * other.a[0] + this.y * other.a[4] + this.z * other.a[8] + this.t * other.a[12],
            y: this.x * other.a[1] + this.y * other.a[5] + this.z * other.a[9] + this.t * other.a[13],
            z: this.x * other.a[2] + this.y * other.a[6] + this.z * other.a[10] + this.t * other.a[14],
            t: this.x * other.a[3] + this.y * other.a[7] + this.z * other.a[11] + this.t * other.a[15],
        }
    }

    #[allow(clippy::needless_range_loop)]
    #[rustfmt::skip]
    #[inline]
    fn m4x4_mul_vector(this: Matrix4x4<Self>, other: Vector4d<Self>) -> Vector4d<Self> {
        let mut res = [0.0; 4];
        let v = [other.x, other.y, other.z, other.t];

        for i in 0..4 {
            let col_scalar = v[i];
            for j in 0..4 {
                // Accessing the matrix columns
                res[j] += this.a[i + j * 4] * col_scalar;
            }
        }
        
        Vector4d { x: res[0], y: res[1], z: res[2], t: res[3] }
    }

    #[inline]
    fn m4x4_mul(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let mut ret = [0.0; 16];
        
        // Explicitly tell the compiler we are working with fixed 4-element chunks
        for i in 0..4 {
            let row = &this.a[i*4..(i*4)+4];
            for j in 0..4 {
                // By using a local sum and fixed indices, 
                // LLVM can more easily unroll this into SIMD 'Multiply-Add' instructions.
                ret[i*4 + j] = 
                    row[0] * other.a[j]      +
                    row[1] * other.a[j + 4]  +
                    row[2] * other.a[j + 8]  +
                    row[3] * other.a[j + 12];
            }
        }
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_trace(this: Matrix4x4<Self>) -> Self {
        this.a[0] + this.a[5] + this.a[10] + this.a[15]
    }

    #[inline(always)]
    fn m4x4_trace_sum_squares(this: Matrix4x4<Self>) -> Self {
        { this.a[0] * this.a[0] + this.a[5] * this.a[5] + this.a[10] * this.a[10] + this.a[15] * this.a[15] }
    }

    #[inline(always)]
    fn m4x4_sum(this: Matrix4x4<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m4x4_mean(this: Matrix4x4<Self>) -> Self {
        Self::m4x4_sum(this) / 16.0
    }

    #[inline(always)]
    fn m4x4_product(this: Matrix4x4<Self>) -> Self {
        this.a.iter().product()
    }

    #[inline]
    fn m4x4_top_right_sum_squares(this: Matrix4x4<Self>) -> Self {
        this.a[1] * this.a[1]
            + this.a[2] * this.a[2]
            + this.a[3] * this.a[3]
            + this.a[6] * this.a[6]
            + this.a[7] * this.a[7]
            + this.a[11] * this.a[11]
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_top_right_determinant(this: Matrix4x4<Self>) -> Self {
        0.0
    }

    #[rustfmt::skip]
    #[inline]
    fn m4x4_determinant(this: Matrix4x4<Self>) -> Self {
         this.a[0] * Self::m3x3_determinant(Matrix3x3::from([this.a[5], this.a[6], this.a[7],   this.a[9], this.a[10], this.a[11],   this.a[13], this.a[14], this.a[15]]))
        -this.a[1] * Self::m3x3_determinant(Matrix3x3::from([this.a[4], this.a[6], this.a[7],   this.a[8], this.a[10], this.a[11],   this.a[12], this.a[14], this.a[15]]))
        +this.a[2] * Self::m3x3_determinant(Matrix3x3::from([this.a[4], this.a[5], this.a[7],   this.a[8],  this.a[9], this.a[11],   this.a[12], this.a[13], this.a[15]]))
        -this.a[3] * Self::m3x3_determinant(Matrix3x3::from([this.a[4], this.a[5], this.a[6],   this.a[8],  this.a[9], this.a[10],   this.a[12], this.a[13], this.a[14]]))
    }

    #[rustfmt::skip]
    fn m4x4_adjugate(s: Matrix4x4<Self>) -> (Matrix4x4<Self>, Self) {
        let s0  = s.a[0];  let s1  = s.a[1];  let s2  = s.a[2];  let s3  = s.a[3];
        let s4  = s.a[4];  let s5  = s.a[5];  let s6  = s.a[6];  let s7  = s.a[7];
        let s8  = s.a[8];  let s9  = s.a[9];  let s10 = s.a[10]; let s11 = s.a[11];
        let s12 = s.a[12]; let s13 = s.a[13]; let s14 = s.a[14]; let s15 = s.a[15];

        // Pre-calculate 2x2 determinants for the bottom two rows
        let b0 = s8 * s13 - s9 * s12;
        let b1 = s8 * s14 - s10 * s12;
        let b2 = s8 * s15 - s11 * s12;
        let b3 = s9 * s14 - s10 * s13;
        let b4 = s9 * s15 - s11 * s13;
        let b5 = s10 * s15 - s11 * s14;

        // Pre-calculate 2x2 determinants for the top two rows
        let t0 = s0 * s5 - s1 * s4;
        let t1 = s0 * s6 - s2 * s4;
        let t2 = s0 * s7 - s3 * s4;
        let t3 = s1 * s6 - s2 * s5;
        let t4 = s1 * s7 - s3 * s5;
        let t5 = s2 * s7 - s3 * s6;

        // Calculate cofactors (already transposed)
        let c00 =  s5 * b5 - s6 * b4 + s7 * b3;
        let c01 = -s1 * b5 + s2 * b4 - s3 * b3;
        let c02 =  s13 * t5 - s14 * t4 + s15 * t3;
        let c03 = -s9 * t5 + s10 * t4 - s11 * t3;

        let c10 = -s4 * b5 + s6 * b2 - s7 * b1;
        let c11 =  s0 * b5 - s2 * b2 + s3 * b1;
        let c12 = -s12 * t5 + s14 * t2 - s15 * t1;
        let c13 =  s8 * t5 - s10 * t2 + s11 * t1;

        let c20 =  s4 * b4 - s5 * b2 + s7 * b0;
        let c21 = -s0 * b4 + s1 * b2 - s3 * b0;
        let c22 =  s12 * t4 - s13 * t2 + s15 * t0;
        let c23 = -s8 * t4 + s9 * t2 - s11 * t0;

        let c30 = -s4 * b3 + s5 * b1 - s6 * b0;
        let c31 =  s0 * b3 - s1 * b1 + s2 * b0;
        let c32 = -s12 * t3 + s13 * t1 - s14 * t0;
        let c33 =  s8 * t3 - s9 * t1 + s10 * t0;

        let det = s0 * c00 + s1 * c10 + s2 * c20 + s3 * c30;

        (Matrix4x4::from([
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33
        ]), det)
    }
}

impl Matrix4x4Math for f64 {
    #[inline(always)]
    fn m4x4_neg(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| -this.a[ii]);
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_abs(this: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii].abs());
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_add(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] + other.a[ii]);
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_mul_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        let ret = core::array::from_fn(|ii| this.a[ii] * other);
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_div_scalar(this: Matrix4x4<Self>, other: Self) -> Matrix4x4<Self> {
        Self::m4x4_mul_scalar(this, 1.0 / other)
    }

    #[inline(always)]
    fn m4x4_mul_add(this: Matrix4x4<Self>, k: Self, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        Self::m4x4_add(Self::m4x4_mul_scalar(this, k), other)
    }

    #[rustfmt::skip]
    #[inline]
    fn m4x4_vector_mul(this: Vector4d<Self>, other: Matrix4x4<Self>) -> Vector4d<Self> {
        Vector4d {
            x: this.x * other.a[0] + this.y * other.a[4] + this.z * other.a[8] + this.t * other.a[12],
            y: this.x * other.a[1] + this.y * other.a[5] + this.z * other.a[9] + this.t * other.a[13],
            z: this.x * other.a[2] + this.y * other.a[6] + this.z * other.a[10] + this.t * other.a[14],
            t: this.x * other.a[3] + this.y * other.a[7] + this.z * other.a[11] + this.t * other.a[15],
        }
    }

    #[rustfmt::skip]
    #[inline]
    fn m4x4_mul_vector(this: Matrix4x4<Self>, other: Vector4d<Self>) -> Vector4d<Self> {
        Vector4d {
            x:  this.a[0] * other.x +  this.a[1] * other.y +  this.a[2] * other.z +  this.a[3] * other.t,
            y:  this.a[4] * other.x +  this.a[5] * other.y +  this.a[6] * other.z +  this.a[7] * other.t,
            z:  this.a[8] * other.x +  this.a[9] * other.y + this.a[10] * other.z + this.a[11] * other.t,
            t: this.a[12] * other.x + this.a[13] * other.y + this.a[14] * other.z + this.a[15] * other.t,
        }
    }

    #[inline]
    fn m4x4_mul(this: Matrix4x4<Self>, other: Matrix4x4<Self>) -> Matrix4x4<Self> {
        let mut ret = [0.0; 16];
        
        // Explicitly tell the compiler we are working with fixed 4-element chunks
        for i in 0..4 {
            let row = &this.a[i*4..(i*4)+4];
            for j in 0..4 {
                // By using a local sum and fixed indices, 
                // LLVM can more easily unroll this into SIMD 'Multiply-Add' instructions.
                ret[i*4 + j] = 
                    row[0] * other.a[j]      +
                    row[1] * other.a[j + 4]  +
                    row[2] * other.a[j + 8]  +
                    row[3] * other.a[j + 12];
            }
        }
        Matrix4x4::from(ret)
    }

    #[inline(always)]
    fn m4x4_trace(this: Matrix4x4<Self>) -> Self {
        this.a[0] + this.a[5] + this.a[10] + this.a[15]
    }

    #[inline(always)]
    fn m4x4_trace_sum_squares(this: Matrix4x4<Self>) -> Self {
        { this.a[0] * this.a[0] + this.a[5] * this.a[5] + this.a[10] * this.a[10] + this.a[15] * this.a[15] }
    }

    #[inline(always)]
    fn m4x4_sum(this: Matrix4x4<Self>) -> Self {
        this.a.iter().sum()
    }

    #[inline(always)]
    fn m4x4_mean(this: Matrix4x4<Self>) -> Self {
        Self::m4x4_sum(this) / 16.0
    }

    #[inline(always)]
    fn m4x4_product(this: Matrix4x4<Self>) -> Self {
        this.a.iter().product()
    }

    #[inline(always)]
    fn m4x4_top_right_sum_squares(this: Matrix4x4<Self>) -> Self {
        this.a[1] * this.a[1]
            + this.a[2] * this.a[2]
            + this.a[3] * this.a[3]
            + this.a[6] * this.a[6]
            + this.a[7] * this.a[7]
            + this.a[11] * this.a[11]
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn m4x4_top_right_determinant(this: Matrix4x4<Self>) -> Self {
        0.0
    }

    #[rustfmt::skip]
    #[inline]
    fn m4x4_determinant(this: Matrix4x4<Self>) -> Self {
         this.a[0] * Self::m3x3_determinant(Matrix3x3::from([this.a[5], this.a[6], this.a[7],   this.a[9], this.a[10], this.a[11],   this.a[13], this.a[14], this.a[15]]))
        -this.a[1] * Self::m3x3_determinant(Matrix3x3::from([this.a[4], this.a[6], this.a[7],   this.a[8], this.a[10], this.a[11],   this.a[12], this.a[14], this.a[15]]))
        +this.a[2] * Self::m3x3_determinant(Matrix3x3::from([this.a[4], this.a[5], this.a[7],   this.a[8],  this.a[9], this.a[11],   this.a[12], this.a[13], this.a[15]]))
        -this.a[3] * Self::m3x3_determinant(Matrix3x3::from([this.a[4], this.a[5], this.a[6],   this.a[8],  this.a[9], this.a[10],   this.a[12], this.a[13], this.a[14]]))
    }

    #[rustfmt::skip]
    fn m4x4_adjugate(s: Matrix4x4<Self>) -> (Matrix4x4<Self>, Self) {
        let s0  = s.a[0];  let s1  = s.a[1];  let s2  = s.a[2];  let s3  = s.a[3];
        let s4  = s.a[4];  let s5  = s.a[5];  let s6  = s.a[6];  let s7  = s.a[7];
        let s8  = s.a[8];  let s9  = s.a[9];  let s10 = s.a[10]; let s11 = s.a[11];
        let s12 = s.a[12]; let s13 = s.a[13]; let s14 = s.a[14]; let s15 = s.a[15];

        // Pre-calculate 2x2 determinants for the bottom two rows
        let b0 = s8 * s13 - s9 * s12;
        let b1 = s8 * s14 - s10 * s12;
        let b2 = s8 * s15 - s11 * s12;
        let b3 = s9 * s14 - s10 * s13;
        let b4 = s9 * s15 - s11 * s13;
        let b5 = s10 * s15 - s11 * s14;

        // Pre-calculate 2x2 determinants for the top two rows
        let t0 = s0 * s5 - s1 * s4;
        let t1 = s0 * s6 - s2 * s4;
        let t2 = s0 * s7 - s3 * s4;
        let t3 = s1 * s6 - s2 * s5;
        let t4 = s1 * s7 - s3 * s5;
        let t5 = s2 * s7 - s3 * s6;

        // Calculate cofactors (already transposed)
        let c00 =  s5 * b5 - s6 * b4 + s7 * b3;
        let c01 = -s1 * b5 + s2 * b4 - s3 * b3;
        let c02 =  s13 * t5 - s14 * t4 + s15 * t3;
        let c03 = -s9 * t5 + s10 * t4 - s11 * t3;

        let c10 = -s4 * b5 + s6 * b2 - s7 * b1;
        let c11 =  s0 * b5 - s2 * b2 + s3 * b1;
        let c12 = -s12 * t5 + s14 * t2 - s15 * t1;
        let c13 =  s8 * t5 - s10 * t2 + s11 * t1;

        let c20 =  s4 * b4 - s5 * b2 + s7 * b0;
        let c21 = -s0 * b4 + s1 * b2 - s3 * b0;
        let c22 =  s12 * t4 - s13 * t2 + s15 * t0;
        let c23 = -s8 * t4 + s9 * t2 - s11 * t0;

        let c30 = -s4 * b3 + s5 * b1 - s6 * b0;
        let c31 =  s0 * b3 - s1 * b1 + s2 * b0;
        let c32 = -s12 * t3 + s13 * t1 - s14 * t0;
        let c33 =  s8 * t3 - s9 * t1 + s10 * t0;

        let det = s0 * c00 + s1 * c10 + s2 * c20 + s3 * c30;

        (Matrix4x4::from([
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33
        ]), det)
    }
}
