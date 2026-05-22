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

    #[inline(always)]
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self {
        this.a[0] + this.a[5] + this.a[10] + this.a[15]
    }

    #[inline(always)]
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self {
        this.a[0] * this.a[0] + this.a[5] * this.a[5] + this.a[10] * this.a[10] + this.a[15] * this.a[15]
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

    #[inline(always)]
    fn m9x9_trace(this: Matrix9x9<Self>) -> Self {
        this.a[0] + this.a[5] + this.a[10] + this.a[15]
    }

    #[inline(always)]
    fn m9x9_trace_sum_squares(this: Matrix9x9<Self>) -> Self {
        { this.a[0] * this.a[0] + this.a[5] * this.a[5] + this.a[10] * this.a[10] + this.a[15] * this.a[15] }
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
