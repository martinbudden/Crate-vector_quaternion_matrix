#![allow(clippy::inline_always)]

use cfg_if::cfg_if;

/// `no_std` implementations of `sqrt` and `sqrt_reciprocal` in  method call syntax<br>
/// ie `x.sqrt()`, `x.sqrt_reciprocal()`.
#[allow(missing_docs)]
pub trait SqrtMethods: Sized {
    #[must_use]
    fn sqrt(self) -> Self;
    #[must_use]
    fn sqrt_reciprocal(self) -> Self;
}

cfg_if! {
    if #[cfg(feature = "std")] {
        // Use the hardware-linked math methods in Standard Library
        impl SqrtMethods for f32 {
            #[inline(always)]
            fn sqrt(self) -> f32 {
                self.sqrt()
            }
            #[inline(always)]
            fn sqrt_reciprocal(self) -> f32 {
                1.0 / self.sqrt()
            }
        }
        impl SqrtMethods for f64 {
            #[inline(always)]
            fn sqrt(self) -> f64 {
                self.sqrt()
            }
            #[inline(always)]
            fn sqrt_reciprocal(self) -> f64 {
                1.0 / self.sqrt()
            }
        }
    } else if #[cfg(all(not(feature = "std"), feature = "libm"))] {
        impl SqrtMethods for f32 {
            #[inline(always)]
            fn sqrt(self) -> f32 {
                // Use hardware ASM for ARM chips with an FPU
                #[cfg(all(target_arch = "arm", target_feature = "vfp2"))]
                { sqrt_reciprocal_f32(self) }
                // Fallback for non-ARM, non-FPU target
                #[cfg(not(all(target_arch = "arm", target_feature = "vfp2")))]
                { libm::sqrtf(self) }
            }
            #[inline(always)]
            fn sqrt_reciprocal(self) -> f32 {
                // Use hardware ASM for ARM chips with an FPU
                #[cfg(all(target_arch = "arm", target_feature = "vfp2"))]
                { sqrt_reciprocal_f32(self) }

                // Fallback for non-ARM, non-FPU target
                #[cfg(not(all(target_arch = "arm", target_feature = "vfp2")))]
                { 1.0 / libm::sqrtf(self) }
            }
        }
        impl SqrtMethods for f64 {
            #[inline(always)]
            fn sqrt(self) -> f64 {
                libm::sqrt(self)
            }
            #[inline(always)]
            fn sqrt_reciprocal(self) -> f64 {
                1.0 / libm::sqrt(self)
            }
        }
    } else if #[cfg(all(not(feature = "std"), not(feature = "libm")))] {
        impl SqrtMethods for f32 {
            fn sqrt(self) -> f32 {
                sqrt_f32(self)
            }
            fn sqrt_reciprocal(self) -> f32 {
                sqrt_reciprocal_f32(self)
            }
        }
        impl SqrtMethods for f64 {
            fn sqrt(self) -> f64 {
                sqrt_f64(self)
            }
            fn sqrt_reciprocal(self) -> f64 {
                sqrt_reciprocal_f64(self)
            }
        }
    }
}

#[allow(unused)]
#[inline(always)]
pub fn sqrt_f32(x: f32) -> f32 {
    // Use hardware ASM for ARM chips with an FPU
    #[cfg(all(target_arch = "arm", target_feature = "vfp2"))]
    {
        let mut result: f32;
        unsafe {
            core::arch::asm!(
                "vsqrt.f32 {res}, {val}", // takes approximately 14 CPU cycles
                res = out(vreg) result,
                val = in(vreg) self,
                options(nomem, nostack, preserves_flags)
            );
        }
        result
    }
    // Fallback for non-ARM, non-FPU target
    #[cfg(not(all(target_arch = "arm", target_feature = "vfp2")))]
    {
        sqrt_approx_f32(x)
    }
}

#[allow(unused)]
#[inline(always)]
pub fn sqrt_f64(x: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    f64::from(sqrt_f32(x as f32))
}

#[allow(unused)]
#[inline(always)]
pub fn sqrt_approx_f32(x: f32) -> f32 {
    1.0 / sqrt_reciprocal_approx_f32(x)
}

#[allow(unused)]
#[inline(always)]
pub fn sqrt_approx_f64(x: f64) -> f64 {
    1.0 / sqrt_reciprocal_approx_f64(x)
}

#[allow(unused)]
#[inline(always)]
pub fn sqrt_reciprocal_f32(x: f32) -> f32 {
    // Use hardware ASM for ARM chips with an FPU
    #[cfg(all(target_arch = "arm", target_feature = "vfp2"))]
    {
        let mut result: f32;
        unsafe {
            core::arch::asm!(
                "vrsqrt.f32 {res}, {val}",
                res = out(vreg) result,
                val = in(vreg) self,
                options(nomem, nostack)
            );
        }
        result
    }
    // Fallback for non-ARM, non-FPU target
    #[cfg(not(all(target_arch = "arm", target_feature = "vfp2")))]
    {
        sqrt_reciprocal_approx_f32(x)
    }
}

#[allow(unused)]
#[inline(always)]
pub fn sqrt_reciprocal_f64(x: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    f64::from(sqrt_reciprocal_f32(x as f32))
}

/// Implementation of [fast inverse square root](http://en.wikipedia.org/wiki/Fast_inverse_square_root)
/// using [Pizer’s optimization](https://pizer.wordpress.com/2008/10/12/fast-inverse-square-root/).
/// Maximum relative error: 0.00065.
#[allow(clippy::excessive_precision)]
#[allow(unused)]
#[inline(always)]
pub fn sqrt_reciprocal_approx_f32(x: f32) -> f32 {
    let i = 0x_5F1F_1412 - (x.to_bits().cast_signed() >> 1); // Initial estimate for Newton's method.
    let y = f32::from_bits(i.cast_unsigned());
    y * (1.690_002_31 - 0.714_158_168 * x * y * y) // First iteration of Newton's method.
}

#[allow(unused)]
#[inline(always)]
pub fn sqrt_reciprocal_approx_f64(x: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    f64::from(sqrt_reciprocal_approx_f32(x as f32))
}

/// Implementation of [fast inverse square root](http://en.wikipedia.org/wiki/Fast_inverse_square_root).
/// Maximum relative error: 0.001735.
#[allow(unused)]
#[inline(always)]
fn quake_sqrt_reciprocal_approx_f32(x: f32) -> f32 {
    let i = 0x_5F37_5A86 - (x.to_bits().cast_signed() >> 1); // Initial estimate for Newton's method.
    let y = f32::from_bits(i.cast_unsigned());
    y * (1.5 - 0.5 * x * y * y) // First iteration of Newton's method.
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn sqrt_reciprocal() {
        assert_eq!(quake_sqrt_reciprocal_approx_f32(4.0), 0.499_154_06);
        assert_eq!(sqrt_reciprocal_approx_f32(4.0), 0.500_059_37);
        assert_eq!(4.0.sqrt_reciprocal(), 0.5);
    }
    #[cfg(feature = "libm")]
    #[test]
    fn sqrt() {
        assert_eq!(0.0_f32.sqrt(), libm::sqrtf(0.0));
        assert_eq!(4.0_f32.sqrt(), libm::sqrtf(4.0));
    }
}
