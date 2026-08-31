#![allow(clippy::inline_always)]
#![allow(clippy::excessive_precision)]

use core::ops::Neg;
use num_traits::{Num, float::FloatCore};
// see [Optimized Trigonometric Functions on TI Arm Cores](https://www.ti.com/lit/an/sprad27a/sprad27a.pdf)
// for explanation of range mapping and coefficients
// r (remainder) is in range [-0.5, 0.5] and pre-scaled by 2/PI
trait Sin5Coefficients {
    const SIN_C1: Self;
    const SIN_C3: Self;
    const SIN_C5: Self;
}

trait Cos6Coefficients {
    const COS_C0: Self;
    const COS_C2: Self;
    const COS_C4: Self;
    const COS_C6: Self;
}

trait ATan7Coefficients {
    const ATAN_C1: Self;
    const ATAN_C3: Self;
    const ATAN_C5: Self;
    const ATAN_C7: Self;
}

// sin4 (5.60E-07): x * (0.9999949932098388671875 + x2*(-0.166601598262786865234375 + x2*8.12153331935405731201171875e-3))
// sin5 (1.80E-09): x * (1 + x2*(-0.166666507720947265625 +x2*(8.331983350217342376708984375e-3 + x2*(-1.94961365195922553539276123046875e-4))))
// cos4 (6.70E-06): 0.999990046024322509765625 + x2*(-0.4997082054615020751953125 + x2*4.03986163437366485595703125e-2))
// cos5 (6.00E-08): 0.999999940395355224609375 + x2*(-0.499998986721038818359375 + x2*(4.1663490235805511474609375e-2 + x2*(-1.385320327244699001312255859375e-3 + x2*2.31450176215730607509613037109375e-5)))
// tan4 (8.00E-05): x * (0.99921381473541259765625 + x2 * (-0.321175038814544677734375 + x2 * (0.146264731884002685546875 + x2 * (-3.8986742496490478515625e-2))))
// tan4 (2.30E-05): x * (0.999970018863677978515625 + x2 * (-0.3317006528377532958984375 + x2 * (0.1852150261402130126953125 + x2 * (-9.1925732791423797607421875e-2 + x2 * 2.386303804814815521240234375e-2))))

impl Sin5Coefficients for f32 {
    const SIN_C1: Self = core::f32::consts::FRAC_PI_2;
    const SIN_C3: Self = -0.645_685_195_92;
    const SIN_C5: Self = 0.077_562_883_496;
}

impl Sin5Coefficients for f64 {
    const SIN_C1: Self = core::f64::consts::FRAC_PI_2;
    const SIN_C3: Self = -0.645_685_195_92;
    const SIN_C5: Self = 0.077_562_883_496;
}

impl Cos6Coefficients for f32 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.233_697_652_82;
    const COS_C4: Self = 0.253_601_074_22;
    const COS_C6: Self = -0.020_408_373_326;
}

impl Cos6Coefficients for f64 {
    const COS_C0: Self = 1.0;
    const COS_C2: Self = -1.233_697_652_82;
    const COS_C4: Self = 0.253_601_074_22;
    const COS_C6: Self = -0.020_408_373_326;
}

impl ATan7Coefficients for f32 {
    const ATAN_C1: Self = 0.999_213_814_735_412_597_656_25;
    const ATAN_C3: Self = -0.321_175_038_814_544_677_734_375;
    const ATAN_C5: Self = 0.146_264_731_884_002_685_546_875;
    const ATAN_C7: Self = -3.898_674_249_649_047_851_562_5e-2;
}

impl ATan7Coefficients for f64 {
    const ATAN_C1: Self = 0.999_213_814_735_412_597_656_25;
    const ATAN_C3: Self = -0.321_175_038_814_544_677_734_375;
    const ATAN_C5: Self = 0.146_264_731_884_002_685_546_875;
    const ATAN_C7: Self = -3.898_674_249_649_047_851_562_5e-2;
}

#[inline(always)]
fn sin_poly5<T>(r: T) -> T
where
    T: Copy + Num + Sin5Coefficients,
{
    let r2 = r * r;
    r * (T::SIN_C1 + r2 * (T::SIN_C3 + r2 * T::SIN_C5))
}

#[inline(always)]
fn cos_poly6<T>(r: T) -> T
where
    T: Copy + Num + Cos6Coefficients,
{
    let r2 = r * r;
    T::COS_C0 + r2 * (T::COS_C2 + r2 * (T::COS_C4 + r2 * T::COS_C6))
}

#[allow(unused)]
#[inline(always)]
fn atan_poly7<T>(r: T) -> T
where
    T: Copy + Num + ATan7Coefficients,
{
    let r2 = r * r;
    r * (T::ATAN_C1 + r2 * (T::ATAN_C3 + r2 * (T::ATAN_C5 + r2 * T::ATAN_C7)))
}

// For sin/cos quadrant helper functions:
// 2 least significant bits of q are quadrant index, ie [0, 1, 2, 3].
fn sin_quadrant<T>(r: T, q: i32) -> T
where
    T: Copy + Num + Neg<Output = T> + Sin5Coefficients + Cos6Coefficients,
{
    if q & 1 == 0 {
        // even quadrant: use sin
        let sin = sin_poly5::<T>(r);
        return if q & 2 == 0 { sin } else { -sin };
    }
    // odd quadrant: use cos
    let cos = cos_poly6::<T>(r);
    if q & 2 == 0 { cos } else { -cos }
}

fn cos_quadrant<T>(r: T, q: i32) -> T
where
    T: Copy + Num + Neg<Output = T> + Sin5Coefficients + Cos6Coefficients,
{
    if q & 1 == 0 {
        // even quadrant: use cos
        let cos = cos_poly6::<T>(r);
        return if q & 2 == 0 { cos } else { -cos };
    }
    // odd quadrant: use sin
    let sin = sin_poly5::<T>(r);
    if q & 2 == 0 { -sin } else { sin }
}

fn sin_cos_quadrant<T>(r: T, q: i32) -> (T, T)
where
    T: Copy + Num + Neg<Output = T> + Sin5Coefficients + Cos6Coefficients,
{
    let sin = sin_poly5::<T>(r);
    let cos = cos_poly6::<T>(r);

    // map values according to quadrant
    let sin_cos = if q & 1 == 0 { (sin, cos) } else { (cos, -sin) };

    if q & 2 == 0 { sin_cos } else { (-sin_cos.0, -sin_cos.1) }
}

fn atan_quadrant<T>(r: T) -> T
where
    T: Copy + Num + Neg<Output = T> + ATan7Coefficients,
{
    atan_poly7::<T>(r)
}

#[allow(unused)]
#[must_use]
pub fn sin_approx_f32(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q;
    #[allow(clippy::cast_possible_truncation)]
    sin_quadrant(r, q as i32)
}

#[allow(unused)]
#[must_use]
pub fn sin_approx_f64(x: f64) -> f64 {
    let t = x * core::f64::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q;
    #[allow(clippy::cast_possible_truncation)]
    sin_quadrant(r, q as i32)
}

#[allow(unused)]
#[must_use]
pub fn cos_approx_f32(x: f32) -> f32 {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    cos_quadrant(r, q as i32)
}

#[allow(unused)]
#[must_use]
pub fn cos_approx_f64(x: f64) -> f64 {
    let t = x * core::f64::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    cos_quadrant(r, q as i32)
}

#[must_use]
pub fn sin_cos_approx_f32(x: f32) -> (f32, f32) {
    let t = x * core::f32::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    sin_cos_quadrant(r, q as i32)
}

#[allow(unused)]
#[must_use]
pub fn sin_cos_approx_f64(x: f64) -> (f64, f64) {
    let t = x * core::f64::consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = FloatCore::round(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    #[allow(clippy::cast_possible_truncation)]
    sin_cos_quadrant(r, q as i32)
}

#[allow(unused)]
#[must_use]
pub fn tan_approx_f32(x: f32) -> f32 {
    let (sin, cos) = sin_cos_approx_f32(x);
    sin / cos
}

#[allow(unused)]
#[must_use]
pub fn tan_approx_f64(x: f64) -> f64 {
    let (sin, cos) = sin_cos_approx_f64(x);
    sin / cos
}

/// Arctangent of y/x (f64).
#[allow(unused)]
#[must_use]
#[allow(clippy::panic)]
pub fn atan2_approx_f64(x: f64, y: f64) -> f64 {
    let r = x / y;
    if r > 1.0 {
        return core::f64::consts::FRAC_PI_2 - atan_quadrant(1.0 / r);
    }
    let q = 1;
    atan_quadrant(r)
}

/// Arctangent of y/x (f32).
#[allow(unused)]
#[must_use]
#[allow(clippy::panic)]
pub fn atan2_approx_f32(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }

    let abs_y = y.abs();
    let abs_x = x.abs();

    // Octant reduction
    let (ratio, offset, sign) = if abs_y > abs_x {
        // Octant 2: FRAC_PI_2 - atan(x/y)
        (abs_x / abs_y, core::f32::consts::FRAC_PI_2, -1.0)
    } else {
        // Octant 1: 0.0 + atan(y/x)
        (abs_y / abs_x, 0.0, 1.0)
    };

    // Calculate core first-quadrant angle cleanly
    let mut angle = offset + (sign * atan_quadrant(ratio));

    // Map back to the correct quadrant based on original signs
    if x < 0.0 {
        angle = core::f32::consts::PI - angle;
    }
    if y < 0.0 {
        angle = -angle;
    }

    angle
}

#[cfg(test)]
mod tests {
    #![allow(unused)]
    use super::*;
    use approx::assert_abs_diff_eq;
    macro_rules! assert_near {
        ($left:expr, $right:expr) => {
            approx::assert_abs_diff_eq!($left, $right, epsilon = 4e-6);
        };
    }

    #[cfg(feature = "libm")]
    #[test]
    fn asin() {
        assert_abs_diff_eq!(0.0_f32.asin(), libm::asinf(0.0));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn sin() {
        assert_near!(sin_approx_f32(10.0_f32.to_radians()), libm::sinf(10.0_f32.to_radians()));
        assert_near!(sin_approx_f32(20.0_f32.to_radians()), libm::sinf(20.0_f32.to_radians()));
        assert_near!(sin_approx_f32(30.0_f32.to_radians()), libm::sinf(30.0_f32.to_radians()));
        assert_near!(sin_approx_f32(40.0_f32.to_radians()), libm::sinf(40.0_f32.to_radians()));
        assert_near!(sin_approx_f32(50.0_f32.to_radians()), libm::sinf(50.0_f32.to_radians()));
        assert_near!(sin_approx_f32(60.0_f32.to_radians()), libm::sinf(60.0_f32.to_radians()));
        assert_near!(sin_approx_f32(70.0_f32.to_radians()), libm::sinf(70.0_f32.to_radians()));
        assert_near!(sin_approx_f32(80.0_f32.to_radians()), libm::sinf(80.0_f32.to_radians()));
        assert_near!(sin_approx_f32(90.0_f32.to_radians()), libm::sinf(90.0_f32.to_radians()));
        assert_near!(sin_approx_f32(100.0_f32.to_radians()), libm::sinf(100.0_f32.to_radians()));
        assert_near!(sin_approx_f32(110.0_f32.to_radians()), libm::sinf(110.0_f32.to_radians()));
        assert_near!(sin_approx_f32(120.0_f32.to_radians()), libm::sinf(120.0_f32.to_radians()));
        assert_near!(sin_approx_f32(130.0_f32.to_radians()), libm::sinf(130.0_f32.to_radians()));
        assert_near!(sin_approx_f32(140.0_f32.to_radians()), libm::sinf(140.0_f32.to_radians()));
        assert_near!(sin_approx_f32(150.0_f32.to_radians()), libm::sinf(150.0_f32.to_radians()));
        assert_near!(sin_approx_f32(160.0_f32.to_radians()), libm::sinf(160.0_f32.to_radians()));
        assert_near!(sin_approx_f32(170.0_f32.to_radians()), libm::sinf(170.0_f32.to_radians()));
        assert_near!(sin_approx_f32(180.0_f32.to_radians()), libm::sinf(180.0_f32.to_radians()));
        assert_near!(sin_approx_f32(190.0_f32.to_radians()), libm::sinf(190.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-10.0_f32.to_radians()), libm::sinf(-10.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-20.0_f32.to_radians()), libm::sinf(-20.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-30.0_f32.to_radians()), libm::sinf(-30.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-40.0_f32.to_radians()), libm::sinf(-40.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-50.0_f32.to_radians()), libm::sinf(-50.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-60.0_f32.to_radians()), libm::sinf(-60.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-70.0_f32.to_radians()), libm::sinf(-70.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-80.0_f32.to_radians()), libm::sinf(-80.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-90.0_f32.to_radians()), libm::sinf(-90.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-100.0_f32.to_radians()), libm::sinf(-100.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-110.0_f32.to_radians()), libm::sinf(-110.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-120.0_f32.to_radians()), libm::sinf(-120.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-130.0_f32.to_radians()), libm::sinf(-130.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-140.0_f32.to_radians()), libm::sinf(-140.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-150.0_f32.to_radians()), libm::sinf(-150.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-160.0_f32.to_radians()), libm::sinf(-160.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-170.0_f32.to_radians()), libm::sinf(-170.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-180.0_f32.to_radians()), libm::sinf(-180.0_f32.to_radians()));
        assert_near!(sin_approx_f32(-190.0_f32.to_radians()), libm::sinf(-190.0_f32.to_radians()));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn cos() {
        assert_near!(cos_approx_f32(10.0_f32.to_radians()), libm::cosf(10.0_f32.to_radians()));
        assert_near!(cos_approx_f32(20.0_f32.to_radians()), libm::cosf(20.0_f32.to_radians()));
        assert_near!(cos_approx_f32(30.0_f32.to_radians()), libm::cosf(30.0_f32.to_radians()));
        assert_near!(cos_approx_f32(40.0_f32.to_radians()), libm::cosf(40.0_f32.to_radians()));
        assert_near!(cos_approx_f32(50.0_f32.to_radians()), libm::cosf(50.0_f32.to_radians()));
        assert_near!(cos_approx_f32(60.0_f32.to_radians()), libm::cosf(60.0_f32.to_radians()));
        assert_near!(cos_approx_f32(70.0_f32.to_radians()), libm::cosf(70.0_f32.to_radians()));
        assert_near!(cos_approx_f32(80.0_f32.to_radians()), libm::cosf(80.0_f32.to_radians()));
        assert_near!(cos_approx_f32(90.0_f32.to_radians()), libm::cosf(90.0_f32.to_radians()));
        assert_near!(cos_approx_f32(100.0_f32.to_radians()), libm::cosf(100.0_f32.to_radians()));
        assert_near!(cos_approx_f32(110.0_f32.to_radians()), libm::cosf(110.0_f32.to_radians()));
        assert_near!(cos_approx_f32(120.0_f32.to_radians()), libm::cosf(120.0_f32.to_radians()));
        assert_near!(cos_approx_f32(130.0_f32.to_radians()), libm::cosf(130.0_f32.to_radians()));
        assert_near!(cos_approx_f32(140.0_f32.to_radians()), libm::cosf(140.0_f32.to_radians()));
        assert_near!(cos_approx_f32(150.0_f32.to_radians()), libm::cosf(150.0_f32.to_radians()));
        assert_near!(cos_approx_f32(160.0_f32.to_radians()), libm::cosf(160.0_f32.to_radians()));
        assert_near!(cos_approx_f32(170.0_f32.to_radians()), libm::cosf(170.0_f32.to_radians()));
        assert_near!(cos_approx_f32(180.0_f32.to_radians()), libm::cosf(180.0_f32.to_radians()));
        assert_near!(cos_approx_f32(190.0_f32.to_radians()), libm::cosf(190.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-10.0_f32.to_radians()), libm::cosf(-10.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-20.0_f32.to_radians()), libm::cosf(-20.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-30.0_f32.to_radians()), libm::cosf(-30.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-40.0_f32.to_radians()), libm::cosf(-40.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-50.0_f32.to_radians()), libm::cosf(-50.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-60.0_f32.to_radians()), libm::cosf(-60.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-70.0_f32.to_radians()), libm::cosf(-70.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-80.0_f32.to_radians()), libm::cosf(-80.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-90.0_f32.to_radians()), libm::cosf(-90.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-100.0_f32.to_radians()), libm::cosf(-100.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-110.0_f32.to_radians()), libm::cosf(-110.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-120.0_f32.to_radians()), libm::cosf(-120.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-130.0_f32.to_radians()), libm::cosf(-130.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-140.0_f32.to_radians()), libm::cosf(-140.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-150.0_f32.to_radians()), libm::cosf(-150.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-160.0_f32.to_radians()), libm::cosf(-160.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-170.0_f32.to_radians()), libm::cosf(-170.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-180.0_f32.to_radians()), libm::cosf(-180.0_f32.to_radians()));
        assert_near!(cos_approx_f32(-190.0_f32.to_radians()), libm::cosf(-190.0_f32.to_radians()));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn sin_cos() {
        let (sin, cos) = 0.0_f32.sin_cos();
        assert_near!(sin, libm::sinf(0.0));
        assert_near!(cos, libm::cosf(0.0));

        let (sin, cos) = 10.0_f32.to_radians().sin_cos();
        assert_near!(sin, libm::sinf(10.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(10.0_f32.to_radians()));

        let (sin, cos) = (-10.0_f32).to_radians().sin_cos();
        assert_near!(sin, libm::sinf(-10.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(-10.0_f32.to_radians()));

        let (sin, cos) = 110.0_f32.to_radians().sin_cos();
        assert_near!(sin, libm::sinf(110.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(110.0_f32.to_radians()));

        let (sin, cos) = (-110.0_f32).to_radians().sin_cos();
        assert_near!(sin, libm::sinf(-110.0_f32.to_radians()));
        assert_near!(cos, libm::cosf(-110.0_f32.to_radians()));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn atan2() {
        assert_abs_diff_eq!(0.0_f32.atan2(1.0), 0.0);
        assert_abs_diff_eq!(libm::atan2f(0.0, 1.0), 0.0);

        assert_abs_diff_eq!(1.0_f32.atan2(0.0), libm::atan2f(1.0, 0.0));
        assert_abs_diff_eq!(-1.0_f32.atan2(0.0), libm::atan2f(-1.0, 0.0));

        assert_abs_diff_eq!(0.0_f32.atan2(1.0), libm::atan2f(0.0, 1.0));
        assert_abs_diff_eq!(0.1_f32.atan2(1.0), libm::atan2f(0.1, 1.0));
        assert_abs_diff_eq!(0.5_f32.atan2(1.0), libm::atan2f(0.5, 1.0));
        assert_abs_diff_eq!(1.0_f32.atan2(1.0), libm::atan2f(1.0, 1.0));
        assert_abs_diff_eq!(2.0_f32.atan2(1.0), libm::atan2f(2.0, 1.0));
        assert_abs_diff_eq!(8.0_f32.atan2(1.0), libm::atan2f(8.0, 1.0));
        assert_abs_diff_eq!(1000.0_f32.atan2(0.0), libm::atan2f(1000.0, 0.0));

        assert_abs_diff_eq!(-0.1_f32.atan2(1.0), libm::atan2f(-0.1, 1.0));
        assert_abs_diff_eq!(-0.5_f32.atan2(1.0), libm::atan2f(-0.5, 1.0));
        assert_abs_diff_eq!(-1.0_f32.atan2(1.0), libm::atan2f(-1.0, 1.0));
        assert_abs_diff_eq!(-2.0_f32.atan2(1.0), libm::atan2f(-2.0, 1.0));
        assert_abs_diff_eq!(-8.0_f32.atan2(1.0), libm::atan2f(-8.0, 1.0));
        assert_abs_diff_eq!(-1000.0_f32.atan2(0.0), libm::atan2f(-1000.0, 0.0));
    }
    #[cfg(feature = "libm")]
    #[test]
    fn atan2_approx() {
        assert_abs_diff_eq!(atan2_approx_f32(0.0, 0.0), libm::atan2f(0.0, 0.0), epsilon = 7.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(0.0, 1.0), libm::atan2f(0.0, 1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.1, 1.0), libm::atan2f(0.1, 1.0), epsilon = 7.0e-5); // 0.09966865
        assert_abs_diff_eq!(atan2_approx_f32(0.5, 1.0), libm::atan2f(0.5, 1.0), epsilon = 8.0e-5); // 0.4636476
        assert_abs_diff_eq!(atan2_approx_f32(1.0, 1.0), libm::atan2f(1.0, 1.0), epsilon = 8.2e-5); // 0.7853982, PI/4
        assert_abs_diff_eq!(atan2_approx_f32(2.0, 1.0), libm::atan2f(2.0, 1.0), epsilon = 8.0e-5); // 1.1071488
        assert_abs_diff_eq!(atan2_approx_f32(8.0, 1.0), libm::atan2f(8.0, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(100.0, 1.0), libm::atan2f(100.00, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1000.0, 1.0), libm::atan2f(1000.0, 1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(-0.1, 1.0), libm::atan2f(-0.1, 1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.5, 1.0), libm::atan2f(-0.5, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1.0, 1.0), libm::atan2f(-1.0, 1.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-2.0, 1.0), libm::atan2f(-2.0, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-8.0, 1.0), libm::atan2f(-8.0, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-100.0, 1.0), libm::atan2f(-100.00, 1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1000.0, 1.0), libm::atan2f(-1000.0, 1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(0.0, -1.0), libm::atan2f(0.0, -1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.1, -1.0), libm::atan2f(0.1, -1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.5, -1.0), libm::atan2f(0.5, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1.0, -1.0), libm::atan2f(1.0, -1.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(2.0, -1.0), libm::atan2f(2.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(8.0, -1.0), libm::atan2f(8.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(100.0, -1.0), libm::atan2f(100.00, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1000.0, -1.0), libm::atan2f(1000.0, -1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(-0.1, -1.0), libm::atan2f(-0.1, -1.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.5, -1.0), libm::atan2f(-0.5, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1.0, -1.0), libm::atan2f(-1.0, -1.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-2.0, -1.0), libm::atan2f(-2.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-8.0, -1.0), libm::atan2f(-8.0, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-100.0, -1.0), libm::atan2f(-100.00, -1.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1000.0, -1.0), libm::atan2f(-1000.0, -1.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(0.1, 0.0), libm::atan2f(0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.1, 0.0), libm::atan2f(0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(0.5, 0.0), libm::atan2f(0.5, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1.0, 0.0), libm::atan2f(1.0, 0.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(2.0, 0.0), libm::atan2f(2.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(8.0, 0.0), libm::atan2f(8.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(100.0, 0.0), libm::atan2f(100.00, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(1000.0, 0.0), libm::atan2f(1000.0, 0.0), epsilon = 8.0e-5);

        assert_abs_diff_eq!(atan2_approx_f32(-0.1, 0.0), libm::atan2f(-0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.1, 0.0), libm::atan2f(-0.1, 0.0), epsilon = 7.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-0.5, 0.0), libm::atan2f(-0.5, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1.0, 0.0), libm::atan2f(-1.0, 0.0), epsilon = 8.2e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-2.0, 0.0), libm::atan2f(-2.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-8.0, 0.0), libm::atan2f(-8.0, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-100.0, 0.0), libm::atan2f(-100.00, 0.0), epsilon = 8.0e-5);
        assert_abs_diff_eq!(atan2_approx_f32(-1000.0, 0.0), libm::atan2f(-1000.0, 0.0), epsilon = 8.0e-5);
    }
}
