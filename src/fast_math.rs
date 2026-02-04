use core::f32::consts;

pub trait FastMath: Sized {
    fn sqrt(self) -> Self;
    fn reciprocal_sqrt(self) -> Self;
    fn half_reciprocal_sqrt(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, y: Self) -> Self;
}

impl FastMath for f32 {
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    fn reciprocal_sqrt(self) -> Self {
        1.0 / libm::sqrtf(self)
    }
    fn half_reciprocal_sqrt(self) -> Self {
        0.5 / libm::sqrtf(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        //(libm::sinf(self), libm::cosf(self))
        sin_cos(self)
    }
    fn sin(self) -> Self {
        sin(self)
    }
    fn cos(self) -> Self {
        cos(self)
    }
    fn tan(self) -> Self {
        libm::tanf(self)
    }
    fn asin(self) -> Self {
        libm::asinf(self)
    }
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    fn atan2(self, y: Self) -> Self {
        libm::atan2f(y, self)
    }
}

#[cfg(test)]
fn reciprocal_sqrtf(x: f32) -> f32 {
    let mut y: f32 = x;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F375A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.69000231 - 0.714158168 * x * y * y) // First iteration
}

#[cfg(test)]
fn quake_reciprocal_sqrt(number: f32) -> f32 {
    let mut y: f32 = number;
    let mut i: i32 = y.to_bits().cast_signed();
    i = 0x5F375A86 - (i >> 1);
    y = f32::from_bits(i.cast_unsigned());
    y * (1.5 - (number * 0.5 * y * y))
}

// see [Optimized Trigonometric Functions on TI Arm Cores](https://www.ti.com/lit/an/sprad27a/sprad27a.pdf)
// for explanation of range mapping and coefficients
// r (remainder) is in range [-0.5, 0.5] and pre-scaled by 2/PI
#[allow(clippy::excessive_precision)]
fn sin_poly5_r(r: f32) -> f32 {
    const C1: f32 = 1.57078719139;
    const C3: f32 = -0.64568519592;
    const C5: f32 = 0.077562883496;
    let r2 = r * r;
    r * (C1 + r2 * (C3 + r2 * C5))
}

#[allow(clippy::excessive_precision)]
fn cos_poly6_r(r: f32) -> f32 {
    const C2: f32 = -1.23369765282;
    const C4: f32 = 0.25360107422;
    const C6: f32 = -0.020408373326;
    let r2 = r * r;
    1.0 + r2 * (C2 + r2 * (C4 + r2 * C6))
}

// For sin/cos quadrant helper functions:
// 2 least significant bits of q are quadrant index, ie [0, 1, 2, 3].
fn sin_quadrant(r: f32, q: i32) -> f32 {
    if q & 1 == 0 {
        // even quadrant: use sin
        let sin = sin_poly5_r(r);
        return if q & 2 == 0 { sin } else { -sin };
    }
    // odd quadrant: use cos
    let cos = cos_poly6_r(r);
    if q & 2 == 0 { cos } else { -cos }
}

fn cos_quadrant(r: f32, q: i32) -> f32 {
    if q & 1 == 0 {
        // even quadrant: use cos
        let cos = cos_poly6_r(r);
        return if q & 2 == 0 { cos } else { -cos };
    }
    // odd quadrant: use sin
    let sin = sin_poly5_r(r);
    if q & 2 == 0 { -sin } else { sin }
}

fn sin_cos_quadrant(r: f32, q: i32) -> (f32, f32) {
    let sin = sin_poly5_r(r);
    let cos = cos_poly6_r(r);

    // map values according to quadrant
    let sin_cos = if q & 1 == 0 { (sin, cos) } else { (cos, -sin) };

    if q & 2 == 0 {
        sin_cos
    } else {
        (-sin_cos.0, -sin_cos.1)
    }
}

pub fn sin(x: f32) -> f32 {
    let t = x * consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q;
    sin_quadrant(r, q as i32)
}

pub fn cos(x: f32) -> f32 {
    let t = x * consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    cos_quadrant(r, q as i32)
}

pub fn sin_cos(x: f32) -> (f32, f32) {
    let t = x * consts::FRAC_2_PI; // so remainder will be scaled from range [-PI/4, PI/4] ([-45, 45] degrees) to [-0.5, 0.5]
    let q = libm::roundf(t); // nearest quadrant
    let r = t - q; // remainder in range [-0.5, 0.5]
    sin_cos_quadrant(r, q as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_sqrt() {
        assert_eq!(quake_reciprocal_sqrt(4.0), 0.49915406);
        assert_eq!(reciprocal_sqrtf(4.0), 0.49435496);
        assert_eq!(4.0.reciprocal_sqrt(), 0.5);
    }
    #[test]
    fn sqrt() {
        assert_eq!(0.0.sqrt(), libm::sqrtf(0.0));
    }
    #[test]
    fn asin() {
        assert_eq!(0.0.asin(), libm::asinf(0.0));
    }
    #[test]
    fn sin() {
        assert_eq!(0.0.sin(), 0.0);
        assert_eq!(
            10.0_f32.to_radians().sin(),
            libm::sinf(10.0_f32.to_radians())
        );
        assert_eq!(
            20.0_f32.to_radians().sin(),
            libm::sinf(20.0_f32.to_radians())
        );
        assert_eq!(
            30.0_f32.to_radians().sin(),
            libm::sinf(30.0_f32.to_radians())
        );
        assert_eq!(
            40.0_f32.to_radians().sin(),
            libm::sinf(40.0_f32.to_radians())
        );
        assert_eq!(
            50.0_f32.to_radians().sin(),
            libm::sinf(50.0_f32.to_radians())
        );
        assert_eq!(
            60.0_f32.to_radians().sin(),
            libm::sinf(60.0_f32.to_radians())
        );
        assert_eq!(
            70.0_f32.to_radians().sin(),
            libm::sinf(70.0_f32.to_radians())
        );
        assert_eq!(
            80.0_f32.to_radians().sin(),
            libm::sinf(80.0_f32.to_radians())
        );
        assert_eq!(
            90.0_f32.to_radians().sin(),
            libm::sinf(90.0_f32.to_radians())
        );
        assert_eq!(
            100.0_f32.to_radians().sin(),
            libm::sinf(100.0_f32.to_radians())
        );
        assert_eq!(
            110.0_f32.to_radians().sin(),
            libm::sinf(110.0_f32.to_radians())
        );
        assert_eq!(
            120.0_f32.to_radians().sin(),
            libm::sinf(120.0_f32.to_radians())
        );
        assert_eq!(
            130.0_f32.to_radians().sin(),
            libm::sinf(130.0_f32.to_radians())
        );
        assert_eq!(
            140.0_f32.to_radians().sin(),
            libm::sinf(140.0_f32.to_radians())
        );
        assert_eq!(
            150.0_f32.to_radians().sin(),
            libm::sinf(150.0_f32.to_radians())
        );
        assert_eq!(
            160.0_f32.to_radians().sin(),
            libm::sinf(160.0_f32.to_radians())
        );
        assert_eq!(
            170.0_f32.to_radians().sin(),
            libm::sinf(170.0_f32.to_radians())
        );
        assert_eq!(
            180.0_f32.to_radians().sin(),
            libm::sinf(180.0_f32.to_radians())
        );
        assert_eq!(
            190.0_f32.to_radians().sin(),
            libm::sinf(190.0_f32.to_radians())
        );
        assert_eq!(
            (-10.0_f32).to_radians().sin(),
            libm::sinf(-10.0_f32.to_radians())
        );
        assert_eq!(
            (-20.0_f32).to_radians().sin(),
            libm::sinf(-20.0_f32.to_radians())
        );
        assert_eq!(
            (-30.0_f32).to_radians().sin(),
            libm::sinf(-30.0_f32.to_radians())
        );
        assert_eq!(
            (-40.0_f32).to_radians().sin(),
            libm::sinf(-40.0_f32.to_radians())
        );
        assert_eq!(
            (-50.0_f32).to_radians().sin(),
            libm::sinf(-50.0_f32.to_radians())
        );
        assert_eq!(
            (-60.0_f32).to_radians().sin(),
            libm::sinf(-60.0_f32.to_radians())
        );
        assert_eq!(
            (-70.0_f32).to_radians().sin(),
            libm::sinf(-70.0_f32.to_radians())
        );
        assert_eq!(
            (-80.0_f32).to_radians().sin(),
            libm::sinf(-80.0_f32.to_radians())
        );
        assert_eq!(
            (-90.0_f32).to_radians().sin(),
            libm::sinf(-90.0_f32.to_radians())
        );
        assert_eq!(
            (-100.0_f32).to_radians().sin(),
            libm::sinf(-100.0_f32.to_radians())
        );
        assert_eq!(
            (-110.0_f32).to_radians().sin(),
            libm::sinf(-110.0_f32.to_radians())
        );
        assert_eq!(
            (-120.0_f32).to_radians().sin(),
            libm::sinf(-120.0_f32.to_radians())
        );
        assert_eq!(
            (-130.0_f32).to_radians().sin(),
            libm::sinf(-130.0_f32.to_radians())
        );
        assert_eq!(
            (-140.0_f32).to_radians().sin(),
            libm::sinf(-140.0_f32.to_radians())
        );
        assert_eq!(
            (-150.0_f32).to_radians().sin(),
            libm::sinf(-150.0_f32.to_radians())
        );
        assert_eq!(
            (-160.0_f32).to_radians().sin(),
            libm::sinf(-160.0_f32.to_radians())
        );
        assert_eq!(
            (-170.0_f32).to_radians().sin(),
            libm::sinf(-170.0_f32.to_radians())
        );
        assert_eq!(
            (-180.0_f32).to_radians().sin(),
            libm::sinf(-180.0_f32.to_radians())
        );
        assert_eq!(
            (-190.0_f32).to_radians().sin(),
            libm::sinf(-190.0_f32.to_radians())
        );
    }
    #[test]
    fn cos() {
        assert_eq!(0.0.cos(), 1.0);
        assert_eq!(
            10.0_f32.to_radians().cos(),
            libm::cosf(10.0_f32.to_radians())
        );
        assert_eq!(
            20.0_f32.to_radians().cos(),
            libm::cosf(20.0_f32.to_radians())
        );
        assert_eq!(
            30.0_f32.to_radians().cos(),
            libm::cosf(30.0_f32.to_radians())
        );
        assert_eq!(
            40.0_f32.to_radians().cos(),
            libm::cosf(40.0_f32.to_radians())
        );
        assert_eq!(
            50.0_f32.to_radians().cos(),
            libm::cosf(50.0_f32.to_radians())
        );
        assert_eq!(
            60.0_f32.to_radians().cos(),
            libm::cosf(60.0_f32.to_radians())
        );
        assert_eq!(
            70.0_f32.to_radians().cos(),
            libm::cosf(70.0_f32.to_radians())
        );
        assert_eq!(
            80.0_f32.to_radians().cos(),
            libm::cosf(80.0_f32.to_radians())
        );
        assert_eq!(
            90.0_f32.to_radians().cos(),
            libm::cosf(90.0_f32.to_radians())
        );
        assert_eq!(
            100.0_f32.to_radians().cos(),
            libm::cosf(100.0_f32.to_radians())
        );
        assert_eq!(
            110.0_f32.to_radians().cos(),
            libm::cosf(110.0_f32.to_radians())
        );
        assert_eq!(
            120.0_f32.to_radians().cos(),
            libm::cosf(120.0_f32.to_radians())
        );
        assert_eq!(
            130.0_f32.to_radians().cos(),
            libm::cosf(130.0_f32.to_radians())
        );
        assert_eq!(
            140.0_f32.to_radians().cos(),
            libm::cosf(140.0_f32.to_radians())
        );
        assert_eq!(
            150.0_f32.to_radians().cos(),
            libm::cosf(150.0_f32.to_radians())
        );
        assert_eq!(
            160.0_f32.to_radians().cos(),
            libm::cosf(160.0_f32.to_radians())
        );
        assert_eq!(
            170.0_f32.to_radians().cos(),
            libm::cosf(170.0_f32.to_radians())
        );
        assert_eq!(
            180.0_f32.to_radians().cos(),
            libm::cosf(180.0_f32.to_radians())
        );
        assert_eq!(
            190.0_f32.to_radians().cos(),
            libm::cosf(190.0_f32.to_radians())
        );
        assert_eq!(
            (-10.0_f32).to_radians().cos(),
            libm::cosf(-10.0_f32.to_radians())
        );
        assert_eq!(
            (-20.0_f32).to_radians().cos(),
            libm::cosf(-20.0_f32.to_radians())
        );
        assert_eq!(
            (-30.0_f32).to_radians().cos(),
            libm::cosf(-30.0_f32.to_radians())
        );
        assert_eq!(
            (-40.0_f32).to_radians().cos(),
            libm::cosf(-40.0_f32.to_radians())
        );
        assert_eq!(
            (-50.0_f32).to_radians().cos(),
            libm::cosf(-50.0_f32.to_radians())
        );
        assert_eq!(
            (-60.0_f32).to_radians().cos(),
            libm::cosf(-60.0_f32.to_radians())
        );
        assert_eq!(
            (-70.0_f32).to_radians().cos(),
            libm::cosf(-70.0_f32.to_radians())
        );
        assert_eq!(
            (-80.0_f32).to_radians().cos(),
            libm::cosf(-80.0_f32.to_radians())
        );
        assert_eq!(
            (-90.0_f32).to_radians().cos(),
            libm::cosf(-90.0_f32.to_radians())
        );
        assert_eq!(
            (-100.0_f32).to_radians().cos(),
            libm::cosf(-100.0_f32.to_radians())
        );
        assert_eq!(
            (-110.0_f32).to_radians().cos(),
            libm::cosf(-110.0_f32.to_radians())
        );
        assert_eq!(
            (-120.0_f32).to_radians().cos(),
            libm::cosf(-120.0_f32.to_radians())
        );
        assert_eq!(
            (-130.0_f32).to_radians().cos(),
            libm::cosf(-130.0_f32.to_radians())
        );
        assert_eq!(
            (-140.0_f32).to_radians().cos(),
            libm::cosf(-140.0_f32.to_radians())
        );
        assert_eq!(
            (-150.0_f32).to_radians().cos(),
            libm::cosf(-150.0_f32.to_radians())
        );
        assert_eq!(
            (-160.0_f32).to_radians().cos(),
            libm::cosf(-160.0_f32.to_radians())
        );
        assert_eq!(
            (-170.0_f32).to_radians().cos(),
            libm::cosf(-170.0_f32.to_radians())
        );
        assert_eq!(
            (-180.0_f32).to_radians().cos(),
            libm::cosf(-180.0_f32.to_radians())
        );
        assert_eq!(
            (-190.0_f32).to_radians().cos(),
            libm::cosf(-190.0_f32.to_radians())
        );
    }
    #[test]
    fn sin_cos() {
        assert_eq!(0.0.sin_cos(), (0.0, 1.0));
        assert_eq!(
            10.0_f32.to_radians().sin_cos(),
            (
                libm::sinf(10.0_f32.to_radians()),
                libm::cosf(10.0_f32.to_radians())
            )
        );
        assert_eq!(
            (-10.0_f32).to_radians().sin_cos(),
            (
                libm::sinf(-10.0_f32.to_radians()),
                libm::cosf(-10.0_f32.to_radians())
            )
        );
        assert_eq!(
            110.0_f32.to_radians().sin_cos(),
            (
                libm::sinf(110.0_f32.to_radians()),
                libm::cosf(110.0_f32.to_radians())
            )
        );
        assert_eq!(
            (-110.0_f32).to_radians().sin_cos(),
            (
                libm::sinf(-110.0_f32.to_radians()),
                libm::cosf(-110.0_f32.to_radians())
            )
        );
    }
    #[test]
    fn atan2() {
        assert_eq!(1.0.atan2(0.0), 0.0);
        assert_eq!(libm::atan2f(0.0, 1.0), 0.0);
    }
}
