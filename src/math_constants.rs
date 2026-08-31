#![allow(clippy::excessive_precision)]

/// Math constants for use in generic code, eg `T:PI`, `T:SQRT_2` etc.<br><br>
pub trait MathConstants {
    const EPSILON: Self;

    const RADIANS_TO_DEGREES: Self;
    const DEGREES_TO_RADIANS: Self;

    /// Standard acceleration of earth gravity.
    const G0: Self;
    /// Reciprocal of standard acceleration of earth gravity.
    const G0_RECIPROCAL: Self;

    const HALF: Self;
    const TWO: Self;
    const THREE: Self;
    const FOUR: Self;
    const FIVE: Self;
    const SIX: Self;
    const SEVEN: Self;
    const EIGHT: Self;
    const NINE: Self;
    const TEN: Self;
    const ELEVEN: Self;
    const TWELVE: Self;
    const SIXTEEN: Self;
    const TWENTY_FIVE: Self;
    const SIXTY_FOUR: Self;
    const EIGHTY_ONE: Self;
    const TWO_FIFTY_SIX: Self;
    const SIX_TWENTY_FIVE: Self;
}

impl MathConstants for f32 {
    const EPSILON: Self = f32::EPSILON;

    const RADIANS_TO_DEGREES: f32 = 180.0 / core::f32::consts::PI;
    const DEGREES_TO_RADIANS: f32 = core::f32::consts::PI / 180.0;

    const G0: Self = 9.806_65;
    const G0_RECIPROCAL: Self = 1.0 / 9.806_65;

    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const THREE: Self = 3.0;
    const FOUR: Self = 4.0;
    const FIVE: Self = 5.0;
    const SIX: Self = 6.0;
    const SEVEN: Self = 7.0;
    const EIGHT: Self = 8.0;
    const NINE: Self = 9.0;
    const TEN: Self = 10.0;
    const ELEVEN: Self = 11.0;
    const TWELVE: Self = 12.0;
    const SIXTEEN: Self = 16.0;
    const TWENTY_FIVE: Self = 25.0;
    const SIXTY_FOUR: Self = 64.0;
    const EIGHTY_ONE: Self = 81.0;
    const TWO_FIFTY_SIX: Self = 256.0;
    const SIX_TWENTY_FIVE: Self = 625.0;
}

impl MathConstants for f64 {
    const EPSILON: Self = f64::EPSILON;

    const RADIANS_TO_DEGREES: f64 = 180.0 / core::f64::consts::PI;
    const DEGREES_TO_RADIANS: f64 = core::f64::consts::PI / 180.0;

    const G0: Self = 9.806_65;
    const G0_RECIPROCAL: Self = 1.0 / 9.806_65;

    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const THREE: Self = 3.0;
    const FOUR: Self = 4.0;
    const FIVE: Self = 5.0;
    const SIX: Self = 6.0;
    const SEVEN: Self = 7.0;
    const EIGHT: Self = 8.0;
    const NINE: Self = 9.0;
    const TEN: Self = 10.0;
    const ELEVEN: Self = 11.0;
    const TWELVE: Self = 12.0;
    const SIXTEEN: Self = 16.0;
    const TWENTY_FIVE: Self = 25.0;
    const SIXTY_FOUR: Self = 64.0;
    const EIGHTY_ONE: Self = 81.0;
    const TWO_FIFTY_SIX: Self = 256.0;
    const SIX_TWENTY_FIVE: Self = 625.0;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    #[allow(unused)]
    struct Test<F> {
        t: F,
    }
    impl<F> Test<F>
    where
        F: Copy + MathConstants,
    {
        fn half() -> F {
            F::HALF
        }
        fn two() -> F {
            F::TWO
        }
    }
    type Testf32 = Test<f32>;
    type Testf64 = Test<f64>;

    #[test]
    fn f32() {
        assert_eq!(0.5, Testf32::half());
        assert_eq!(2.0, Testf32::two());
    }
    #[test]
    fn f64() {
        assert_eq!(0.5, Testf64::half());
        assert_eq!(2.0, Testf64::two());
    }
}
