use vqm::{Matrix9x9, Matrix9x9f32};

// **** Align

const _: () = assert!(size_of::<Matrix9x9<f32>>() == 324); // 384 if align 64
const _: () = assert!(align_of::<Matrix9x9<f32>>() == 4);

const _: () = assert!(size_of::<Matrix9x9<f64>>() == 648); // 704 if align 64
const _: () = assert!(align_of::<Matrix9x9<f64>>() == 8);

const _: () = {
    const SIZE: usize = 81;
    // 1. Gather all 81 constants into a compile-time array
    let indices = [
        Matrix9x9f32::M11,
        Matrix9x9f32::M12,
        Matrix9x9f32::M13,
        Matrix9x9f32::M14,
        Matrix9x9f32::M15,
        Matrix9x9f32::M16,
        Matrix9x9f32::M17,
        Matrix9x9f32::M18,
        Matrix9x9f32::M19,
        Matrix9x9f32::M21,
        Matrix9x9f32::M22,
        Matrix9x9f32::M23,
        Matrix9x9f32::M24,
        Matrix9x9f32::M25,
        Matrix9x9f32::M26,
        Matrix9x9f32::M27,
        Matrix9x9f32::M28,
        Matrix9x9f32::M29,
        Matrix9x9f32::M31,
        Matrix9x9f32::M32,
        Matrix9x9f32::M33,
        Matrix9x9f32::M34,
        Matrix9x9f32::M35,
        Matrix9x9f32::M36,
        Matrix9x9f32::M37,
        Matrix9x9f32::M38,
        Matrix9x9f32::M39,
        Matrix9x9f32::M41,
        Matrix9x9f32::M42,
        Matrix9x9f32::M43,
        Matrix9x9f32::M44,
        Matrix9x9f32::M45,
        Matrix9x9f32::M46,
        Matrix9x9f32::M47,
        Matrix9x9f32::M48,
        Matrix9x9f32::M49,
        Matrix9x9f32::M51,
        Matrix9x9f32::M52,
        Matrix9x9f32::M53,
        Matrix9x9f32::M54,
        Matrix9x9f32::M55,
        Matrix9x9f32::M56,
        Matrix9x9f32::M57,
        Matrix9x9f32::M58,
        Matrix9x9f32::M59,
        Matrix9x9f32::M61,
        Matrix9x9f32::M62,
        Matrix9x9f32::M63,
        Matrix9x9f32::M64,
        Matrix9x9f32::M65,
        Matrix9x9f32::M66,
        Matrix9x9f32::M67,
        Matrix9x9f32::M68,
        Matrix9x9f32::M69,
        Matrix9x9f32::M71,
        Matrix9x9f32::M72,
        Matrix9x9f32::M73,
        Matrix9x9f32::M74,
        Matrix9x9f32::M75,
        Matrix9x9f32::M76,
        Matrix9x9f32::M77,
        Matrix9x9f32::M78,
        Matrix9x9f32::M79,
        Matrix9x9f32::M81,
        Matrix9x9f32::M82,
        Matrix9x9f32::M83,
        Matrix9x9f32::M84,
        Matrix9x9f32::M85,
        Matrix9x9f32::M86,
        Matrix9x9f32::M87,
        Matrix9x9f32::M88,
        Matrix9x9f32::M89,
        Matrix9x9f32::M91,
        Matrix9x9f32::M92,
        Matrix9x9f32::M93,
        Matrix9x9f32::M94,
        Matrix9x9f32::M95,
        Matrix9x9f32::M96,
        Matrix9x9f32::M97,
        Matrix9x9f32::M98,
        Matrix9x9f32::M99,
    ];

    // 2. Compile-time verification loop
    let mut i = 0;
    while i < SIZE {
        // Assert every value is tracking tightly inside the 0..81 array bound
        assert!(indices[i] < 81, "Matrix9x9f32 index constant is out of bounds!");

        // Nested comparison to guarantee no two constants share the same memory cell
        let mut j = i + 1;
        while j < SIZE {
            assert!(indices[i] != indices[j], "Duplicate matrix index constant detected!");
            j += 1;
        }
        i += 1;
    }
};

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Matrix9x9<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use vqm::Matrix3x3f32;
    use super::*;

    #[rustfmt::skip]
    #[test]
    fn extract_9x3_array() {
        let m = Matrix9x9f32::new([
            11.0, 21.0, 31.0,   41.0, 51.0, 61.0, 71.0, 81.0, 91.0,
            12.0, 22.0, 32.0,   42.0, 52.0, 62.0, 72.0, 82.0, 92.0,
            13.0, 23.0, 33.0,   43.0, 53.0, 63.0, 73.0, 83.0, 93.0,

            14.0, 24.0, 34.0,   44.0, 54.0, 64.0, 74.0, 84.0, 94.0,
            15.0, 25.0, 35.0,   45.0, 55.0, 65.0, 75.0, 85.0, 95.0,
            16.0, 26.0, 36.0,   46.0, 56.0, 66.0, 76.0, 86.0, 96.0,

            17.0, 27.0, 37.0,   47.0, 57.0, 67.0, 77.0, 87.0, 97.0,
            18.0, 28.0, 38.0,   48.0, 58.0, 68.0, 78.0, 88.0, 98.0,
            19.0, 29.0, 39.0,   49.0, 59.0, 69.0, 78.0, 89.0, 99.0,
        ]);
        let n = m.extract_9x3_array();
        assert_eq!( n, [
            11.0, 12.0, 13.0,   14.0, 15.0, 16.0,   17.0, 18.0, 19.0,
            21.0, 22.0, 23.0,   24.0, 25.0, 26.0,   27.0, 28.0, 29.0,
            31.0, 32.0, 33.0,   34.0, 35.0, 36.0,   37.0, 38.0, 39.0,
        ]);
    }
    #[rustfmt::skip]
    #[test]
    fn multiply_9x3_by_3x3() {
        let m = Matrix9x9f32::new([
            11.0, 21.0, 31.0,   41.0, 51.0, 61.0, 71.0, 81.0, 91.0,
            12.0, 22.0, 32.0,   42.0, 52.0, 62.0, 72.0, 82.0, 92.0,
            13.0, 23.0, 33.0,   43.0, 53.0, 63.0, 73.0, 83.0, 93.0,

            14.0, 24.0, 34.0,   44.0, 54.0, 64.0, 74.0, 84.0, 94.0,
            15.0, 25.0, 35.0,   45.0, 55.0, 65.0, 75.0, 85.0, 95.0,
            16.0, 26.0, 36.0,   46.0, 56.0, 66.0, 76.0, 86.0, 96.0,

            17.0, 27.0, 37.0,   47.0, 57.0, 67.0, 77.0, 87.0, 97.0,
            18.0, 28.0, 38.0,   48.0, 58.0, 68.0, 78.0, 88.0, 98.0,
            19.0, 29.0, 39.0,   49.0, 59.0, 69.0, 78.0, 89.0, 99.0,
        ]);
        let i = Matrix3x3f32::identity();
        let (a, b, c) = m.multiply_9x3_by_3x3(i);
        assert_eq!(a, Matrix3x3f32::new([
            11.0, 21.0, 31.0,
            12.0, 22.0, 32.0,
            13.0, 23.0, 33.0,
        ]));
        assert_eq!(b, Matrix3x3f32::new([
            14.0, 24.0, 34.0,
            15.0, 25.0, 35.0,
            16.0, 26.0, 36.0,
        ]));
        assert_eq!(c, Matrix3x3f32::new([
            17.0, 27.0, 37.0,
            18.0, 28.0, 38.0,
            19.0, 29.0, 39.0,
        ]));
    }
}
