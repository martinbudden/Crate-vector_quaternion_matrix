use vqm::{Matrix9x9, Matrix9x9f32};

// **** Align

const _: () = assert!(core::mem::size_of::<Matrix9x9<f32>>() == 324); // 384 if align 64
const _: () = assert!(core::mem::align_of::<Matrix9x9<f32>>() == 4);

const _: () = assert!(core::mem::size_of::<Matrix9x9<f64>>() == 648); // 704 if align 64
const _: () = assert!(core::mem::align_of::<Matrix9x9<f64>>() == 8);

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
mod tests {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Matrix9x9<f32>>();
    }
}
