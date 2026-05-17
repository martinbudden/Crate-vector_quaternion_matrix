use vqm::{Vector4d, Vector4df32, Vector4df64};

const _: () = assert!(core::mem::size_of::<Vector4df32>() == 16);
const _: () = assert!(core::mem::align_of::<Vector4df32>() == 16);

const _: () = assert!(core::mem::size_of::<Vector4df64>() == 32);
const _: () = assert!(core::mem::align_of::<Vector4df64>() == 16);

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;
    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    #[allow(unused)]
    fn is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}
    #[cfg(feature = "serde")]
    fn is_config<
        T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq + Serialize + for<'a> Deserialize<'a>,
    >() {
    }

    #[test]
    fn normal_types() {
        is_full::<Vector4d<f32>>();
        #[cfg(feature = "serde")]
        is_config::<Vector4d<f32>>();
    }
    #[test]
    fn default() {
        let a: Vector4df32 = Vector4df32::default();
        assert_eq!(Vector4df32::zero(), a);
    }
    #[test]
    fn zero() {
        use num_traits::{Zero, zero};
        let z: Vector4df32 = zero();
        assert!(z.is_zero());
    }
}
