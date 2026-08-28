use vqm::{Vector4, Vector4f32, Vector4f64};

const _: () = assert!(size_of::<Vector4f32>() == 16);
const _: () = assert!(align_of::<Vector4f32>() == 16);

const _: () = assert!(size_of::<Vector4f64>() == 32);
const _: () = assert!(align_of::<Vector4f64>() == 16);

#[cfg(test)]
mod test_traits {
    use super::*;

    #[cfg(feature = "serde")]
    use {
        postcard::experimental::max_size::MaxSize,
        sequential_storage::map::PostcardValue,
        serde::{Deserialize, Serialize},
    };

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}
    #[cfg(feature = "serde")]
    fn is_config<T: Serialize + MaxSize + for<'a> Deserialize<'a> + for<'a> PostcardValue<'a>>() {}

    #[test]
    fn normal_types() {
        is_full::<Vector4<f32>>();
        #[cfg(feature = "serde")]
        is_config::<Vector4<f32>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    #[test]
    fn default() {
        let a: Vector4f32 = Vector4f32::default();
        assert_eq!(Vector4f32::zero(), a);
    }
    #[test]
    fn zero() {
        use num_traits::{Zero, zero};
        let z: Vector4f32 = zero();
        assert!(z.is_zero());
    }
}
