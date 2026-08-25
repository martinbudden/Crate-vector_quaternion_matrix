use vqm::Matrix4x4;

// **** Align

const _: () = assert!(size_of::<Matrix4x4<f32>>() == 64);
const _: () = assert!(align_of::<Matrix4x4<f32>>() == 64);

const _: () = assert!(size_of::<Matrix4x4<f64>>() == 128);
const _: () = assert!(align_of::<Matrix4x4<f64>>() == 64);

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "serde")]
    use {
        postcard::experimental::max_size::MaxSize,
        sequential_storage::map::PostcardValue,
        serde::{Deserialize, Serialize},
    };

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}
    #[cfg(feature = "serde")]
    fn is_config<T: Serialize + MaxSize + for<'a> Deserialize<'a> + for<'a> PostcardValue<'a>>() {}

    #[test]
    fn normal_types() {
        is_full::<Matrix4x4<f32>>();
        #[cfg(feature = "serde")]
        is_config::<Matrix4x4<f32>>();
    }
}
