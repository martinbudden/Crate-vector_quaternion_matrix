#![cfg_attr(feature = "uom", doc = include_str!("../README.md"))]
#![cfg_attr(feature = "simd", feature(portable_simd, min_specialization))]
#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
//#![deny(missing_docs)]
#![deny(
    missing_copy_implementations,
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_must_use,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_results
)]
#![warn(unused_results)]
#![warn(clippy::pedantic)]
#![warn(clippy::doc_paragraphs_missing_punctuation)]

mod math_constants;
mod math_methods;
mod sqrt_methods;

mod vector2d;
mod vector2d_math;

mod vector3d;
mod vector3d_math;

mod vector4d;
mod vector4d_math;

mod matrix2x2;
mod matrix2x2_math;

mod matrix3x3;
mod matrix3x3_math;

mod matrix4x4;
mod matrix4x4_math;
mod matrix9x9;
mod matrix9x9_math;

mod quaternion;
mod quaternion_math;
mod roll_pitch_yaw;

pub use math_constants::MathConstants;
pub use math_methods::TrigonometricMethods;

// The trigonometric approximation functions need to be pub for benchmarking, but are not documented for general use.
#[doc(hidden)]
pub use math_methods::{cos_approx, sin_approx, sin_cos_approx};
pub use sqrt_methods::SqrtMethods;

pub use vector2d::{Vector2d, Vector2df32, Vector2df64};

pub use vector3d::{Vector3d, Vector3df32, Vector3df64, Vector3di16, Vector3di32};

pub use vector4d::{Vector4d, Vector4df32, Vector4df64};

pub use quaternion::{Quaternion, Quaternionf32, Quaternionf64};
pub use quaternion_math::QuaternionMath;
pub use roll_pitch_yaw::{RollPitch, RollPitchYaw, RollPitchYawf32, RollPitchYawf64, RollPitchf32, RollPitchf64};

pub use matrix2x2::{Matrix2x2, Matrix2x2f32, Matrix2x2f64};
pub use matrix2x2_math::Matrix2x2Math;

pub use matrix3x3::{Matrix3x3, Matrix3x3f32, Matrix3x3f64};
pub use matrix3x3_math::Matrix3x3Math;

pub use matrix4x4::{Matrix4x4, Matrix4x4f32, Matrix4x4f64};
pub use matrix4x4_math::Matrix4x4Math;

pub use matrix9x9::{Matrix9x9, Matrix9x9f32, Matrix9x9f64};
pub use matrix9x9_math::Matrix9x9Math;
