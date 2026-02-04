#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

mod fast_math;
mod matrix3x3;
mod quaternion;
mod vector3d;
mod vector3d_i16;
mod vector3d_i32;

pub use fast_math::FastMath;
pub use matrix3x3::Matrix3x3;
pub use quaternion::Quaternion;
pub use vector3d::Vector3d;
pub use vector3d_i16::Vector3dI16;
pub use vector3d_i32::Vector3dI32;
