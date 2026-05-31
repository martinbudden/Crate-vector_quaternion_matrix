# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Releases of the form `0.1.n` do not adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
that is each release may contain incompatible API changes.

Once the API has stabilized this project will adopt semantic versioning, the first release to do so will be `0.2.0`.

## [Possible future]

### Added

- May consider adding eigenvalues and eigenvectors.

## [Unreleased]

### Added

### Changed

### Removed

### Deprecated

At some point [0.1.0] to [0.1.8] will be [YANKED]

### Fixed

### Security

## [0.1.11] - 2026-05-31

### Changed

- Changed `TrigonometricMethods` trait to use `Num` trait rather than `Float` trait.
- Improved sqrt_reciprocal methods.

## [0.1.10] - 2026-05-27

### Added

- dot product to quaternion.
- `sequential-storage` support to `serde`.
- examples to `README.md`.
- `Display` trait to vectors.

### Changed

- fixed `atan2` parameter order error..

## [0.1.9] - 2026-05-24

### Added

- `Deref` and `DerefMut` traits to matrices.
- `AsRef` and `AsMut` traits to matrices.
- range traits to matrices.
- iterator traits to matrices.
- custom `Debug` trait to `Matrix4x4` and `Matrix9x9`.
- improved `multiply_9x3_by_3x3` for `Matrix9x9`.

### Changed

- Changed Apache license to standard unabridged text.

### Removed

- multiplication from `Matrix9x9`.
- `One` and `ConstOne` traits from `Matrix9x9` (necessitated by removal of multiplication).

## [0.1.8] - 2026-05-23

### Added

- `Mnn` constants to index matrix elements (eg `M11`, `M23` etc).
- `identity` functions for matrices.
- `Matrix9x9` - partial implementation to support Kalman filters.
- `outer_product` function to matrices.
- `KalmanStateVector9` to support Kalman filters.
- `cos_tilt` and `sin_tilt` functions to `Quaternion`.
- conversion functions to `RollPitch` and `RollPitchYaw`.

### Changed

- optimized `m3x3_mul_vector` to be more compiler-friendly for generating SIMD instructions.
- renamed `try_invert` to `try_inverse`.
- tidied `zero` and `one` functions.
- improved documentation.

### Removed

- `katex-header.html`

## [0.1.7] - 2026-05-17

### Added

- `.cargo/config.toml`

### Changed

- fixes to `Cargo.toml`, especially better handling of features.
- default no longer includes `serde`.
- release build no longer dependent on `approx` crate.
- fixed bug in `quaternion` `rotate`.

## [0.1.6] - 2026-05-16

### Changed

- `serde` to use `default-features = false`.

## [0.1.5] - 2026-05-16

### Added

- `ConstZero` traits to vectors, quaternions, and matrices.
- `ConstOne` traits to quaternions and matrices.

### Removed

- `BitSet64` and `BitSet128`.

## [0.1.4] - 2026-05-13

### Added

- `Deserialize` and `Serialize` for `BitSet64` and `BitSet128`.

### Changed

- removed return value from `set` and `reset` in `BitSet`s.

## [0.1.3] - 2026-05-06

### Added

- `BitSet64Iter`.

### Changed

- removed return value from `set` and `reset` in `BitSet`s.

## [0.1.2]

### Added

- Quaternion calculate_euler_angles_radians() and calculate_euler_angles_degrees() functions
- Elementwise multiplication and division of vectors.

### Changed

- changed many functions from `#[inline(always)]` to `#[inline]`

## [0.1.1]

### Added

- `Vector4d`, `Vector3di32`.
- `to_radians` and `to_degrees` for vectors.
- `BitSet64` and `BitSet128`.
- Benchmarks.
- Many `From` traits to convert between vectors of different dimensions, arrays and matrices.
- **CHANGELOG.md**, **ARCHITECTURE.md**, **CONTRIBUTING.md**.

### Changed

- Renamed crate from `vector-quaternion-matrix` to `vqm`.
- Changed back to use verbal form for return functions and _in_place suffix for in place versions,
  ie `transpose` and `transpose_in_place` rather than `transpose` and `transposed`.
- Updated README.md

## [0.1.0] - 2023-03-05

Initial release.
