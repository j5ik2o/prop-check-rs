# prop-check-rs

A Rust crate for property-based testing.

[![Workflow Status](https://github.com/j5ik2o/prop-check-rs/workflows/ci/badge.svg)](https://github.com/j5ik2o/prop-check-rs/actions?query=workflow%3A%22ci%22)
[![crates.io](https://img.shields.io/crates/v/prop-check-rs.svg)](https://crates.io/crates/prop-check-rs)
[![docs.rs](https://docs.rs/prop-check-rs/badge.svg)](https://docs.rs/prop-check-rs)
[![tokei](https://tokei.rs/b1/github/j5ik2o/prop-check-rs)](https://github.com/XAMPPRocky/tokei)

## Install to Cargo.toml

Add this to your `Cargo.toml`:

```toml
[dependencies]
prop-check-rs = "<<version>>"
```

## Usage

### Choose one value from a list

```rust
#[test]
fn test_one_of() -> Result<()> {
  let gen = Gens::one_of_values(['a', 'b', 'c', 'x', 'y', 'z']);
  let prop = for_all_gen(gen, move |value| {
      log::info!("value = {}", value);
      true
  });
  test_with_prop(prop, 1, 100, new_rng())
}
```

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
