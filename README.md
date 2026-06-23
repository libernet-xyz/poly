# Polynomials

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/poly/ci.yml?label=CI)](https://github.com/libernet-xyz/poly/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-poly)](https://crates.io/crates/starkom-poly)
[![license](https://img.shields.io/crates/l/starkom-poly)](https://github.com/libernet-xyz/poly/blob/main/LICENSE)

## Overview

This crate contains several algorithms over polynomials used in Starkom (NTT, Lagrange
interpolation, etc.).

Most algorithms are implemented generically for any prime field. The main requirement is that the
field implements the
[`PrimeField`](https://docs.rs/starkom-ff/latest/starkom_ff/trait.PrimeField.html) trait provided in
the [`starkom-ff`](https://crates.io/crates/starkom-ff) crate.
