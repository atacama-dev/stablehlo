# tblgen

This crate provides raw bindings and a safe wrapper for TableGen, a domain-specific language used by the LLVM project.

The goal of this crate is to enable users to develop custom TableGen backends in Rust. Hence the primary use case of this crate are procedural macros that generate Rust code from TableGen description files.

## Documentation

Read the documentation at https://danacus.gitlab.io/tblgen-rs/tblgen/.

## Supported LLVM Versions

An installation of LLVM is required to use this crate. Both LLVM 16 and 17 are supported and can be selected using feature flags.

The `TABLEGEN_<version>_PREFIX` environment variable can be used to specify a custom directory of the LLVM installation.
