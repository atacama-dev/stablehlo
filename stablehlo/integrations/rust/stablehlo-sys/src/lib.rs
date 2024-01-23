#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

type mlir_DialectRegistry = ::std::os::raw::c_void;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
