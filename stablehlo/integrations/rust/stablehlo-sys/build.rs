use std::{
    env,
    error::Error,
    fs, io,
    path::Path,
    process::{exit, Command},
    str,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

const STABLEHLO_LIBDIR: &'static str = "../../../../build/lib";
const LLVM_INSTALLDIR: &'static str = "../../../../llvm-install";

fn run() -> Result<(), Box<dyn Error>> {
    let llvm_libdir = llvm_config("--libdir")?;
    println!("cargo:rustc-link-search={}", STABLEHLO_LIBDIR);
    println!("cargo:rustc-link-search={}", llvm_libdir);
    println!("cargo:rerun-if-changed=wrapper.hpp");

    // link LLVM libs
    for name in llvm_config("--libnames")?.trim().split(' ') {
        if let Some(name) = trim_library_name(name) {
            println!("cargo:rustc-link-lib={}", name);
        }
    }

    println!("cargo:rustc-link-lib=static=LLVMSupport");
    println!("cargo:rustc-link-lib=static=LLVMDemangle");

    for flag in llvm_config("--system-libs")?.trim().split(' ') {
        let flag = flag.trim_start_matches("-l");

        if flag.starts_with('/') {
            // llvm-config returns absolute paths for dynamically linked libraries.
            let path = Path::new(flag);

            println!(
                "cargo:rustc-link-search={}",
                path.parent().unwrap().display()
            );
            println!(
                "cargo:rustc-link-lib={}",
                path.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .split_once('.')
                    .unwrap()
                    .0
                    .trim_start_matches("lib")
            );
        } else {
            println!("cargo:rustc-link-lib={}", flag);
        }
    }

    // statically link MLIR libs
    for name in fs::read_dir(llvm_libdir)?
        .map(|entry| {
            Ok(if let Some(name) = entry?.path().file_name() {
                name.to_str().map(String::from)
            } else {
                None
            })
        })
        .collect::<Result<Vec<_>, io::Error>>()?
        .into_iter()
        .flatten()
    {
        if name.starts_with("libMLIR")
            && name.ends_with(".a")
            && !name.contains("Main")
            && name != "libMLIRSupportIndentedOstream.a"
        {
            if let Some(name) = trim_library_name(&name) {
                println!("cargo:rustc-link-lib=static={}", name);
            }
        }
    }

    // statically link stablehlo libs
    for name in fs::read_dir(STABLEHLO_LIBDIR)?
        .map(|entry| {
            Ok(if let Some(name) = entry?.path().file_name() {
                name.to_str().map(String::from)
            } else {
                None
            })
        })
        .collect::<Result<Vec<_>, io::Error>>()?
        .into_iter()
        .flatten()
    {
        if let Some(name) = trim_library_name(&name) {
            println!("cargo:rustc-link-lib=static={}", name);
        }
    }

    // link zstd
    println!("cargo:rustc-link-lib=static=zstd");

    if let Some(name) = get_system_libcpp() {
        println!("cargo:rustc-link-lib={}", name);
    }

    let llvm_includedir = llvm_config("--includedir")?;
    let mlir_includedir = Path::new(llvm_includedir.as_str()).join("../../mlir/include");

    bindgen::builder()
        .header("wrapper.hpp")
        .clang_arg(format!("-I{}", llvm_includedir))
        .clang_arg(format!("-I{}", mlir_includedir.display()))
        .clang_arg("-std=c++17")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_recursively(false)
        .blocklist_type("mlir::DialectRegistry")
        .opaque_type("mlir::DialectRegistry")
        // .allowlist_file("../../../dialect/Register.h")
        // .allowlist_file(
        //     Path::new(llvm_config("--includepath")?.as_str())
        //         .join("mlir/IR/DialectRegistry.h")
        //         .display()
        //         .to_string(),
        // )
        .allowlist_function("mlir::stablehlo::registerAllDialects")
        .generate()
        .unwrap()
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

fn llvm_config(argument: &str) -> Result<String, Box<dyn Error>> {
    let prefix = Path::new(LLVM_INSTALLDIR).join("bin");
    let call = format!(
        "{} --link-static {}",
        prefix.join("llvm-config").display(),
        argument
    );

    Ok(str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}

fn get_system_libcpp() -> Option<&'static str> {
    if cfg!(target_env = "msvc") {
        None
    } else if cfg!(target_os = "macos") {
        Some("c++")
    } else {
        Some("stdc++")
    }
}

fn trim_library_name(name: &str) -> Option<&str> {
    if let Some(name) = name.strip_prefix("lib") {
        name.strip_suffix(".a")
    } else {
        None
    }
}
