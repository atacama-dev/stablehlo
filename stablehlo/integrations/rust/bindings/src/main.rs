use melior::ir::{
    attribute::{StringAttribute, TypeAttribute},
    r#type::FunctionType,
    *,
};
use melior::{dialect::func, dialect::DialectRegistry, Context};
// use std::{env, fmt::Display, path::Path, process::Command, str};
// use tblgen::{record::Record, RecordKeeper, TableGenParser};

// melior_macro::dialect! {
//     name: "affine",
//     table_gen: r#"include "mlir/Dialect/Affine/IR/AffineOps.td""#,
//     include_dirs: ["/home/ilias/sources/stablehlo/llvm-install/include"]
// }

melior::dialect! {
    name: "stablehlo",
    table_gen: r#"include "stablehlo/dialect/StablehloOps.td""#,
    include_dirs: ["/home/ilias/sources/stablehlo/"] ,
}

// #[link(name = "StablehloRegister", kind = "static")]
// extern "C" {
//     fn _ZN4mlir9stablehlo19registerAllDialectsERNS_15DialectRegistryE(x: MlirDialectRegistry)
//         -> ();
//     // registerAllDialects
// }

/// Registers all dialects to a dialect registry.
pub fn stablehlo_register_all_dialects(registry: &DialectRegistry) {
    unsafe {
        let mlir_sys_registry: mlir_sys::MlirDialectRegistry = registry.to_raw();
        stablehlo_sys::mlir_stablehlo_registerAllDialects(mlir_sys_registry.ptr);
    }
}

fn main() {
    let registry = DialectRegistry::new();
    // melior::utility::register_all_dialects(&registry);

    let context = Context::new();

    println!(
        "loaded: {} registered: {}",
        context.loaded_dialect_count(),
        context.registered_dialect_count()
    );
    // context.append_dialect_registry(&registry);
    stablehlo_register_all_dialects(&registry);
    // context.append_dialect_registry(&registry);
    context.append_dialect_registry(&registry);
    let _stablhlo_dialect = context.get_or_load_dialect("stablehlo");
    let _stablhlo_dialect = context.get_or_load_dialect("func");
    println!(
        "loaded: {} registered: {}",
        context.loaded_dialect_count(),
        context.registered_dialect_count()
    );

    let location = melior::ir::Location::unknown(&context);
    let module = melior::ir::Module::new(location);

    let scalar_type = melior::ir::Type::float32(&context);

    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "add"),
        TypeAttribute::new(
            FunctionType::new(&context, &[scalar_type, scalar_type], &[scalar_type]).into(),
        ),
        {
            let block = Block::new(&[(scalar_type, location), (scalar_type, location)]);

            let add_op: stablehlo::AddOp = stablehlo::add(
                &context,
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            );

            let sum = block.append_operation(add_op.into());

            let return_op =
                stablehlo::r#return(&context, &[sum.result(0).unwrap().into()], location);

            block.append_operation(return_op.into());

            let region = Region::new();
            region.append_block(block);
            region
        },
        &[],
        location,
    ));

    println!("{}", module.as_operation());

    assert!(module.as_operation().verify());
}
