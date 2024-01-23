use melior::ir::{
    attribute::{StringAttribute, TypeAttribute},
    r#type::FunctionType,
    *,
};
use melior::{dialect::func, dialect::DialectRegistry, Context};
// use std::{env, fmt::Display, path::Path, process::Command, str};
// use tblgen::{record::Record, RecordKeeper, TableGenParser};

melior::dialect! {
    name: "stablehlo",
    table_gen: r#"include "stablehlo/dialect/StablehloOps.td""#,
    include_dirs: ["/Users/ilias/sources/stablehlo/"] ,
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
    melior::utility::register_all_dialects(&registry);

    let context = Context::new();

    println!("loaded: {}", context.loaded_dialect_count());

    context.append_dialect_registry(&registry);
    stablehlo_register_all_dialects(&registry);

    let _stablhlo_dialect = context.get_or_load_dialect("stablehlo");
    println!("loaded: {}", context.loaded_dialect_count());

    let location = melior::ir::Location::unknown(&context);
    let module = melior::ir::Module::new(location);

    let index_type = melior::ir::Type::index(&context);

    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "add"),
        TypeAttribute::new(
            FunctionType::new(&context, &[index_type, index_type], &[index_type]).into(),
        ),
        {
            let block = Block::new(&[(index_type, location), (index_type, location)]);

            let add_op: stablehlo::AddOp = stablehlo::add(
                &context,
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            );

            let sum = block.append_operation(add_op.into());

            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);
            region
        },
        &[],
        location,
    ));

    //println!("{}", module.as_operation());

    // assert!(module.as_operation().verify());
}
