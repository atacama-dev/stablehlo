//! Sparse tensor passes.

melior_macro::passes!(
    "SparseTensor",
    [
        mlirCreateSparseTensorPreSparsificationRewrite,
        mlirCreateSparseTensorSparseBufferRewrite,
        mlirCreateSparseTensorSparseTensorCodegen,
        mlirCreateSparseTensorSparseTensorConversionPass,
        mlirCreateSparseTensorSparseVectorization,
        mlirCreateSparseTensorSparsificationPass,
        mlirCreateSparseTensorStorageSpecifierToLLVM,
    ]
);
