// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui16>, tensor<2x7xui16>)
    %2 = call @expected() : () -> tensor<5x6x7xui16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true} : (tensor<5x6x7xui16>, tensor<2x2xi32>, tensor<2x7xui16>) -> tensor<5x6x7xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui16>, tensor<5x6x7xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui16>, tensor<2x7xui16>) {
    %0 = stablehlo.constant dense<"0x0600030004000300010004000400000003000200000000000000050003000200030000000400000001000100020002000300020000000200030007000500000000000000010008000100000003000500010002000000040003000100040001000300030003000100040000000100000001000A00000006000400020001000200000001000000000003000400000001000A0004000000000001000100020001000100050005000100000000000100000001000200040000000200000001000000040002000300060000000200000001000400030001000300010001000200000000000000050002000600030002000200000001000300000002000000040001000000000004000500030001000500000000000300020001000300000002000100000002000100020000000300000001000300000004000500090005000300010001000500020007000200010003000200060002000100010002000400000006000000040000000100020004000000020003000200000000000200000001000100020001000300020000000400020000000100010003000000050001000400030004000000"> : tensor<5x6x7xui16>
    %1 = stablehlo.constant dense<[[2, 5, 1, 2, 2, 2, 1], [0, 3, 0, 2, 2, 0, 1]]> : tensor<2x7xui16>
    return %0, %1 : tensor<5x6x7xui16>, tensor<2x7xui16>
  }
  func.func private @expected() -> tensor<5x6x7xui16> {
    %0 = stablehlo.constant dense<"0x0600030004000300010004000400020005000200020002000200050003000200030000000400000001000100020002000300020000000200030007000500000000000000010008000100000003000500010002000000040003000100040001000300030003000100040000000100000001000A00000006000400020001000200000001000000000003000400000001000A0004000000000001000100020001000100050005000100000000000100000001000200040000000200000001000000040002000300060000000200000001000400030003000300020002000200010000000000050002000600030002000200000001000300000002000000040001000000000004000500030001000500000000000300020001000300000002000100000002000100020000000300000001000300000004000500090005000300010001000500020007000200010003000200060002000100010002000400000006000000040000000100020004000000020003000200000000000200000001000100020001000300020000000400020000000100010003000000050001000400030004000000"> : tensor<5x6x7xui16>
    return %0 : tensor<5x6x7xui16>
  }
}

