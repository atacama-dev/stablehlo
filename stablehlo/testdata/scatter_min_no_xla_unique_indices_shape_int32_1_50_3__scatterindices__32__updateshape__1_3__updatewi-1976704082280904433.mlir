// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi32>, tensor<1x3xi32>)
    %2 = call @expected() : () -> tensor<1x50x3xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi32>, tensor<1xi32>, tensor<1x3xi32>) -> tensor<1x50x3xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi32>, tensor<1x50x3xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi32>, tensor<1x3xi32>) {
    %0 = stablehlo.constant dense<"0xFCFFFFFF030000000000000000000000FEFFFFFF0000000002000000010000000300000002000000FEFFFFFF00000000FEFFFFFF0600000000000000000000000100000000000000FEFFFFFF010000000400000001000000FDFFFFFF01000000FFFFFFFF0100000002000000FEFFFFFF0000000002000000FDFFFFFF00000000FDFFFFFF010000000000000005000000FEFFFFFF00000000FCFFFFFF00000000FFFFFFFFFCFFFFFFFEFFFFFF01000000030000000000000000000000FDFFFFFFFDFFFFFF010000000300000000000000FCFFFFFFFCFFFFFF00000000020000000500000002000000FFFFFFFF0200000000000000FFFFFFFFFFFFFFFF04000000FDFFFFFF000000000600000002000000000000000100000000000000020000000000000000000000FFFFFFFF0100000002000000FEFFFFFF00000000FFFFFFFF00000000FBFFFFFF000000000000000001000000010000000100000000000000FDFFFFFF01000000000000000000000004000000020000000300000000000000FCFFFFFF02000000FFFFFFFFFFFFFFFF010000000000000002000000FFFFFFFF0200000001000000FDFFFFFF000000000600000002000000FFFFFFFFFCFFFFFF04000000010000000000000002000000010000000100000000000000FFFFFFFF000000000300000003000000FDFFFFFFFFFFFFFF00000000FEFFFFFFFDFFFFFF00000000FFFFFFFF050000000200000003000000040000000000000004000000FFFFFFFFFCFFFFFF00000000050000000400000001000000000000000000000003000000FBFFFFFFFBFFFFFF0000000000000000FEFFFFFF"> : tensor<1x50x3xi32>
    %1 = stablehlo.constant dense<[[-3, 0, 0]]> : tensor<1x3xi32>
    return %0, %1 : tensor<1x50x3xi32>, tensor<1x3xi32>
  }
  func.func private @expected() -> tensor<1x50x3xi32> {
    %0 = stablehlo.constant dense<"0xFCFFFFFF030000000000000000000000FEFFFFFF0000000002000000010000000300000002000000FEFFFFFF00000000FEFFFFFF0600000000000000000000000100000000000000FEFFFFFF010000000400000001000000FDFFFFFF01000000FFFFFFFF0100000002000000FEFFFFFF0000000002000000FDFFFFFF00000000FDFFFFFF010000000000000005000000FEFFFFFF00000000FCFFFFFF00000000FFFFFFFFFCFFFFFFFEFFFFFF01000000030000000000000000000000FDFFFFFFFDFFFFFF010000000300000000000000FCFFFFFFFCFFFFFF00000000020000000500000002000000FFFFFFFF0200000000000000FFFFFFFFFFFFFFFF04000000FDFFFFFF000000000600000002000000000000000100000000000000020000000000000000000000FFFFFFFF0100000002000000FEFFFFFF00000000FFFFFFFF00000000FBFFFFFF000000000000000001000000010000000100000000000000FDFFFFFF01000000000000000000000004000000020000000300000000000000FCFFFFFF00000000FFFFFFFFFFFFFFFF010000000000000002000000FFFFFFFF0200000001000000FDFFFFFF000000000600000002000000FFFFFFFFFCFFFFFF04000000010000000000000002000000010000000100000000000000FFFFFFFF000000000300000003000000FDFFFFFFFFFFFFFF00000000FEFFFFFFFDFFFFFF00000000FFFFFFFF050000000200000003000000040000000000000004000000FFFFFFFFFCFFFFFF00000000050000000400000001000000000000000000000003000000FBFFFFFFFBFFFFFF0000000000000000FEFFFFFF"> : tensor<1x50x3xi32>
    return %0 : tensor<1x50x3xi32>
  }
}

