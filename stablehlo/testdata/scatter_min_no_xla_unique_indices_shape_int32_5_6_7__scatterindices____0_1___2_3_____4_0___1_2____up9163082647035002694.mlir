// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xi32>, tensor<5x2x2xi32>)
    %2 = call @expected() : () -> tensor<5x6x7xi32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i32>
      stablehlo.return %5 : tensor<i32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xi32>, tensor<2x2x2xi32>, tensor<5x2x2xi32>) -> tensor<5x6x7xi32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xi32>, tensor<5x6x7xi32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xi32>, tensor<5x2x2xi32>) {
    %0 = stablehlo.constant dense<"0x00000000FEFFFFFF00000000FEFFFFFF0200000000000000040000000300000000000000FFFFFFFFFCFFFFFF0200000001000000FFFFFFFFFBFFFFFFFFFFFFFF03000000FFFFFFFFFDFFFFFFFFFFFFFF04000000FEFFFFFFFEFFFFFF0000000001000000FCFFFFFF03000000FFFFFFFF0300000002000000FCFFFFFF0000000000000000000000000000000000000000000000000000000000000000000000000200000000000000000000000300000000000000FFFFFFFFFCFFFFFF00000000FFFFFFFFFFFFFFFFFFFFFFFF02000000FAFFFFFFFCFFFFFF00000000FFFFFFFFFBFFFFFF010000000100000000000000FDFFFFFF00000000FDFFFFFF0200000000000000FFFFFFFF000000000000000000000000FEFFFFFF0200000001000000FCFFFFFF02000000FDFFFFFF000000000200000005000000020000000000000003000000000000000000000000000000FEFFFFFFFEFFFFFFFCFFFFFF02000000FEFFFFFF01000000010000000000000006000000FFFFFFFF00000000FBFFFFFF030000000000000002000000FCFFFFFF020000000200000004000000FEFFFFFFFFFFFFFF0000000002000000000000000700000000000000000000000100000002000000FFFFFFFFFDFFFFFF00000000010000000000000002000000020000000500000001000000040000000100000003000000FFFFFFFFFFFFFFFF010000000000000000000000070000000400000004000000FEFFFFFF00000000FDFFFFFF04000000FEFFFFFF04000000FFFFFFFF0000000003000000FEFFFFFF02000000000000000400000001000000FCFFFFFF010000000000000000000000FEFFFFFFF6FFFFFF010000000500000000000000FEFFFFFF01000000FDFFFFFF0100000000000000FEFFFFFF010000000000000000000000000000000300000000000000000000000100000000000000050000000000000006000000FCFFFFFF0200000004000000020000000300000000000000FEFFFFFF00000000FEFFFFFFFEFFFFFF00000000FFFFFFFFFAFFFFFF0000000000000000FBFFFFFF00000000000000000000000001000000FCFFFFFF000000000000000000000000FDFFFFFF04000000000000000100000004000000FEFFFFFFFFFFFFFF04000000FFFFFFFF010000000100000000000000"> : tensor<5x6x7xi32>
    %1 = stablehlo.constant dense<[[[3, 4], [1, 0]], [[1, 0], [3, 0]], [[2, -2], [2, -2]], [[1, -6], [-5, -1]], [[0, 2], [-3, 2]]]> : tensor<5x2x2xi32>
    return %0, %1 : tensor<5x6x7xi32>, tensor<5x2x2xi32>
  }
  func.func private @expected() -> tensor<5x6x7xi32> {
    %0 = stablehlo.constant dense<"0x00000000FEFFFFFF00000000FEFFFFFF0200000000000000040000000300000000000000FFFFFFFFFCFFFFFF0200000001000000FFFFFFFFFBFFFFFFFFFFFFFF03000000FFFFFFFFFDFFFFFFFFFFFFFF04000000FEFFFFFFFEFFFFFF0000000001000000FCFFFFFF03000000FFFFFFFF0100000002000000FCFFFFFF0000000000000000000000000000000000000000000000000000000000000000000000000200000000000000000000000100000000000000FFFFFFFFFCFFFFFF00000000FFFFFFFFFFFFFFFFFFFFFFFF00000000FAFFFFFFFCFFFFFF00000000FFFFFFFFFBFFFFFF010000000100000000000000FDFFFFFF00000000FDFFFFFF0200000000000000FFFFFFFF000000000000000000000000FEFFFFFF0200000001000000FCFFFFFF02000000FDFFFFFF000000000200000005000000020000000000000003000000000000000000000000000000FEFFFFFFFEFFFFFFFCFFFFFF02000000FEFFFFFF01000000010000000000000006000000FEFFFFFF00000000FBFFFFFF030000000000000002000000FCFFFFFF02000000FEFFFFFF04000000FEFFFFFFFFFFFFFF0000000002000000000000000700000000000000000000000100000002000000FFFFFFFFFDFFFFFF00000000010000000000000002000000020000000500000001000000040000000100000003000000FFFFFFFFFFFFFFFF010000000000000000000000070000000400000004000000FEFFFFFF00000000FDFFFFFF04000000FEFFFFFF04000000FFFFFFFF0000000003000000FEFFFFFFFAFFFFFF000000000400000001000000FCFFFFFF010000000000000000000000FEFFFFFFF6FFFFFF01000000FBFFFFFF00000000FEFFFFFF01000000FDFFFFFF0100000000000000FEFFFFFF010000000000000000000000000000000300000000000000000000000000000000000000050000000000000006000000FCFFFFFF0200000004000000020000000300000000000000FEFFFFFF00000000FEFFFFFFFEFFFFFF00000000FFFFFFFFFAFFFFFF0000000000000000FBFFFFFF00000000000000000000000001000000FCFFFFFF00000000FDFFFFFF00000000FDFFFFFF04000000000000000100000004000000FEFFFFFFFFFFFFFF04000000FFFFFFFF010000000100000000000000"> : tensor<5x6x7xi32>
    return %0 : tensor<5x6x7xi32>
  }
}

