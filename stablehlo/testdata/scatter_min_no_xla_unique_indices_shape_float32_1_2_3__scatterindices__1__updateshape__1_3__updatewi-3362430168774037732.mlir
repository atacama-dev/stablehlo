// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x2x3xf32>, tensor<1x3xf32>)
    %2 = call @expected() : () -> tensor<1x2x3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x2x3xf32>, tensor<1xi32>, tensor<1x3xf32>) -> tensor<1x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2x3xf32>, tensor<1x3xf32>) {
    %0 = stablehlo.constant dense<[[[-2.45538878, 0.131931543, -4.36057186], [5.52165604, 3.64848495, -1.30776644]]]> : tensor<1x2x3xf32>
    %1 = stablehlo.constant dense<[[-0.600486815, -6.23648548, -3.97061229]]> : tensor<1x3xf32>
    return %0, %1 : tensor<1x2x3xf32>, tensor<1x3xf32>
  }
  func.func private @expected() -> tensor<1x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-2.45538878, 0.131931543, -4.36057186], [-0.600486815, -6.23648548, -3.97061229]]]> : tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }
}

