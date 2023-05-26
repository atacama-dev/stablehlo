// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x40xbf16>, tensor<3x5x2xbf16>)
    %2 = call @expected() : () -> tensor<3x5x40xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>} : (tensor<3x5x40xbf16>, tensor<2x1xi32>, tensor<3x5x2xbf16>) -> tensor<3x5x40xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x40xbf16>, tensor<3x5x40xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x40xbf16>, tensor<3x5x2xbf16>) {
    %0 = stablehlo.constant dense<"0x9BC0A2C0FEBF0B40B1BFC2C0FEBFED3F0ABEE8C05EBF03C0AD400A4070C0AD3D90C0B7BF5A3F0EC060C05EBFFF40F1BE78C0103F73C043BF29C0D8C0C3BF1BC004C05C403AC028BF824001C01E403E4002BE0341AF406A4069C08A3EB4BE0BC0CD3DA440743F92BFC0C08A3E85C0873FDEBF5940A53FA240F83EC44011BFBB40903F5FC0C93F5340B5BF893F233D99C08E3E8640984060C01A408DC08E3E93BF0BC01FC0FFBE773ED040D34051C0F5C07CBF6A402E40134079C097C0A2C08D4058C0AB401DC05E408C3FA9BF593FD23FAE3FB440A9C00F403DC077BF86C0CC3FAD3EA7BF06C06CC0483F5740B740D0BF29408DBFB7C0603B273F15C0E13F61C0983F47404BBF73406D40AC3F9A3FA4BF2B408E40B83FAD4071C0AF408DBFB640194046BF6D40F6BF07C0A53F08C1963F494018C1A440BABF07409C3E49BE6EBDBC3E8840C140B7BFB7BFFA3FD53B943E773C57C006C00F3FA44012C02B4039407FC07EC077BE16BF45C001411540DB3E4140213FA7C07240303FFA40BCBE7D3F9EC0B6C0AA40B6BF483F55BC093F8D3F5740CEBE03BFD8C077C0ADBDCEBFA640D2408FBDA2BF4340BB4018C076C0354003BF384067C08940AE40FC3F5040684051BF4C4064C062C0CD3F25409C3D32BF36C0F5BE643F66BF6EBF3AC0A0C096BF26BF7140A3401E40B4BFEA3EA14010C016BFBEBF5F40DFC020BF27BFAE3FD03FBE3F1DBFCD4027C098C08BBE90BF1B4097BFE1BED73FB540FABE334063BD7EBF5EBE62C0EFBFD93E0E3F713FCE3FD240DBC080409C40BFBE49C0EB3F35407BC0A6C07C3E71BF3740153EF5BF2AC0A6BFF5BFE3408BC0BE40F73FCDBF97C0DFBF9E3F5F3F0C3F01C08940ADC01E3FDABE93C0CFC0CC3EE83F70C0F13F7CC003C028C0B4C08C409CC0E740E4BE2340B9BF97C085BF9840333F13C09140D3BE3CC0483F08C0E63F38C0B9BF8DC038402540B53F20C009C0B740A93F103F8CC07540B2BF6DC0C9BFBE3F70C0B540A6400EBD9F40153FC03FA5BDC73FD63F07C078C08F3FDE3F47BFDABF00C09EBF8540AF3E9A40C63DE53FD4408F402CC00B40ABBF393EA6BFE63EC13C0841B23F36BFF9BFBD3FA13FB83FA3BF594014BF2040593F55C0E53F1FBF333F70C07B4038BF1340A2BFC5BF1540C93F87C0783F00BF3040A2BF8840DF3E823F814015BF83C0673FE93F70BF71409DC018C0BC3F00BF8C401F3FB6BFB03F2740EE3EFF3F7CC010C09E3F1E40783F0F401540B1BF8240BB3D55BE04409DC0B53DADBFB840CCBF11C0AF3FC0BFB9400FC0B940BABF06BFC440A43FAA3F4DC086C016C09140A4BFA63F67409ABFB2BFA1BFE93EC03E503F54C086BFA54008C0A8404B3F394009402AC08FBE35BE42C1B4BFBB3E08C0F7C08BC0FB3E4BC01EC02040DEBE8D40843F7A406D40C4BB28C0483E92BF9D405AC0D5BF6DC0DC3E95BF093DAABF504003C04140BB40E03F89BF09C081BFB24036C0E1BF03400C40FF3DCCC0744012BFE4BF073F0B415C3F76BE0340C93F4140A93EB040E3BF42403F3F14C022401F4064C0E53D1BC0454020BFE240DEBFC5BBD73F45C029C08CBF9CBF2BBE723F07C00940F0BBC0C0EE3F6DC0DC3C55BF09C094C0B03E1AC02340603F09BF02C010C0023FFABFDFBFCEBF484024401440464013C0B8BF41C080C0F4BF89BFBABFB0C095BF0240"> : tensor<3x5x40xbf16>
    %1 = stablehlo.constant dense<[[[3.250000e+00, 1.375000e+00], [-1.890630e+00, -1.390630e+00], [4.625000e+00, -5.375000e+00], [-3.945310e-01, -4.718750e+00], [-9.609370e-01, 1.585940e+00]], [[6.298830e-02, 1.265630e+00], [-4.468750e+00, -2.216800e-01], [-9.125000e+00, -2.921880e+00], [-1.351560e+00, -2.953130e+00], [4.656250e+00, 1.625000e+00]], [[-5.187500e+00, -3.085940e-01], [-2.500000e+00, -2.875000e+00], [2.125000e+00, -2.187500e-01], [2.453130e+00, -3.859380e+00], [2.843750e+00, 2.640630e+00]]]> : tensor<3x5x2xbf16>
    return %0, %1 : tensor<3x5x40xbf16>, tensor<3x5x2xbf16>
  }
  func.func private @expected() -> tensor<3x5x40xbf16> {
    %0 = stablehlo.constant dense<"0x9BC0E0BEFEBF0B40B1BFC2C0FEBFED3F0ABEE8C05EBF03C0AD400A4070C0AD3D90C0B7BF5A3F0EC060C05EBFFF40F1BE78C0103F73C043BF29C0D8C0C3BF1BC004C05C403AC028BF824001C01E403E4002BE9E40AF406A4069C08A3EB4BE0BC0CD3DA440743F92BFC0C08A3E85C0873FDEBF5940A53FA240F83EC44011BFBB40903F5FC0C93F5340B5BF893F233D99C08E3E8640984060C01A408DC08E3E93BF0BC04FC0FFBE773ED040D34051C0F5C07CBF6A402E40134079C097C0A2C08D4058C0AB401DC05E408C3FA9BF593FD23FAE3FB440A9C00F403DC077BF86C0CC3FAD3EA7BF06C06CC0483F5740B740D0BF2940C7C0B7C0603B273F15C0E13F61C0983F47404BBF73406D40AC3F9A3FA4BF2B408E40B83FAD4071C0AF408DBFB640194046BF6D40F6BF07C0A53F08C1963F494018C1A440BABF07409C3E49BE6EBDBC3E9C40C140B7BFB7BFFA3FD53B943E773C57C006C00F3FA44012C02B4039407FC07EC077BE16BF45C001411540DB3E4140213FA7C07240303FFA40BCBE7D3F9EC0B6C0AA40B6BF483F55BC093F8D3F57406D3F03BFD8C077C0ADBDCEBFA640D2408FBDA2BF4340BB4018C076C0354003BF384067C08940AE40FC3F5040684051BF4C4064C062C0CD3F25409C3D32BF36C0F5BE643F66BF6EBF3AC0A0C096BF26BF6DBFA3401E40B4BFEA3EA14010C016BFBEBF5F40DFC020BF27BFAE3FD03FBE3F1DBFCD4027C098C08BBE90BF1B4097BFE1BED73FB540FABE334063BD7EBF5EBE62C0EFBFD93E0E3F713FCE3FD240DBC001C19C40BFBE49C0EB3F35407BC0A6C07C3E71BF3740153EF5BF2AC0A6BFF5BFE3408BC0BE40F73FCDBF97C0DFBF9E3F5F3F0C3F01C08940ADC01E3FDABE93C0CFC0CC3EE83F70C0F13F7CC003C028C01FC18C409CC0E740E4BE2340B9BF97C085BF9840333F13C09140D3BE3CC0483F08C0E63F38C0B9BF8DC038402540B53F20C009C0B740A93F103F8CC07540B2BF6DC0C9BFBE3F70C0B540A6400EBD9F40DC40C03FA5BDC73FD63F07C078C08F3FDE3F47BFDABF00C09EBF8540AF3E9A40C63DE53FD4408F402CC00B40ABBF393EA6BFE63EC13C0841B23F36BFF9BFBD3FA13FB83FA3BF594014BF2040593F55C06EC01FBF333F70C07B4038BF1340A2BFC5BF1540C93F87C0783F00BF3040A2BF8840DF3E823F814015BF83C0673FE93F70BF71409DC018C0BC3F00BF8C401F3FB6BFB03F2740EE3EFF3F7CC010C09E3F3AC0783F0F401540B1BF8240BB3D55BE04409DC0B53DADBFB840CCBF11C0AF3FC0BFB9400FC0B940BABF06BFC440A43FAA3F4DC086C016C09140A4BFA63F67409ABFB2BFA1BFE93EC03E503F54C086BFE24008C0A8404B3F394009402AC08FBE35BE42C1B4BFBB3E08C0F7C08BC0FB3E4BC01EC02040DEBE8D40843F7A406D40C4BB28C0483E92BF9D405AC0D5BF6DC0DC3E95BF093DAABF504003C04140BB40A83E89BF09C081BFB24036C0E1BF03400C40FF3DCCC0744012BFE4BF073F0B415C3F76BE0340C93F4140A93EB040E3BF42403F3F14C022401F4064C0E53D1BC0454020BFE240DEBFC5BBD73F45C029C08C409CBF2BBE723F07C00940F0BBC0C0EE3F6DC0DC3C55BF09C094C0B03E1AC02340603F09BF02C010C0023FFABFDFBFCEBF484024401440464013C0B8BF41C080C0F4BF89BFBABFB0C095BF0240"> : tensor<3x5x40xbf16>
    return %0 : tensor<3x5x40xbf16>
  }
}

