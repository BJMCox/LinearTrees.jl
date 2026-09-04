"""
Six weighted sums over a row set. Fields are `V`, the coefficient type, so
the softmax case carries one sum per fitted logit.
"""
struct MomentSums{V}
    sw::V
    sx::V
    sxx::V
    sz::V
    sxz::V
    szz::V
end
