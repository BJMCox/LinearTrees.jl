"""
Per-node model kind. `LIN` nodes have one child and add no depth.
"""
@enum ModelKind::UInt8 CON LIN PCON BLIN PLIN

"""
One node of a linear model tree. Coefficients have type `V`, which is the
feature type `T` for scalar-score losses and `SVector{K-1,T}` for softmax.
Categorical split nodes keep a packed left-level mask in the tree's mask pool
at words `catstart:catstart+catwords-1`. Numeric nodes have `catwords == 0`.
"""
struct Node{T,V}
    feature::Int32
    threshold::T
    left::Int32
    right::Int32
    lcoef::V
    lintercept::V
    rcoef::V
    rintercept::V
    xmin::T
    xmax::T
    cover::T
    xmean::T
    catstart::Int32
    catwords::Int32
    model::ModelKind
end

function Node{T,V}(; feature = 0, threshold = zero(T), left = 0, right = 0,
        lcoef = zero(V), lintercept = zero(V), rcoef = zero(V), rintercept = zero(V),
        xmin = T(-Inf), xmax = T(Inf), cover = zero(T), xmean = zero(T),
        catstart = 0, catwords = 0, model = CON) where {T,V}
    Node{T,V}(feature, threshold, left, right, lcoef, lintercept, rcoef, rintercept,
        xmin, xmax, cover, xmean, catstart, catwords, model)
end

isleaf(n::Node) = n.feature == 0
iscategorical(n::Node) = n.catwords > 0

"Copy `n` with new children, coefficients, or intercepts."
Node{T,V}(n::Node{T,V}; left = n.left, right = n.right, lcoef = n.lcoef, rcoef = n.rcoef,
        lintercept = n.lintercept, rintercept = n.rintercept) where {T,V} =
    Node{T,V}(n.feature, n.threshold, left, right, lcoef, lintercept, rcoef, rintercept,
        n.xmin, n.xmax, n.cover, n.xmean, n.catstart, n.catwords, n.model)

"Row-count gate: below this, threaded work is not worth the task overhead."
const PARALLEL_MIN_ROWS = 2^14

"""
Fitted tree. `lo`, `hi` are the stored first-truncation bounds on the score
scale. `base` is the SHAP empty-coalition value. `truncate == false` disables
both clamps at prediction.
"""
struct LinearTree{T,V,L<:Loss}
    nodes::Vector{Node{T,V}}
    catmasks::Vector{UInt64}
    loss::L
    lo::V
    hi::V
    base::V
    nfeatures::Int
    truncate::Bool
end

"""
True when level code `code` (1-based) is in the node's left set.
Codes beyond the mask route right, which is the unseen-level policy.
"""
function category_is_left(masks::Vector{UInt64}, n::Node, code::Integer)
    code < 1 && return false
    word = (code - 1) >> 6
    word >= n.catwords && return false
    bit = (code - 1) & 63
    return (masks[n.catstart + word] >> bit) & 0x1 == 0x1
end

category_is_left(tree::LinearTree, n::Node, code::Integer) = category_is_left(tree.catmasks, n, code)
