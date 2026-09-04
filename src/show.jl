# Base.show methods for LinearTree.
using AbstractTrees

"A named view of a tree for `AbstractTrees` traversal and printing."
struct TreeView{Tr<:LinearTree}
    tree::Tr
    names::Vector{Symbol}
end
TreeView(tree::LinearTree) = TreeView(tree, [Symbol("x", j) for j in 1:tree.nfeatures])

"A handle to one node plus the branch it was reached by, for `AbstractTrees` traversal."
struct NodeHandle{Tr}
    view::TreeView{Tr}
    index::Int32
    branch::Symbol           # :root, :left, :right
end

AbstractTrees.children(h::NodeHandle) = begin
    n = h.view.tree.nodes[h.index]
    isleaf(n) ? () : n.model == LIN ? (NodeHandle(h.view, n.left, :left),) :
        (NodeHandle(h.view, n.left, :left), NodeHandle(h.view, n.right, :right))
end
AbstractTrees.nodevalue(h::NodeHandle) = h.view.tree.nodes[h.index]
AbstractTrees.children(v::TreeView) = (NodeHandle(v, Int32(1), :root),)
AbstractTrees.nodevalue(v::TreeView) = v.tree

fmtnum(x::Real) = replace(string(round(x; sigdigits = 3)), "-" => "−")
fmtnum(x::SVector) = string("[", join(map(fmtnum, x), ", "), "]")
piece(a, b, name) = a == 0 ? fmtnum(b) : string(fmtnum(a), "·", name, b < 0 ? " − " : " + ", fmtnum(abs(b)))

function AbstractTrees.printnode(io::IO, h::NodeHandle)
    n = AbstractTrees.nodevalue(h)
    if isleaf(n)
        print(io, "leaf  intercept = ", fmtnum(n.lintercept))
        return
    end
    name = String(h.view.names[n.feature])
    kind = lowercase(string(n.model))
    if n.model == LIN
        print(io, piece(n.lcoef, n.lintercept, name), "  [lin]")
    elseif iscategorical(n)
        print(io, name, " ∈ left set  [", kind, "]")
    else
        print(io, name, " ≤ ", fmtnum(n.threshold), "  [", kind, "]")
        print(io, "\n  ├ ", piece(n.lcoef, n.lintercept, name), "\n  └ ", piece(n.rcoef, n.rintercept, name))
    end
end
AbstractTrees.printnode(io::IO, v::TreeView) = print(io, "LinearTree with ", length(v.tree.nodes), " nodes")

function Base.show(io::IO, ::MIME"text/plain", t::LinearTree{T,V,L}) where {T,V,L}
    print(io, "LinearTree{", T, ", ", V, ", ", nameof(L), "} with ", length(t.nodes), " nodes, ",
        count(isleaf, t.nodes), " leaves, ", t.nfeatures, " features")
    if length(t.nodes) > 1
        println(io)
        print_tree(io, TreeView(t); maxdepth = 3)
    end
end
