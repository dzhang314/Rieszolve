module SphericalSymmetries

using StaticArrays: SA

################################################################################


export chiral_icosahedral_group, full_icosahedral_group


@inline function chiral_icosahedral_group(::Type{T}) where {T}
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _five = _four + _one
    _sqrt_five = sqrt(_five)
    half = inv(_two)
    hphi = (_one + _sqrt_five) / _four # half of the golden ratio
    hpsi = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
    return SA{T}[
        +_one _zero _zero; _zero +_one _zero; _zero _zero +_one;;;
        +_one _zero _zero; _zero -_one _zero; _zero _zero -_one;;;
        -_one _zero _zero; _zero +_one _zero; _zero _zero -_one;;;
        -_one _zero _zero; _zero -_one _zero; _zero _zero +_one;;;
        _zero +_one _zero; _zero _zero +_one; +_one _zero _zero;;;
        _zero +_one _zero; _zero _zero -_one; -_one _zero _zero;;;
        _zero -_one _zero; _zero _zero +_one; -_one _zero _zero;;;
        _zero -_one _zero; _zero _zero -_one; +_one _zero _zero;;;
        _zero _zero +_one; +_one _zero _zero; _zero +_one _zero;;;
        _zero _zero +_one; -_one _zero _zero; _zero -_one _zero;;;
        _zero _zero -_one; +_one _zero _zero; _zero -_one _zero;;;
        _zero _zero -_one; -_one _zero _zero; _zero +_one _zero;;;
        +half +hphi +hpsi; +hphi +hpsi +half; -hpsi -half -hphi;;;
        +half +hphi +hpsi; -hphi -hpsi -half; +hpsi +half +hphi;;;
        +half +hphi -hpsi; +hphi +hpsi -half; +hpsi +half -hphi;;;
        +half +hphi -hpsi; -hphi -hpsi +half; -hpsi -half +hphi;;;
        +half -hphi +hpsi; +hphi -hpsi +half; +hpsi -half +hphi;;;
        +half -hphi +hpsi; -hphi +hpsi -half; -hpsi +half -hphi;;;
        +half -hphi -hpsi; +hphi -hpsi -half; -hpsi +half +hphi;;;
        +half -hphi -hpsi; -hphi +hpsi +half; +hpsi -half -hphi;;;
        -half +hphi +hpsi; +hphi -hpsi -half; +hpsi -half -hphi;;;
        -half +hphi +hpsi; -hphi +hpsi +half; -hpsi +half +hphi;;;
        -half +hphi -hpsi; +hphi -hpsi +half; -hpsi +half -hphi;;;
        -half +hphi -hpsi; -hphi +hpsi -half; +hpsi -half +hphi;;;
        -half -hphi +hpsi; +hphi +hpsi -half; -hpsi -half +hphi;;;
        -half -hphi +hpsi; -hphi -hpsi +half; +hpsi +half -hphi;;;
        -half -hphi -hpsi; +hphi +hpsi +half; +hpsi +half +hphi;;;
        -half -hphi -hpsi; -hphi -hpsi -half; -hpsi -half -hphi;;;
        +hphi +hpsi +half; +hpsi +half +hphi; -half -hphi -hpsi;;;
        +hphi +hpsi +half; -hpsi -half -hphi; +half +hphi +hpsi;;;
        +hphi +hpsi -half; +hpsi +half -hphi; +half +hphi -hpsi;;;
        +hphi +hpsi -half; -hpsi -half +hphi; -half -hphi +hpsi;;;
        +hphi -hpsi +half; +hpsi -half +hphi; +half -hphi +hpsi;;;
        +hphi -hpsi +half; -hpsi +half -hphi; -half +hphi -hpsi;;;
        +hphi -hpsi -half; +hpsi -half -hphi; -half +hphi +hpsi;;;
        +hphi -hpsi -half; -hpsi +half +hphi; +half -hphi -hpsi;;;
        -hphi +hpsi +half; +hpsi -half -hphi; +half -hphi -hpsi;;;
        -hphi +hpsi +half; -hpsi +half +hphi; -half +hphi +hpsi;;;
        -hphi +hpsi -half; +hpsi -half +hphi; -half +hphi -hpsi;;;
        -hphi +hpsi -half; -hpsi +half -hphi; +half -hphi +hpsi;;;
        -hphi -hpsi +half; +hpsi +half -hphi; -half -hphi +hpsi;;;
        -hphi -hpsi +half; -hpsi -half +hphi; +half +hphi -hpsi;;;
        -hphi -hpsi -half; +hpsi +half +hphi; +half +hphi +hpsi;;;
        -hphi -hpsi -half; -hpsi -half -hphi; -half -hphi -hpsi;;;
        +hpsi +half +hphi; +half +hphi +hpsi; -hphi -hpsi -half;;;
        +hpsi +half +hphi; -half -hphi -hpsi; +hphi +hpsi +half;;;
        +hpsi +half -hphi; +half +hphi -hpsi; +hphi +hpsi -half;;;
        +hpsi +half -hphi; -half -hphi +hpsi; -hphi -hpsi +half;;;
        +hpsi -half +hphi; +half -hphi +hpsi; +hphi -hpsi +half;;;
        +hpsi -half +hphi; -half +hphi -hpsi; -hphi +hpsi -half;;;
        +hpsi -half -hphi; +half -hphi -hpsi; -hphi +hpsi +half;;;
        +hpsi -half -hphi; -half +hphi +hpsi; +hphi -hpsi -half;;;
        -hpsi +half +hphi; +half -hphi -hpsi; +hphi -hpsi -half;;;
        -hpsi +half +hphi; -half +hphi +hpsi; -hphi +hpsi +half;;;
        -hpsi +half -hphi; +half -hphi +hpsi; -hphi +hpsi -half;;;
        -hpsi +half -hphi; -half +hphi -hpsi; +hphi -hpsi +half;;;
        -hpsi -half +hphi; +half +hphi -hpsi; -hphi -hpsi +half;;;
        -hpsi -half +hphi; -half -hphi +hpsi; +hphi +hpsi -half;;;
        -hpsi -half -hphi; +half +hphi +hpsi; +hphi +hpsi +half;;;
        -hpsi -half -hphi; -half -hphi -hpsi; -hphi -hpsi -half
    ]
end


@inline function full_icosahedral_group(::Type{T}) where {T}
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _five = _four + _one
    _sqrt_five = sqrt(_five)
    half = inv(_two)
    hphi = (_one + _sqrt_five) / _four # half of the golden ratio
    hpsi = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
    return SA{T}[
        +_one _zero _zero; _zero +_one _zero; _zero _zero +_one;;;
        +_one _zero _zero; _zero +_one _zero; _zero _zero -_one;;;
        +_one _zero _zero; _zero -_one _zero; _zero _zero +_one;;;
        +_one _zero _zero; _zero -_one _zero; _zero _zero -_one;;;
        -_one _zero _zero; _zero +_one _zero; _zero _zero +_one;;;
        -_one _zero _zero; _zero +_one _zero; _zero _zero -_one;;;
        -_one _zero _zero; _zero -_one _zero; _zero _zero +_one;;;
        -_one _zero _zero; _zero -_one _zero; _zero _zero -_one;;;
        _zero +_one _zero; _zero _zero +_one; +_one _zero _zero;;;
        _zero +_one _zero; _zero _zero +_one; -_one _zero _zero;;;
        _zero +_one _zero; _zero _zero -_one; +_one _zero _zero;;;
        _zero +_one _zero; _zero _zero -_one; -_one _zero _zero;;;
        _zero -_one _zero; _zero _zero +_one; +_one _zero _zero;;;
        _zero -_one _zero; _zero _zero +_one; -_one _zero _zero;;;
        _zero -_one _zero; _zero _zero -_one; +_one _zero _zero;;;
        _zero -_one _zero; _zero _zero -_one; -_one _zero _zero;;;
        _zero _zero +_one; +_one _zero _zero; _zero +_one _zero;;;
        _zero _zero +_one; +_one _zero _zero; _zero -_one _zero;;;
        _zero _zero +_one; -_one _zero _zero; _zero +_one _zero;;;
        _zero _zero +_one; -_one _zero _zero; _zero -_one _zero;;;
        _zero _zero -_one; +_one _zero _zero; _zero +_one _zero;;;
        _zero _zero -_one; +_one _zero _zero; _zero -_one _zero;;;
        _zero _zero -_one; -_one _zero _zero; _zero +_one _zero;;;
        _zero _zero -_one; -_one _zero _zero; _zero -_one _zero;;;
        +half +hphi +hpsi; +hphi +hpsi +half; +hpsi +half +hphi;;;
        +half +hphi +hpsi; +hphi +hpsi +half; -hpsi -half -hphi;;;
        +half +hphi +hpsi; -hphi -hpsi -half; +hpsi +half +hphi;;;
        +half +hphi +hpsi; -hphi -hpsi -half; -hpsi -half -hphi;;;
        +half +hphi -hpsi; +hphi +hpsi -half; +hpsi +half -hphi;;;
        +half +hphi -hpsi; +hphi +hpsi -half; -hpsi -half +hphi;;;
        +half +hphi -hpsi; -hphi -hpsi +half; +hpsi +half -hphi;;;
        +half +hphi -hpsi; -hphi -hpsi +half; -hpsi -half +hphi;;;
        +half -hphi +hpsi; +hphi -hpsi +half; +hpsi -half +hphi;;;
        +half -hphi +hpsi; +hphi -hpsi +half; -hpsi +half -hphi;;;
        +half -hphi +hpsi; -hphi +hpsi -half; +hpsi -half +hphi;;;
        +half -hphi +hpsi; -hphi +hpsi -half; -hpsi +half -hphi;;;
        +half -hphi -hpsi; +hphi -hpsi -half; +hpsi -half -hphi;;;
        +half -hphi -hpsi; +hphi -hpsi -half; -hpsi +half +hphi;;;
        +half -hphi -hpsi; -hphi +hpsi +half; +hpsi -half -hphi;;;
        +half -hphi -hpsi; -hphi +hpsi +half; -hpsi +half +hphi;;;
        -half +hphi +hpsi; +hphi -hpsi -half; +hpsi -half -hphi;;;
        -half +hphi +hpsi; +hphi -hpsi -half; -hpsi +half +hphi;;;
        -half +hphi +hpsi; -hphi +hpsi +half; +hpsi -half -hphi;;;
        -half +hphi +hpsi; -hphi +hpsi +half; -hpsi +half +hphi;;;
        -half +hphi -hpsi; +hphi -hpsi +half; +hpsi -half +hphi;;;
        -half +hphi -hpsi; +hphi -hpsi +half; -hpsi +half -hphi;;;
        -half +hphi -hpsi; -hphi +hpsi -half; +hpsi -half +hphi;;;
        -half +hphi -hpsi; -hphi +hpsi -half; -hpsi +half -hphi;;;
        -half -hphi +hpsi; +hphi +hpsi -half; +hpsi +half -hphi;;;
        -half -hphi +hpsi; +hphi +hpsi -half; -hpsi -half +hphi;;;
        -half -hphi +hpsi; -hphi -hpsi +half; +hpsi +half -hphi;;;
        -half -hphi +hpsi; -hphi -hpsi +half; -hpsi -half +hphi;;;
        -half -hphi -hpsi; +hphi +hpsi +half; +hpsi +half +hphi;;;
        -half -hphi -hpsi; +hphi +hpsi +half; -hpsi -half -hphi;;;
        -half -hphi -hpsi; -hphi -hpsi -half; +hpsi +half +hphi;;;
        -half -hphi -hpsi; -hphi -hpsi -half; -hpsi -half -hphi;;;
        +hphi +hpsi +half; +hpsi +half +hphi; +half +hphi +hpsi;;;
        +hphi +hpsi +half; +hpsi +half +hphi; -half -hphi -hpsi;;;
        +hphi +hpsi +half; -hpsi -half -hphi; +half +hphi +hpsi;;;
        +hphi +hpsi +half; -hpsi -half -hphi; -half -hphi -hpsi;;;
        +hphi +hpsi -half; +hpsi +half -hphi; +half +hphi -hpsi;;;
        +hphi +hpsi -half; +hpsi +half -hphi; -half -hphi +hpsi;;;
        +hphi +hpsi -half; -hpsi -half +hphi; +half +hphi -hpsi;;;
        +hphi +hpsi -half; -hpsi -half +hphi; -half -hphi +hpsi;;;
        +hphi -hpsi +half; +hpsi -half +hphi; +half -hphi +hpsi;;;
        +hphi -hpsi +half; +hpsi -half +hphi; -half +hphi -hpsi;;;
        +hphi -hpsi +half; -hpsi +half -hphi; +half -hphi +hpsi;;;
        +hphi -hpsi +half; -hpsi +half -hphi; -half +hphi -hpsi;;;
        +hphi -hpsi -half; +hpsi -half -hphi; +half -hphi -hpsi;;;
        +hphi -hpsi -half; +hpsi -half -hphi; -half +hphi +hpsi;;;
        +hphi -hpsi -half; -hpsi +half +hphi; +half -hphi -hpsi;;;
        +hphi -hpsi -half; -hpsi +half +hphi; -half +hphi +hpsi;;;
        -hphi +hpsi +half; +hpsi -half -hphi; +half -hphi -hpsi;;;
        -hphi +hpsi +half; +hpsi -half -hphi; -half +hphi +hpsi;;;
        -hphi +hpsi +half; -hpsi +half +hphi; +half -hphi -hpsi;;;
        -hphi +hpsi +half; -hpsi +half +hphi; -half +hphi +hpsi;;;
        -hphi +hpsi -half; +hpsi -half +hphi; +half -hphi +hpsi;;;
        -hphi +hpsi -half; +hpsi -half +hphi; -half +hphi -hpsi;;;
        -hphi +hpsi -half; -hpsi +half -hphi; +half -hphi +hpsi;;;
        -hphi +hpsi -half; -hpsi +half -hphi; -half +hphi -hpsi;;;
        -hphi -hpsi +half; +hpsi +half -hphi; +half +hphi -hpsi;;;
        -hphi -hpsi +half; +hpsi +half -hphi; -half -hphi +hpsi;;;
        -hphi -hpsi +half; -hpsi -half +hphi; +half +hphi -hpsi;;;
        -hphi -hpsi +half; -hpsi -half +hphi; -half -hphi +hpsi;;;
        -hphi -hpsi -half; +hpsi +half +hphi; +half +hphi +hpsi;;;
        -hphi -hpsi -half; +hpsi +half +hphi; -half -hphi -hpsi;;;
        -hphi -hpsi -half; -hpsi -half -hphi; +half +hphi +hpsi;;;
        -hphi -hpsi -half; -hpsi -half -hphi; -half -hphi -hpsi;;;
        +hpsi +half +hphi; +half +hphi +hpsi; +hphi +hpsi +half;;;
        +hpsi +half +hphi; +half +hphi +hpsi; -hphi -hpsi -half;;;
        +hpsi +half +hphi; -half -hphi -hpsi; +hphi +hpsi +half;;;
        +hpsi +half +hphi; -half -hphi -hpsi; -hphi -hpsi -half;;;
        +hpsi +half -hphi; +half +hphi -hpsi; +hphi +hpsi -half;;;
        +hpsi +half -hphi; +half +hphi -hpsi; -hphi -hpsi +half;;;
        +hpsi +half -hphi; -half -hphi +hpsi; +hphi +hpsi -half;;;
        +hpsi +half -hphi; -half -hphi +hpsi; -hphi -hpsi +half;;;
        +hpsi -half +hphi; +half -hphi +hpsi; +hphi -hpsi +half;;;
        +hpsi -half +hphi; +half -hphi +hpsi; -hphi +hpsi -half;;;
        +hpsi -half +hphi; -half +hphi -hpsi; +hphi -hpsi +half;;;
        +hpsi -half +hphi; -half +hphi -hpsi; -hphi +hpsi -half;;;
        +hpsi -half -hphi; +half -hphi -hpsi; +hphi -hpsi -half;;;
        +hpsi -half -hphi; +half -hphi -hpsi; -hphi +hpsi +half;;;
        +hpsi -half -hphi; -half +hphi +hpsi; +hphi -hpsi -half;;;
        +hpsi -half -hphi; -half +hphi +hpsi; -hphi +hpsi +half;;;
        -hpsi +half +hphi; +half -hphi -hpsi; +hphi -hpsi -half;;;
        -hpsi +half +hphi; +half -hphi -hpsi; -hphi +hpsi +half;;;
        -hpsi +half +hphi; -half +hphi +hpsi; +hphi -hpsi -half;;;
        -hpsi +half +hphi; -half +hphi +hpsi; -hphi +hpsi +half;;;
        -hpsi +half -hphi; +half -hphi +hpsi; +hphi -hpsi +half;;;
        -hpsi +half -hphi; +half -hphi +hpsi; -hphi +hpsi -half;;;
        -hpsi +half -hphi; -half +hphi -hpsi; +hphi -hpsi +half;;;
        -hpsi +half -hphi; -half +hphi -hpsi; -hphi +hpsi -half;;;
        -hpsi -half +hphi; +half +hphi -hpsi; +hphi +hpsi -half;;;
        -hpsi -half +hphi; +half +hphi -hpsi; -hphi -hpsi +half;;;
        -hpsi -half +hphi; -half -hphi +hpsi; +hphi +hpsi -half;;;
        -hpsi -half +hphi; -half -hphi +hpsi; -hphi -hpsi +half;;;
        -hpsi -half -hphi; +half +hphi +hpsi; +hphi +hpsi +half;;;
        -hpsi -half -hphi; +half +hphi +hpsi; -hphi -hpsi -half;;;
        -hpsi -half -hphi; -half -hphi -hpsi; +hphi +hpsi +half;;;
        -hpsi -half -hphi; -half -hphi -hpsi; -hphi -hpsi -half
    ]
end


################################################################################

end # module SphericalSymmetries
