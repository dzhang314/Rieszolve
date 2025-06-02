module SphericalSymmetries

using StaticArrays: SA

################################################################################


export chiral_icosahedral_group, chiral_icosahedral_orbit,
    full_icosahedral_group, full_icosahedral_orbit,
    icosahedron_vertices, icosahedron_edge_centers, icosahedron_face_centers


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


@inline function chiral_icosahedral_orbit(x::T, y::T, z::T) where {T}
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _five = _four + _one
    _sqrt_five = sqrt(_five)
    half = inv(_two)
    hphi = (_one + _sqrt_five) / _four # half of the golden ratio
    hpsi = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
    hx = half * x
    hy = half * y
    hz = half * z
    ax = hphi * x
    ay = hphi * y
    az = hphi * z
    bx = hpsi * x
    by = hpsi * y
    bz = hpsi * z
    return SA{T}[
        +x; +y; +z;;
        +x; -y; -z;;
        -x; +y; -z;;
        -x; -y; +z;;
        +y; +z; +x;;
        +y; -z; -x;;
        -y; +z; -x;;
        -y; -z; +x;;
        +z; +x; +y;;
        +z; -x; -y;;
        -z; +x; -y;;
        -z; -x; +y;;
        +hx+ay+bz; +ax+by+hz; -bx-hy-az;;
        +hx+ay+bz; -ax-by-hz; +bx+hy+az;;
        +hx+ay-bz; +ax+by-hz; +bx+hy-az;;
        +hx+ay-bz; -ax-by+hz; -bx-hy+az;;
        +hx-ay+bz; +ax-by+hz; +bx-hy+az;;
        +hx-ay+bz; -ax+by-hz; -bx+hy-az;;
        +hx-ay-bz; +ax-by-hz; -bx+hy+az;;
        +hx-ay-bz; -ax+by+hz; +bx-hy-az;;
        -hx+ay+bz; +ax-by-hz; +bx-hy-az;;
        -hx+ay+bz; -ax+by+hz; -bx+hy+az;;
        -hx+ay-bz; +ax-by+hz; -bx+hy-az;;
        -hx+ay-bz; -ax+by-hz; +bx-hy+az;;
        -hx-ay+bz; +ax+by-hz; -bx-hy+az;;
        -hx-ay+bz; -ax-by+hz; +bx+hy-az;;
        -hx-ay-bz; +ax+by+hz; +bx+hy+az;;
        -hx-ay-bz; -ax-by-hz; -bx-hy-az;;
        +ax+by+hz; +bx+hy+az; -hx-ay-bz;;
        +ax+by+hz; -bx-hy-az; +hx+ay+bz;;
        +ax+by-hz; +bx+hy-az; +hx+ay-bz;;
        +ax+by-hz; -bx-hy+az; -hx-ay+bz;;
        +ax-by+hz; +bx-hy+az; +hx-ay+bz;;
        +ax-by+hz; -bx+hy-az; -hx+ay-bz;;
        +ax-by-hz; +bx-hy-az; -hx+ay+bz;;
        +ax-by-hz; -bx+hy+az; +hx-ay-bz;;
        -ax+by+hz; +bx-hy-az; +hx-ay-bz;;
        -ax+by+hz; -bx+hy+az; -hx+ay+bz;;
        -ax+by-hz; +bx-hy+az; -hx+ay-bz;;
        -ax+by-hz; -bx+hy-az; +hx-ay+bz;;
        -ax-by+hz; +bx+hy-az; -hx-ay+bz;;
        -ax-by+hz; -bx-hy+az; +hx+ay-bz;;
        -ax-by-hz; +bx+hy+az; +hx+ay+bz;;
        -ax-by-hz; -bx-hy-az; -hx-ay-bz;;
        +bx+hy+az; +hx+ay+bz; -ax-by-hz;;
        +bx+hy+az; -hx-ay-bz; +ax+by+hz;;
        +bx+hy-az; +hx+ay-bz; +ax+by-hz;;
        +bx+hy-az; -hx-ay+bz; -ax-by+hz;;
        +bx-hy+az; +hx-ay+bz; +ax-by+hz;;
        +bx-hy+az; -hx+ay-bz; -ax+by-hz;;
        +bx-hy-az; +hx-ay-bz; -ax+by+hz;;
        +bx-hy-az; -hx+ay+bz; +ax-by-hz;;
        -bx+hy+az; +hx-ay-bz; +ax-by-hz;;
        -bx+hy+az; -hx+ay+bz; -ax+by+hz;;
        -bx+hy-az; +hx-ay+bz; -ax+by-hz;;
        -bx+hy-az; -hx+ay-bz; +ax-by+hz;;
        -bx-hy+az; +hx+ay-bz; -ax-by+hz;;
        -bx-hy+az; -hx-ay+bz; +ax+by-hz;;
        -bx-hy-az; +hx+ay+bz; +ax+by+hz;;
        -bx-hy-az; -hx-ay-bz; -ax-by-hz
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


@inline function full_icosahedral_orbit(x::T, y::T, z::T) where {T}
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _five = _four + _one
    _sqrt_five = sqrt(_five)
    half = inv(_two)
    hphi = (_one + _sqrt_five) / _four # half of the golden ratio
    hpsi = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
    hx = half * x
    hy = half * y
    hz = half * z
    ax = hphi * x
    ay = hphi * y
    az = hphi * z
    bx = hpsi * x
    by = hpsi * y
    bz = hpsi * z
    return SA{T}[
        +x; +y; +z;;
        +x; +y; -z;;
        +x; -y; +z;;
        +x; -y; -z;;
        -x; +y; +z;;
        -x; +y; -z;;
        -x; -y; +z;;
        -x; -y; -z;;
        +y; +z; +x;;
        +y; +z; -x;;
        +y; -z; +x;;
        +y; -z; -x;;
        -y; +z; +x;;
        -y; +z; -x;;
        -y; -z; +x;;
        -y; -z; -x;;
        +z; +x; +y;;
        +z; +x; -y;;
        +z; -x; +y;;
        +z; -x; -y;;
        -z; +x; +y;;
        -z; +x; -y;;
        -z; -x; +y;;
        -z; -x; -y;;
        +hx+ay+bz; +ax+by+hz; +bx+hy+az;;
        +hx+ay+bz; +ax+by+hz; -bx-hy-az;;
        +hx+ay+bz; -ax-by-hz; +bx+hy+az;;
        +hx+ay+bz; -ax-by-hz; -bx-hy-az;;
        +hx+ay-bz; +ax+by-hz; +bx+hy-az;;
        +hx+ay-bz; +ax+by-hz; -bx-hy+az;;
        +hx+ay-bz; -ax-by+hz; +bx+hy-az;;
        +hx+ay-bz; -ax-by+hz; -bx-hy+az;;
        +hx-ay+bz; +ax-by+hz; +bx-hy+az;;
        +hx-ay+bz; +ax-by+hz; -bx+hy-az;;
        +hx-ay+bz; -ax+by-hz; +bx-hy+az;;
        +hx-ay+bz; -ax+by-hz; -bx+hy-az;;
        +hx-ay-bz; +ax-by-hz; +bx-hy-az;;
        +hx-ay-bz; +ax-by-hz; -bx+hy+az;;
        +hx-ay-bz; -ax+by+hz; +bx-hy-az;;
        +hx-ay-bz; -ax+by+hz; -bx+hy+az;;
        -hx+ay+bz; +ax-by-hz; +bx-hy-az;;
        -hx+ay+bz; +ax-by-hz; -bx+hy+az;;
        -hx+ay+bz; -ax+by+hz; +bx-hy-az;;
        -hx+ay+bz; -ax+by+hz; -bx+hy+az;;
        -hx+ay-bz; +ax-by+hz; +bx-hy+az;;
        -hx+ay-bz; +ax-by+hz; -bx+hy-az;;
        -hx+ay-bz; -ax+by-hz; +bx-hy+az;;
        -hx+ay-bz; -ax+by-hz; -bx+hy-az;;
        -hx-ay+bz; +ax+by-hz; +bx+hy-az;;
        -hx-ay+bz; +ax+by-hz; -bx-hy+az;;
        -hx-ay+bz; -ax-by+hz; +bx+hy-az;;
        -hx-ay+bz; -ax-by+hz; -bx-hy+az;;
        -hx-ay-bz; +ax+by+hz; +bx+hy+az;;
        -hx-ay-bz; +ax+by+hz; -bx-hy-az;;
        -hx-ay-bz; -ax-by-hz; +bx+hy+az;;
        -hx-ay-bz; -ax-by-hz; -bx-hy-az;;
        +ax+by+hz; +bx+hy+az; +hx+ay+bz;;
        +ax+by+hz; +bx+hy+az; -hx-ay-bz;;
        +ax+by+hz; -bx-hy-az; +hx+ay+bz;;
        +ax+by+hz; -bx-hy-az; -hx-ay-bz;;
        +ax+by-hz; +bx+hy-az; +hx+ay-bz;;
        +ax+by-hz; +bx+hy-az; -hx-ay+bz;;
        +ax+by-hz; -bx-hy+az; +hx+ay-bz;;
        +ax+by-hz; -bx-hy+az; -hx-ay+bz;;
        +ax-by+hz; +bx-hy+az; +hx-ay+bz;;
        +ax-by+hz; +bx-hy+az; -hx+ay-bz;;
        +ax-by+hz; -bx+hy-az; +hx-ay+bz;;
        +ax-by+hz; -bx+hy-az; -hx+ay-bz;;
        +ax-by-hz; +bx-hy-az; +hx-ay-bz;;
        +ax-by-hz; +bx-hy-az; -hx+ay+bz;;
        +ax-by-hz; -bx+hy+az; +hx-ay-bz;;
        +ax-by-hz; -bx+hy+az; -hx+ay+bz;;
        -ax+by+hz; +bx-hy-az; +hx-ay-bz;;
        -ax+by+hz; +bx-hy-az; -hx+ay+bz;;
        -ax+by+hz; -bx+hy+az; +hx-ay-bz;;
        -ax+by+hz; -bx+hy+az; -hx+ay+bz;;
        -ax+by-hz; +bx-hy+az; +hx-ay+bz;;
        -ax+by-hz; +bx-hy+az; -hx+ay-bz;;
        -ax+by-hz; -bx+hy-az; +hx-ay+bz;;
        -ax+by-hz; -bx+hy-az; -hx+ay-bz;;
        -ax-by+hz; +bx+hy-az; +hx+ay-bz;;
        -ax-by+hz; +bx+hy-az; -hx-ay+bz;;
        -ax-by+hz; -bx-hy+az; +hx+ay-bz;;
        -ax-by+hz; -bx-hy+az; -hx-ay+bz;;
        -ax-by-hz; +bx+hy+az; +hx+ay+bz;;
        -ax-by-hz; +bx+hy+az; -hx-ay-bz;;
        -ax-by-hz; -bx-hy-az; +hx+ay+bz;;
        -ax-by-hz; -bx-hy-az; -hx-ay-bz;;
        +bx+hy+az; +hx+ay+bz; +ax+by+hz;;
        +bx+hy+az; +hx+ay+bz; -ax-by-hz;;
        +bx+hy+az; -hx-ay-bz; +ax+by+hz;;
        +bx+hy+az; -hx-ay-bz; -ax-by-hz;;
        +bx+hy-az; +hx+ay-bz; +ax+by-hz;;
        +bx+hy-az; +hx+ay-bz; -ax-by+hz;;
        +bx+hy-az; -hx-ay+bz; +ax+by-hz;;
        +bx+hy-az; -hx-ay+bz; -ax-by+hz;;
        +bx-hy+az; +hx-ay+bz; +ax-by+hz;;
        +bx-hy+az; +hx-ay+bz; -ax+by-hz;;
        +bx-hy+az; -hx+ay-bz; +ax-by+hz;;
        +bx-hy+az; -hx+ay-bz; -ax+by-hz;;
        +bx-hy-az; +hx-ay-bz; +ax-by-hz;;
        +bx-hy-az; +hx-ay-bz; -ax+by+hz;;
        +bx-hy-az; -hx+ay+bz; +ax-by-hz;;
        +bx-hy-az; -hx+ay+bz; -ax+by+hz;;
        -bx+hy+az; +hx-ay-bz; +ax-by-hz;;
        -bx+hy+az; +hx-ay-bz; -ax+by+hz;;
        -bx+hy+az; -hx+ay+bz; +ax-by-hz;;
        -bx+hy+az; -hx+ay+bz; -ax+by+hz;;
        -bx+hy-az; +hx-ay+bz; +ax-by+hz;;
        -bx+hy-az; +hx-ay+bz; -ax+by-hz;;
        -bx+hy-az; -hx+ay-bz; +ax-by+hz;;
        -bx+hy-az; -hx+ay-bz; -ax+by-hz;;
        -bx-hy+az; +hx+ay-bz; +ax+by-hz;;
        -bx-hy+az; +hx+ay-bz; -ax-by+hz;;
        -bx-hy+az; -hx-ay+bz; +ax+by-hz;;
        -bx-hy+az; -hx-ay+bz; -ax-by+hz;;
        -bx-hy-az; +hx+ay+bz; +ax+by+hz;;
        -bx-hy-az; +hx+ay+bz; -ax-by-hz;;
        -bx-hy-az; -hx-ay-bz; +ax+by+hz;;
        -bx-hy-az; -hx-ay-bz; -ax-by-hz
    ]
end


@inline function icosahedron_vertices(::Type{T}) where {T}
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _five = _four + _one
    _ten = _five + _five
    _sqrt_five = sqrt(_five)
    a = sqrt((_five + _sqrt_five) / _ten)
    b = sqrt((_five - _sqrt_five) / _ten)
    return SA{T}[
        _zero; +a; +b;;
        _zero; +a; -b;;
        _zero; -a; +b;;
        _zero; -a; -b;;
        +b; _zero; +a;;
        +b; _zero; -a;;
        -b; _zero; +a;;
        -b; _zero; -a;;
        +a; +b; _zero;;
        +a; -b; _zero;;
        -a; +b; _zero;;
        -a; -b; _zero
    ]
end


@inline function icosahedron_edge_centers(::Type{T}) where {T}
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
        +_one; _zero; _zero;;
        -_one; _zero; _zero;;
        _zero; +_one; _zero;;
        _zero; -_one; _zero;;
        _zero; _zero; +_one;;
        _zero; _zero; -_one;;
        +half; +hphi; +hpsi;;
        +half; +hphi; -hpsi;;
        +half; -hphi; +hpsi;;
        +half; -hphi; -hpsi;;
        -half; +hphi; +hpsi;;
        -half; +hphi; -hpsi;;
        -half; -hphi; +hpsi;;
        -half; -hphi; -hpsi;;
        +hphi; +hpsi; +half;;
        +hphi; +hpsi; -half;;
        +hphi; -hpsi; +half;;
        +hphi; -hpsi; -half;;
        -hphi; +hpsi; +half;;
        -hphi; +hpsi; -half;;
        -hphi; -hpsi; +half;;
        -hphi; -hpsi; -half;;
        +hpsi; +half; +hphi;;
        +hpsi; +half; -hphi;;
        +hpsi; -half; +hphi;;
        +hpsi; -half; -hphi;;
        -hpsi; +half; +hphi;;
        -hpsi; +half; -hphi;;
        -hpsi; -half; +hphi;;
        -hpsi; -half; -hphi
    ]
end


@inline function icosahedron_face_centers(::Type{T}) where {T}
    _zero = zero(T)
    _one = one(T)
    _two = _one + _one
    _three = _two + _one
    _four = _two + _two
    _five = _four + _one
    _six = _four + _two
    _sqrt_five = sqrt(_five)
    t = inv(sqrt(_three))
    c = sqrt((_three + _sqrt_five) / _six)
    d = sqrt((_three - _sqrt_five) / _six)
    return SA{T}[
        +t; +t; +t;;
        +t; +t; -t;;
        +t; -t; +t;;
        +t; -t; -t;;
        -t; +t; +t;;
        -t; +t; -t;;
        -t; -t; +t;;
        -t; -t; -t;;
        _zero; +d; +c;;
        _zero; +d; -c;;
        _zero; -d; +c;;
        _zero; -d; -c;;
        +c; _zero; +d;;
        +c; _zero; -d;;
        -c; _zero; +d;;
        -c; _zero; -d;;
        +d; +c; _zero;;
        +d; -c; _zero;;
        -d; +c; _zero;;
        -d; -c; _zero
    ]
end


################################################################################

end # module SphericalSymmetries
