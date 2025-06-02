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
    h = inv(_two)
    a = (_one + _sqrt_five) / _four # half of the golden ratio
    b = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
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
        +h +a +b; +a +b +h; -b -h -a;;;
        +h +a +b; -a -b -h; +b +h +a;;;
        +h +a -b; +a +b -h; +b +h -a;;;
        +h +a -b; -a -b +h; -b -h +a;;;
        +h -a +b; +a -b +h; +b -h +a;;;
        +h -a +b; -a +b -h; -b +h -a;;;
        +h -a -b; +a -b -h; -b +h +a;;;
        +h -a -b; -a +b +h; +b -h -a;;;
        -h +a +b; +a -b -h; +b -h -a;;;
        -h +a +b; -a +b +h; -b +h +a;;;
        -h +a -b; +a -b +h; -b +h -a;;;
        -h +a -b; -a +b -h; +b -h +a;;;
        -h -a +b; +a +b -h; -b -h +a;;;
        -h -a +b; -a -b +h; +b +h -a;;;
        -h -a -b; +a +b +h; +b +h +a;;;
        -h -a -b; -a -b -h; -b -h -a;;;
        +a +b +h; +b +h +a; -h -a -b;;;
        +a +b +h; -b -h -a; +h +a +b;;;
        +a +b -h; +b +h -a; +h +a -b;;;
        +a +b -h; -b -h +a; -h -a +b;;;
        +a -b +h; +b -h +a; +h -a +b;;;
        +a -b +h; -b +h -a; -h +a -b;;;
        +a -b -h; +b -h -a; -h +a +b;;;
        +a -b -h; -b +h +a; +h -a -b;;;
        -a +b +h; +b -h -a; +h -a -b;;;
        -a +b +h; -b +h +a; -h +a +b;;;
        -a +b -h; +b -h +a; -h +a -b;;;
        -a +b -h; -b +h -a; +h -a +b;;;
        -a -b +h; +b +h -a; -h -a +b;;;
        -a -b +h; -b -h +a; +h +a -b;;;
        -a -b -h; +b +h +a; +h +a +b;;;
        -a -b -h; -b -h -a; -h -a -b;;;
        +b +h +a; +h +a +b; -a -b -h;;;
        +b +h +a; -h -a -b; +a +b +h;;;
        +b +h -a; +h +a -b; +a +b -h;;;
        +b +h -a; -h -a +b; -a -b +h;;;
        +b -h +a; +h -a +b; +a -b +h;;;
        +b -h +a; -h +a -b; -a +b -h;;;
        +b -h -a; +h -a -b; -a +b +h;;;
        +b -h -a; -h +a +b; +a -b -h;;;
        -b +h +a; +h -a -b; +a -b -h;;;
        -b +h +a; -h +a +b; -a +b +h;;;
        -b +h -a; +h -a +b; -a +b -h;;;
        -b +h -a; -h +a -b; +a -b +h;;;
        -b -h +a; +h +a -b; -a -b +h;;;
        -b -h +a; -h -a +b; +a +b -h;;;
        -b -h -a; +h +a +b; +a +b +h;;;
        -b -h -a; -h -a -b; -a -b -h
    ]
end


@inline function chiral_icosahedral_orbit(x::T, y::T, z::T) where {T}
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _five = _four + _one
    _sqrt_five = sqrt(_five)
    h = inv(_two)
    a = (_one + _sqrt_five) / _four # half of the golden ratio
    b = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
    hx = h * x
    hy = h * y
    hz = h * z
    ax = a * x
    ay = a * y
    az = a * z
    bx = b * x
    by = b * y
    bz = b * z
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
    h = inv(_two)
    a = (_one + _sqrt_five) / _four # half of the golden ratio
    b = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
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
        +h +a +b; +a +b +h; +b +h +a;;;
        +h +a +b; +a +b +h; -b -h -a;;;
        +h +a +b; -a -b -h; +b +h +a;;;
        +h +a +b; -a -b -h; -b -h -a;;;
        +h +a -b; +a +b -h; +b +h -a;;;
        +h +a -b; +a +b -h; -b -h +a;;;
        +h +a -b; -a -b +h; +b +h -a;;;
        +h +a -b; -a -b +h; -b -h +a;;;
        +h -a +b; +a -b +h; +b -h +a;;;
        +h -a +b; +a -b +h; -b +h -a;;;
        +h -a +b; -a +b -h; +b -h +a;;;
        +h -a +b; -a +b -h; -b +h -a;;;
        +h -a -b; +a -b -h; +b -h -a;;;
        +h -a -b; +a -b -h; -b +h +a;;;
        +h -a -b; -a +b +h; +b -h -a;;;
        +h -a -b; -a +b +h; -b +h +a;;;
        -h +a +b; +a -b -h; +b -h -a;;;
        -h +a +b; +a -b -h; -b +h +a;;;
        -h +a +b; -a +b +h; +b -h -a;;;
        -h +a +b; -a +b +h; -b +h +a;;;
        -h +a -b; +a -b +h; +b -h +a;;;
        -h +a -b; +a -b +h; -b +h -a;;;
        -h +a -b; -a +b -h; +b -h +a;;;
        -h +a -b; -a +b -h; -b +h -a;;;
        -h -a +b; +a +b -h; +b +h -a;;;
        -h -a +b; +a +b -h; -b -h +a;;;
        -h -a +b; -a -b +h; +b +h -a;;;
        -h -a +b; -a -b +h; -b -h +a;;;
        -h -a -b; +a +b +h; +b +h +a;;;
        -h -a -b; +a +b +h; -b -h -a;;;
        -h -a -b; -a -b -h; +b +h +a;;;
        -h -a -b; -a -b -h; -b -h -a;;;
        +a +b +h; +b +h +a; +h +a +b;;;
        +a +b +h; +b +h +a; -h -a -b;;;
        +a +b +h; -b -h -a; +h +a +b;;;
        +a +b +h; -b -h -a; -h -a -b;;;
        +a +b -h; +b +h -a; +h +a -b;;;
        +a +b -h; +b +h -a; -h -a +b;;;
        +a +b -h; -b -h +a; +h +a -b;;;
        +a +b -h; -b -h +a; -h -a +b;;;
        +a -b +h; +b -h +a; +h -a +b;;;
        +a -b +h; +b -h +a; -h +a -b;;;
        +a -b +h; -b +h -a; +h -a +b;;;
        +a -b +h; -b +h -a; -h +a -b;;;
        +a -b -h; +b -h -a; +h -a -b;;;
        +a -b -h; +b -h -a; -h +a +b;;;
        +a -b -h; -b +h +a; +h -a -b;;;
        +a -b -h; -b +h +a; -h +a +b;;;
        -a +b +h; +b -h -a; +h -a -b;;;
        -a +b +h; +b -h -a; -h +a +b;;;
        -a +b +h; -b +h +a; +h -a -b;;;
        -a +b +h; -b +h +a; -h +a +b;;;
        -a +b -h; +b -h +a; +h -a +b;;;
        -a +b -h; +b -h +a; -h +a -b;;;
        -a +b -h; -b +h -a; +h -a +b;;;
        -a +b -h; -b +h -a; -h +a -b;;;
        -a -b +h; +b +h -a; +h +a -b;;;
        -a -b +h; +b +h -a; -h -a +b;;;
        -a -b +h; -b -h +a; +h +a -b;;;
        -a -b +h; -b -h +a; -h -a +b;;;
        -a -b -h; +b +h +a; +h +a +b;;;
        -a -b -h; +b +h +a; -h -a -b;;;
        -a -b -h; -b -h -a; +h +a +b;;;
        -a -b -h; -b -h -a; -h -a -b;;;
        +b +h +a; +h +a +b; +a +b +h;;;
        +b +h +a; +h +a +b; -a -b -h;;;
        +b +h +a; -h -a -b; +a +b +h;;;
        +b +h +a; -h -a -b; -a -b -h;;;
        +b +h -a; +h +a -b; +a +b -h;;;
        +b +h -a; +h +a -b; -a -b +h;;;
        +b +h -a; -h -a +b; +a +b -h;;;
        +b +h -a; -h -a +b; -a -b +h;;;
        +b -h +a; +h -a +b; +a -b +h;;;
        +b -h +a; +h -a +b; -a +b -h;;;
        +b -h +a; -h +a -b; +a -b +h;;;
        +b -h +a; -h +a -b; -a +b -h;;;
        +b -h -a; +h -a -b; +a -b -h;;;
        +b -h -a; +h -a -b; -a +b +h;;;
        +b -h -a; -h +a +b; +a -b -h;;;
        +b -h -a; -h +a +b; -a +b +h;;;
        -b +h +a; +h -a -b; +a -b -h;;;
        -b +h +a; +h -a -b; -a +b +h;;;
        -b +h +a; -h +a +b; +a -b -h;;;
        -b +h +a; -h +a +b; -a +b +h;;;
        -b +h -a; +h -a +b; +a -b +h;;;
        -b +h -a; +h -a +b; -a +b -h;;;
        -b +h -a; -h +a -b; +a -b +h;;;
        -b +h -a; -h +a -b; -a +b -h;;;
        -b -h +a; +h +a -b; +a +b -h;;;
        -b -h +a; +h +a -b; -a -b +h;;;
        -b -h +a; -h -a +b; +a +b -h;;;
        -b -h +a; -h -a +b; -a -b +h;;;
        -b -h -a; +h +a +b; +a +b +h;;;
        -b -h -a; +h +a +b; -a -b -h;;;
        -b -h -a; -h -a -b; +a +b +h;;;
        -b -h -a; -h -a -b; -a -b -h
    ]
end


@inline function full_icosahedral_orbit(x::T, y::T, z::T) where {T}
    _one = one(T)
    _two = _one + _one
    _four = _two + _two
    _five = _four + _one
    _sqrt_five = sqrt(_five)
    h = inv(_two)
    a = (_one + _sqrt_five) / _four # half of the golden ratio
    b = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
    hx = h * x
    hy = h * y
    hz = h * z
    ax = a * x
    ay = a * y
    az = a * z
    bx = b * x
    by = b * y
    bz = b * z
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
    h = inv(_two)
    a = (_one + _sqrt_five) / _four # half of the golden ratio
    b = (_one - _sqrt_five) / _four # half of the golden ratio conjugate
    return SA{T}[
        +_one; _zero; _zero;;
        -_one; _zero; _zero;;
        _zero; +_one; _zero;;
        _zero; -_one; _zero;;
        _zero; _zero; +_one;;
        _zero; _zero; -_one;;
        +h; +a; +b;;
        +h; +a; -b;;
        +h; -a; +b;;
        +h; -a; -b;;
        -h; +a; +b;;
        -h; +a; -b;;
        -h; -a; +b;;
        -h; -a; -b;;
        +a; +b; +h;;
        +a; +b; -h;;
        +a; -b; +h;;
        +a; -b; -h;;
        -a; +b; +h;;
        -a; +b; -h;;
        -a; -b; +h;;
        -a; -b; -h;;
        +b; +h; +a;;
        +b; +h; -a;;
        +b; -h; +a;;
        +b; -h; -a;;
        -b; +h; +a;;
        -b; +h; -a;;
        -b; -h; +a;;
        -b; -h; -a
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
    a = sqrt((_three + _sqrt_five) / _six)
    b = sqrt((_three - _sqrt_five) / _six)
    return SA{T}[
        +t; +t; +t;;
        +t; +t; -t;;
        +t; -t; +t;;
        +t; -t; -t;;
        -t; +t; +t;;
        -t; +t; -t;;
        -t; -t; +t;;
        -t; -t; -t;;
        _zero; +b; +a;;
        _zero; +b; -a;;
        _zero; -b; +a;;
        _zero; -b; -a;;
        +a; _zero; +b;;
        +a; _zero; -b;;
        -a; _zero; +b;;
        -a; _zero; -b;;
        +b; +a; _zero;;
        +b; -a; _zero;;
        -b; +a; _zero;;
        -b; -a; _zero
    ]
end


################################################################################

end # module SphericalSymmetries
