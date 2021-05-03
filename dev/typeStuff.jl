struct MyType
    a::Int
    b::Int
    MyType(a) = new(a,1)
end

fun(a::MyType) = a.b

struct MyType2
    a::Int
end

b(a) = 1

fun(a::MyType2) = begin; if a.a==1 || typeof(a)==MyType2; b(a); end; end

a = MyType(2)
a = MyType(2)

a2 = MyType2(2)
a2 = MyType2(2)

@code_llvm MyType(2)
@code_typed MyType(2)

@code_llvm MyType2(2)
@code_typed MyType2(2)


@code_llvm fun(a)
@code_typed fun(a)

@code_llvm fun(a2)
@code_typed fun(a2)


