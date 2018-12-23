#include <boost/math/autodiff.hpp> // Currently proposed.
#include <iostream>

template<typename T>
T fourth_power(T x)
{
    x *= x;
    return x *= x;
}

int main()
{
    constexpr int Order=5; // The highest order derivative to be calculated.
    const boost::math::autodiff::variable<double,Order> x(2.0); // Find derivatives at x=2.
    const boost::math::autodiff::variable<double,Order> y = fourth_power(x);
    for (int i=0 ; i<=Order ; ++i)
        std::cout << "y.derivative("<<i<<") = " << y.derivative(i) << std::endl;
    return 0;
}
/*
Compile:
$ g++ -std=c++1z -Iinclude example/fourth_power.cpp

Output:
$ ./a.out
y.derivative(0) = 16
y.derivative(1) = 32
y.derivative(2) = 48
y.derivative(3) = 48
y.derivative(4) = 24
y.derivative(5) = 0
*/
