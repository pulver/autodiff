//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/differentiation/autodiff.hpp>
#include <iostream>

int main()
{
    using namespace boost::math::differentiation;

    const autodiff::variable<double,3> x(13);
    const autodiff::variable<double,0,4> y(14);
    const auto z = 10*x*x + 50*x*y + 100*y*y; // promoted to autodiff::variable<double,3,4>
    for (int i=0 ; i<=3 ; ++i)
        for (int j=0 ; j<=4 ; ++j)
            std::cout << "z.derivative("<<i<<","<<j<<") = " << z.derivative(i,j) << std::endl;
    return 0;
}
/*
Compile:
$ g++ -std=c++1z example/simple.cpp

Output:
$ ./a.out
z.derivative(2,0) = 20
z.derivative(1,1) = 50
z.derivative(0,2) = 200
**/
