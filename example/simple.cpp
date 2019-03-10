//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/differentiation/autodiff.hpp>
#include <iostream>

int main()
{
    using namespace boost::math::differentiation;

    const auto x = make_fvar<double,3>(13);
    const auto y = make_fvar<double,0,4>(14);
    const auto z = 10*x*x + 50*x*y + 100*y*y; // promoted to autodiff_fvar<double,3,4>
    for (std::size_t i=0 ; i<=3 ; ++i)
        for (std::size_t j=0 ; j<=4 ; ++j)
            std::cout << "z.derivative("<<i<<","<<j<<") = " << z.derivative(i,j) << std::endl;
    return 0;
}
/*
Output:
z.derivative(2,0) = 20
z.derivative(1,1) = 50
z.derivative(0,2) = 200
**/
