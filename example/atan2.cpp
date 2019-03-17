//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/differentiation/autodiff.hpp>
#include <iostream>

int main()
{
    using namespace boost::math::differentiation;
    using T = double;
    const T x = 0.5;
    const T y = 0.5*boost::math::constants::root_three<T>();
    constexpr size_t m = 5;
    const auto z = atan2(make_fvar<T,m>(y), make_fvar<T,0,m>(x));
    // Mathematica: Flatten@Transpose@Table[D[ArcTan[x,y],{x,i},{y,j}] /. {x->1/2, y->Sqrt[3]/2}, {i,0,5}, {j,0,5}]
    const T expected[(m+1)*(m+1)] { boost::math::constants::third_pi<T>(),
        -0.5*boost::math::constants::root_three<T>(), 0.5*boost::math::constants::root_three<T>(), 0,
        -3*boost::math::constants::root_three<T>(), 12*boost::math::constants::root_three<T>(), 0.5, 0.5, -2,
        3, 12, -120, -0.5*boost::math::constants::root_three<T>(), 0, 3*boost::math::constants::root_three<T>(),
        -12*boost::math::constants::root_three<T>(), 0, 360*boost::math::constants::root_three<T>(), 2, -3, -12,
        120, -360, -2520, -3*boost::math::constants::root_three<T>(), 12*boost::math::constants::root_three<T>(), 0,
        -360*boost::math::constants::root_three<T>(), 2520*boost::math::constants::root_three<T>(), 0, 12, -120, 360,
        2520, -40320, 181440 };
    size_t k=0;
    std::cout << "Should be 0: z.derivative(4,5) = " << z.derivative(4,5) << std::endl;
    /*
    for (size_t i=0 ; i<=m ; ++i)
        for (size_t j=0 ; j<=m ; ++j)
        auto autodiff_v = z.derivative(i,j);
        auto anchor_v = expected[k++];
        //std::cout << "z.derivative("<<i<<','<<j<<") = " << z.derivative(i,j) << std::endl;
    */
    return 0;
}
