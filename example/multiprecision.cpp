//               Copyright Matthew Pulver 2018.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/autodiff.hpp> // Currently proposed.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iostream>

template<typename T>
T f(const T& w, const T& x, const T& y, const T& z)
{
  using namespace std;
  return exp(w*sin(x*log(y)/z) + sqrt(w*z/(x*y))) + w*w/tan(z);
}

int main()
{
  using cpp_dec_float_100 = boost::multiprecision::cpp_dec_float_100;
  // Calculated from Mathematica symbolic differentiation. See multiprecision.nb for script.
  const cpp_dec_float_100 answer("1976.31960074779771777988187529041872090812118921875499076582535951111845769110560421820940516423255314");
  constexpr int Nw=3; // Max order of derivative to calculate for w
  constexpr int Nx=2; // Max order of derivative to calculate for x
  constexpr int Ny=4; // Max order of derivative to calculate for y
  constexpr int Nz=3; // Max order of derivative to calculate for z
  using AdType = boost::math::autodiff::variable<cpp_dec_float_100,Nw,Nx,Ny,Nz>;
  const AdType w = boost::math::autodiff::variable<cpp_dec_float_100,Nw>(11);
  const AdType x = boost::math::autodiff::variable<cpp_dec_float_100,0,Nx>(12);
  const AdType y = boost::math::autodiff::variable<cpp_dec_float_100,0,0,Ny>(13);
  const AdType z = boost::math::autodiff::variable<cpp_dec_float_100,0,0,0,Nz>(14);
  const AdType v = f(w,x,y,z);
  std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
    << "mathematica   : " << answer << '\n'
    << "autodiff      : " << v.derivative(Nw,Nx,Ny,Nz) << '\n'
    << "relative error: " << std::setprecision(3) << (v.derivative(Nw,Nx,Ny,Nz)/answer-1) << std::endl;
  return 0;
}
/*
Compile:
$ g++ -std=c++1z -Iinclude example/multiprecision.cpp

Output:
$ ./a.out
mathematica   : 1976.319600747797717779881875290418720908121189218754990765825359511118457691105604218209405164232553
autodiff      : 1976.319600747797717779881875290418720908121189218754990765825359511118457691105604218209405164232566
relative error: 6.47e-99
**/
