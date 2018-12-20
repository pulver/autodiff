#include <boost/math/autodiff.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iostream>

template<typename W,typename X,typename Y,typename Z>
auto f(const W& w, const X& x, const Y& y, const Z& z)
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
  const boost::math::autodiff::variable<cpp_dec_float_100,Nw> w(11);
  const boost::math::autodiff::variable<cpp_dec_float_100,0,Nx> x(12);
  const boost::math::autodiff::variable<cpp_dec_float_100,0,0,Ny> y(13);
  const boost::math::autodiff::variable<cpp_dec_float_100,0,0,0,Nz> z(14);
  const auto v = f(w,x,y,z); // auto = boost::math::autodiff::variable<double,Nw,Nx,Ny,Nz>
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
*/
