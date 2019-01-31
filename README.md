# Autodiff - Automatic Differentiation C++ Library

[![Build Status](https://travis-ci.com/pulver/autodiff.svg?branch=master)](https://travis-ci.com/pulver/autodiff)
[![Build status](https://ci.appveyor.com/api/projects/status/hmhefrokif2n1b9t/branch/master?svg=true)](https://ci.appveyor.com/project/pulver/autodiff/branch/master)
[![codecov](https://codecov.io/gh/pulver/autodiff/branch/dev-boost/graph/badge.svg)](https://codecov.io/gh/pulver/autodiff)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=kedarbhat_autodiff&metric=alert_status)](https://sonarcloud.io/dashboard/index/kedarbhat_autodiff)

## Introduction and Quick-Start Examples

Autodiff is a header-only C++ library that facilitates the [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) (forward mode) of mathematical functions in single and multiple variables.

The formula central to this implementation of automatic differentiation is the following [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) expansion of an analytic function *f* at a point *x0*:

![\begin{align*} f(x_0+\varepsilon) &= f(x_0) + f'(x_0)\varepsilon + \frac{f''(x_0)}{2!}\varepsilon^2 + \frac{f'''(x_0)}{3!}\varepsilon^3 + \cdots \\ &= \sum_{n=0}^N\frac{f^{(n)}(x_0)}{n!}\varepsilon^n + O\left(\varepsilon^{N+1}\right). \end{align*}](doc/images/taylor_series.png)

The essential idea of autodiff is the substitution of numbers with polynomials in the evaluation of *f*. By selecting the proper polynomial as input, the resulting polynomial contains the function's derivatives within the polynomial coefficients. One then multiplies by a factorial term to obtain the desired derivative of any order.

Assume one is interested in the first *N* derivatives of *f* at *x0*. Then without any loss of precision to the calculation of the derivatives, all terms that include powers greater than *N* can be discarded, and under these truncation rules, *f* provides a polynomial-to-polynomial transformation:

![\[ f\quad:\quad x_0+\varepsilon\quad\mapsto\quad\sum_{n=0}^N\frac{f^{(n)}(x_0)}{n!}\varepsilon^n. \]](doc/images/polynomial_transform.png)

C++ includes the ability to overload operators and functions, and thus when *f* is written as a template function that can receive and return a generic type, then that is sufficient to perform automatic differentiation: Create a class that models polynomials, and overload all of the arithmetic operators to model polynomial arithmetic that drop all terms with powers greater than *N*. The derivatives are then found in the coefficients of the return value. This is essentially what the autodiff library does (generalizing to multiple independent variables.)

See the [autodiff documentation](http://www.unitytechgroup.com/doc/autodiff/) for more details.

### Example 1: Single-Variable Differentiation

![Calculate derivatives for x to 4th power](doc/images/fourth_power.png)

``` c++
#include <boost/math/differentiation/autodiff.hpp>
#include <iostream>

template<typename T>
T fourth_power(T x)
{
    x *= x;
    return x *= x;
}

int main()
{
    using namespace boost::math::differentiation;

    constexpr int Order=5; // The highest order derivative to be calculated.
    const autodiff::variable<double,Order> x(2.0); // Find derivatives at x=2.
    const autodiff::variable<double,Order> y = fourth_power(x);
    for (int i=0 ; i<=Order ; ++i)
        std::cout << "y.derivative("<<i<<") = " << y.derivative(i) << std::endl;
    return 0;
}
/*
Compile:
$ g++ -std=c++1z example/fourth_power.cpp

Output:
$ ./a.out
y.derivative(0) = 16
y.derivative(1) = 32
y.derivative(2) = 48
y.derivative(3) = 48
y.derivative(4) = 24
y.derivative(5) = 0
*/
```

The above calculates

![LaTeX for y.derivative() calls](doc/images/single-variable_derivatives.png)

### Example 2: Multi-Variable and Multi-Precision Differentiation

![12th-order mixed-partial derivative with about 100 decimal digits](doc/images/mixed_partial_multiprecision.png)

``` c++
#include <boost/math/differentiation/autodiff.hpp>
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
  using namespace boost::math::differentiation;

  constexpr int Nw=3; // Max order of derivative to calculate for w
  constexpr int Nx=2; // Max order of derivative to calculate for x
  constexpr int Ny=4; // Max order of derivative to calculate for y
  constexpr int Nz=3; // Max order of derivative to calculate for z
  using var = autodiff::variable<cpp_dec_float_100,Nw,Nx,Ny,Nz>;
  const var w = autodiff::variable<cpp_dec_float_100,Nw>(11);
  const var x = autodiff::variable<cpp_dec_float_100,0,Nx>(12);
  const var y = autodiff::variable<cpp_dec_float_100,0,0,Ny>(13);
  const var z = autodiff::variable<cpp_dec_float_100,0,0,0,Nz>(14);
  const var v = f(w,x,y,z);
  // Calculated from Mathematica symbolic differentiation. See multiprecision.nb for script.
  const cpp_dec_float_100 answer("1976.31960074779771777988187529041872090812118921875499076582535951111845769110560421820940516423255314");
  std::cout << std::setprecision(std::numeric_limits<cpp_dec_float_100>::digits10)
    << "mathematica   : " << answer << '\n'
    << "autodiff      : " << v.derivative(Nw,Nx,Ny,Nz) << '\n'
    << "relative error: " << std::setprecision(3) << (v.derivative(Nw,Nx,Ny,Nz)/answer-1) << std::endl;
  return 0;
}
/*
Compile:
$ g++ -std=c++1z example/multiprecision.cpp

Output:
$ ./a.out
mathematica   : 1976.319600747797717779881875290418720908121189218754990765825359511118457691105604218209405164232553
autodiff      : 1976.319600747797717779881875290418720908121189218754990765825359511118457691105604218209405164232566
relative error: 6.47e-99
*/
```

### Example 3: Black-Scholes option pricing.

Using the standard Black-Scholes model for pricing European options, calculate call/put prices and greeks.

https://en.wikipedia.org/wiki/Greeks_(finance)#Formulas_for_European_option_Greeks

One of the primary benefits of using automatic differentiation is the elimination of additional functions to
calculate derivatives, which is a form of code redundancy.

``` c++
#include <boost/math/differentiation/autodiff.hpp>
#include <iostream>

using namespace boost::math::differentiation;

// Equations and function/variable names are from
// https://en.wikipedia.org/wiki/Greeks_(finance)#Formulas_for_European_option_Greeks

// Standard normal probability density function
template<typename X>
X phi(const X& x)
{
  return boost::math::constants::one_div_root_two_pi<double>()*exp(-0.5*x*x);
}

// Standard normal cumulative distribution function
template<typename X>
X Phi(const X& x)
{
  return 0.5*erfc(-boost::math::constants::one_div_root_two<double>()*x);
}

enum CP { call, put };

// Assume zero annual dividend yield (q=0).
template<typename Price,typename Sigma,typename Tau,typename Rate>
autodiff::promote<Price,Sigma,Tau,Rate>
    black_scholes_option_price(CP cp, double K, const Price& S, const Sigma& sigma, const Tau& tau, const Rate& r)
{
  using namespace std;
  const auto d1 = (log(S/K) + (r+sigma*sigma/2)*tau) / (sigma*sqrt(tau));
  const auto d2 = (log(S/K) + (r-sigma*sigma/2)*tau) / (sigma*sqrt(tau));
  if (cp == call)
    return S*Phi(d1) - exp(-r*tau)*K*Phi(d2);
  else
    return exp(-r*tau)*K*Phi(-d2) - S*Phi(-d1);
}

int main()
{
  const double K = 100.0; // Strike price.
  const autodiff::variable<double,3> S(105); // Stock price.
  const autodiff::variable<double,0,3> sigma(5); // Volatility.
  const autodiff::variable<double,0,0,1> tau(30.0/365); // Time to expiration in years. (30 days).
  const autodiff::variable<double,0,0,0,1> r(1.25/100); // Interest rate.
  const auto call_price = black_scholes_option_price(call, K, S, sigma, tau, r);
  const auto put_price  = black_scholes_option_price(put,  K, S, sigma, tau, r);

  // Compare automatically calculated greeks by autodiff with formulas for greeks.
  // https://en.wikipedia.org/wiki/Greeks_(finance)#Formulas_for_European_option_Greeks
  const double d1 = static_cast<double>((log(S/K) + (r+sigma*sigma/2)*tau) / (sigma*sqrt(tau)));
  const double d2 = static_cast<double>((log(S/K) + (r-sigma*sigma/2)*tau) / (sigma*sqrt(tau)));
  const double formula_call_delta = +Phi(+d1);
  const double formula_put_delta  = -Phi(-d1);
  const double formula_vega = static_cast<double>(S*phi(d1)*sqrt(tau));
  const double formula_call_theta = static_cast<double>(-S*phi(d1)*sigma/(2*sqrt(tau))-r*K*exp(-r*tau)*Phi(+d2));
  const double formula_put_theta  = static_cast<double>(-S*phi(d1)*sigma/(2*sqrt(tau))+r*K*exp(-r*tau)*Phi(-d2));
  const double formula_call_rho = static_cast<double>(+K*tau*exp(-r*tau)*Phi(+d2));
  const double formula_put_rho  = static_cast<double>(-K*tau*exp(-r*tau)*Phi(-d2));
  const double formula_gamma = static_cast<double>(phi(d1)/(S*sigma*sqrt(tau)));
  const double formula_vanna = static_cast<double>(-phi(d1)*d2/sigma);
  const double formula_charm = static_cast<double>(phi(d1)*(d2*sigma*sqrt(tau)-2*r*tau)/(2*tau*sigma*sqrt(tau)));
  const double formula_vomma = static_cast<double>(S*phi(d1)*sqrt(tau)*d1*d2/sigma);
  const double formula_veta = static_cast<double>(-S*phi(d1)*sqrt(tau)*(r*d1/(sigma*sqrt(tau))-(1+d1*d2)/(2*tau)));
  const double formula_speed = static_cast<double>(-phi(d1)*(d1/(sigma*sqrt(tau))+1)/(S*S*sigma*sqrt(tau)));
  const double formula_zomma = static_cast<double>(phi(d1)*(d1*d2-1)/(S*sigma*sigma*sqrt(tau)));
  const double formula_color =
    static_cast<double>(-phi(d1)/(2*S*tau*sigma*sqrt(tau))*(1+(2*r*tau-d2*sigma*sqrt(tau))*d1/(sigma*sqrt(tau))));
  const double formula_ultima = -formula_vega*static_cast<double>((d1*d2*(1-d1*d2)+d1*d1+d2*d2)/(sigma*sigma));

  std::cout << std::setprecision(std::numeric_limits<double>::digits10)
    << "autodiff black-scholes call price = " << call_price.derivative(0,0,0,0) << '\n'
    << "autodiff black-scholes put  price = " << put_price.derivative(0,0,0,0) << '\n'
    << "\n## First-order Greeks\n"
    << "autodiff call delta = " << call_price.derivative(1,0,0,0) << '\n'
    << " formula call delta = " << formula_call_delta << '\n'
    << "autodiff call vega  = " << call_price.derivative(0,1,0,0) << '\n'
    << " formula call vega  = " << formula_vega << '\n'
    << "autodiff call theta = " << -call_price.derivative(0,0,1,0) << '\n' // minus sign due to tau = T-time
    << " formula call theta = " << formula_call_theta << '\n'
    << "autodiff call rho   = " << call_price.derivative(0,0,0,1) << '\n'
    << " formula call rho   = " << formula_call_rho << '\n'
    << '\n'
    << "autodiff put delta = " << put_price.derivative(1,0,0,0) << '\n'
    << " formula put delta = " << formula_put_delta << '\n'
    << "autodiff put vega  = " << put_price.derivative(0,1,0,0) << '\n'
    << " formula put vega  = " << formula_vega << '\n'
    << "autodiff put theta = " << -put_price.derivative(0,0,1,0) << '\n'
    << " formula put theta = " << formula_put_theta << '\n'
    << "autodiff put rho   = " << put_price.derivative(0,0,0,1) << '\n'
    << " formula put rho   = " << formula_put_rho << '\n'
    << "\n## Second-order Greeks\n"
    << "autodiff call gamma = " << call_price.derivative(2,0,0,0) << '\n'
    << "autodiff put  gamma = " << put_price.derivative(2,0,0,0) << '\n'
    << "      formula gamma = " << formula_gamma << '\n'
    << "autodiff call vanna = " << call_price.derivative(1,1,0,0) << '\n'
    << "autodiff put  vanna = " << put_price.derivative(1,1,0,0) << '\n'
    << "      formula vanna = " << formula_vanna << '\n'
    << "autodiff call charm = " << -call_price.derivative(1,0,1,0) << '\n'
    << "autodiff put  charm = " << -put_price.derivative(1,0,1,0) << '\n'
    << "      formula charm = " << formula_charm << '\n'
    << "autodiff call vomma = " << call_price.derivative(0,2,0,0) << '\n'
    << "autodiff put  vomma = " << put_price.derivative(0,2,0,0) << '\n'
    << "      formula vomma = " << formula_vomma << '\n'
    << "autodiff call veta = " << call_price.derivative(0,1,1,0) << '\n'
    << "autodiff put  veta = " << put_price.derivative(0,1,1,0) << '\n'
    << "      formula veta = " << formula_veta << '\n'
    << "\n## Third-order Greeks\n"
    << "autodiff call speed = " << call_price.derivative(3,0,0,0) << '\n'
    << "autodiff put  speed = " << put_price.derivative(3,0,0,0) << '\n'
    << "      formula speed = " << formula_speed << '\n'
    << "autodiff call zomma = " << call_price.derivative(2,1,0,0) << '\n'
    << "autodiff put  zomma = " << put_price.derivative(2,1,0,0) << '\n'
    << "      formula zomma = " << formula_zomma << '\n'
    << "autodiff call color = " << call_price.derivative(2,0,1,0) << '\n'
    << "autodiff put  color = " << put_price.derivative(2,0,1,0) << '\n'
    << "      formula color = " << formula_color << '\n'
    << "autodiff call ultima = " << call_price.derivative(0,3,0,0) << '\n'
    << "autodiff put  ultima = " << put_price.derivative(0,3,0,0) << '\n'
    << "      formula ultima = " << formula_ultima << '\n'
    ;
  return 0;
}
/*
Compile:
$ g++ -std=c++1z example/black_scholes.cpp

Output:
$ ./a.out
autodiff black-scholes call price = 56.5136030677739
autodiff black-scholes put  price = 51.4109161009333

## First-order Greeks
autodiff call delta = 0.773818444921273
 formula call delta = 0.773818444921274
autodiff call vega  = 9.05493427705736
 formula call vega  = 9.05493427705736
autodiff call theta = -275.73013426444
 formula call theta = -275.73013426444
autodiff call rho   = 2.03320550539396
 formula call rho   = 2.03320550539396

autodiff put delta = -0.226181555078726
 formula put delta = -0.226181555078726
autodiff put vega  = 9.05493427705736
 formula put vega  = 9.05493427705736
autodiff put theta = -274.481417851526
 formula put theta = -274.481417851526
autodiff put rho   = -6.17753255212599
 formula put rho   = -6.17753255212599

## Second-order Greeks
autodiff call gamma = 0.00199851912993254
autodiff put  gamma = 0.00199851912993254
      formula gamma = 0.00199851912993254
autodiff call vanna = 0.0410279463126531
autodiff put  vanna = 0.0410279463126531
      formula vanna = 0.0410279463126531
autodiff call charm = -1.2505564233679
autodiff put  charm = -1.2505564233679
      formula charm = -1.2505564233679
autodiff call vomma = -0.928114149313108
autodiff put  vomma = -0.928114149313108
      formula vomma = -0.928114149313107
autodiff call veta = 26.7947073115641
autodiff put  veta = 26.7947073115641
      formula veta = 26.7947073115641

## Third-order Greeks
autodiff call speed = -2.90117322380992e-05
autodiff put  speed = -2.90117322380992e-05
      formula speed = -2.90117322380992e-05
autodiff call zomma = -0.000604548369901419
autodiff put  zomma = -0.000604548369901419
      formula zomma = -0.000604548369901419
autodiff call color = -0.0184014426606065
autodiff put  color = -0.0184014426606065
      formula color = -0.0184014426606065
autodiff call ultima = -0.0922426864775683
autodiff put  ultima = -0.0922426864775683
      formula ultima = -0.0922426864775685
*/
```

## Requirements

 - C++11 compiler, but optimized for C++17.
 - [Boost](https://www.boost.org/) library. (Headers only; no linking required.)
