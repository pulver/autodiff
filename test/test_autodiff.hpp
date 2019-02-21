//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_AUTODIFF_HPP
#define BOOST_MATH_TEST_AUTODIFF_HPP

#if defined(_MSC_VER)
#define NOMINMAX
#endif

#include <boost/math/differentiation/autodiff.hpp>
#include <boost/mp11.hpp>
#include <boost/mp11/mpl.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#define BOOST_TEST_MODULE test_autodiff
#include <boost/test/included/unit_test.hpp>

// using bin_float_types = mp_list<float,double,long
// double,boost::multiprecision::cpp_bin_float_50>;
using bin_float_types = boost::mp11::mp_list<float, double, long double>;  // cpp_bin_float_50 is
                                                                           // fixed in boost 1.70
// cpp_dec_float_50 cannot be used with close_at_tolerance
// using multiprecision_float_types =
// mp_list<boost::multiprecision::cpp_dec_float_50>;
using multiprecision_float_types = boost::mp11::mp_list<>;

using all_float_types = boost::mp11::mp_append<bin_float_types, multiprecision_float_types>;

using namespace boost::math::differentiation;

template <typename W, typename X, typename Y, typename Z>
promote<W, X, Y, Z> mixed_partials_f(const W& w, const X& x, const Y& y, const Z& z) {
  return exp(w * sin(x * log(y) / z) + sqrt(w * z / (x * y))) + w * w / tan(z);
}

// Equations and function/variable names are from
// https://en.wikipedia.org/wiki/Greeks_(finance)#Formulas_for_European_option_Greeks
//
// Standard normal probability density function
template <typename T>
T phi(const T& x) {
  return boost::math::constants::one_div_root_two_pi<T>() * exp(-0.5 * x * x);
}

// Standard normal cumulative distribution function
template <typename T>
T Phi(const T& x) {
  return 0.5 * erfc(-boost::math::constants::one_div_root_two<T>() * x);
}

enum CP { call, put };

// Assume zero annual dividend yield (q=0).
template <typename Price, typename Sigma, typename Tau, typename Rate>
promote<Price, Sigma, Tau, Rate> black_scholes_option_price(CP cp, double K, const Price& S, const Sigma& sigma,
                                                            const Tau& tau, const Rate& r) {
  const auto d1 = (log(S / K) + (r + sigma * sigma / 2) * tau) / (sigma * sqrt(tau));
  const auto d2 = (log(S / K) + (r - sigma * sigma / 2) * tau) / (sigma * sqrt(tau));
  static_assert(std::is_same<decltype(S * Phi(d1) - exp(-r * tau) * K * Phi(d2)),
                             decltype(exp(-r * tau) * K * Phi(-d2) - S * Phi(-d1))>::value,
                "decltype(call) != decltype(put)");
  if (cp == call)
    return S * Phi(d1) - exp(-r * tau) * K * Phi(d2);
  else
    return exp(-r * tau) * K * Phi(-d2) - S * Phi(-d1);
}

template <typename T>
T uncast_return(const T& x) {
  return x == 0 ? 0 : 1;
}

#endif // BOOST_MATH_TEST_AUTODIFF_HPP