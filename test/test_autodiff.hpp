//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_AUTODIFF_HPP
#define BOOST_MATH_TEST_AUTODIFF_HPP

#include <boost/config.hpp>
#include <boost/math/differentiation/autodiff.hpp>
#include <boost/mp11.hpp>
#include <boost/mp11/mpl.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/range/irange.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <random>
#include <type_traits>

#define BOOST_TEST_MODULE test_autodiff
#include <boost/test/included/unit_test.hpp>

#if defined(_MSC_VER) || defined(BOOST_MSVC)
#define NOMINMAX
#endif

// using bin_float_types = mp_list<float,double,long
// double,boost::multiprecision::cpp_bin_float_50>;
using bin_float_types = boost::mp11::mp_list<float, double, long double>;
//  cpp_bin_float_50 is fixed in boost 1.70
// float blows up in unchecked_factorial

// cpp_dec_float_50 cannot be used with close_at_tolerance
/*using multiprecision_float_types =
    mp_list<boost::multiprecision::cpp_dec_float_50, boost::multiprecision::cpp_bin_float_50>;*/
using multiprecision_float_types = boost::mp11::mp_list<>;

using all_float_types = boost::mp11::mp_append<bin_float_types, multiprecision_float_types>;

using namespace boost::math::differentiation;

namespace test_detail {

/**
 * struct to emit pseudo-random values from a given interval.
 * Endpoints are closed or open depending on whether or not they're infinite).
 */
template <typename T, typename = void>
struct RandomSample;

template <typename T>
struct RandomSample<
    T, typename std::enable_if<std::is_floating_point<T>::value || std::numeric_limits<T>::is_integer>::type> {
  using dist_t = typename boost::conditional<std::is_floating_point<T>::value, std::uniform_real_distribution<T>,
                                             std::uniform_int_distribution<T>>::type;
  template <typename U, typename V>
  RandomSample(U start, V finish)
      : start_(static_cast<T>(start)),
        finish_(static_cast<T>(finish)),
        random_device_{},
        rng_(random_device_()),
        dist_(start_, ((std::nextafter))(finish_, ((std::numeric_limits<T>::max))())) {}

  T next() noexcept { return dist_(rng_); }

  T start_;
  T finish_;
  std::random_device random_device_;
  std::mt19937 rng_;
  dist_t dist_;
};
static_assert(std::is_same<typename RandomSample<float>::dist_t, std::uniform_real_distribution<float>>::value, "");
static_assert(std::is_same<typename RandomSample<int64_t>::dist_t, std::uniform_int_distribution<int64_t>>::value, "");

/**
 * Simple struct to hold constants that are used in each test
 * since BOOST_AUTO_TEST_CASE_TEMPLATE doesn't support fixtures.
 */
template <typename T, typename Order>
struct test_constants_t;

template <typename T, typename Order, Order val>
struct test_constants_t<T, std::integral_constant<Order, val>> {
  static constexpr int n_samples = 25;
  static constexpr Order order = val;
  static constexpr T mp_epsilon_multiplier = boost::mp11::mp_if<
      boost::mp11::mp_or<boost::multiprecision::is_number<T>, boost::multiprecision::is_number_expression<T>>,
      boost::mp11::mp_int<1>, boost::mp11::mp_int<0>>::value;
  static constexpr T eps = std::numeric_limits<T>::epsilon();
};

template <typename T, typename U>
constexpr bool check_if_small(const T& lhs, const U& rhs) noexcept {
  using boost::math::differentiation::detail::get_root_type;
  using boost::math::differentiation::detail::is_fvar;
  using real_type = promote<T, U>;

  return std::numeric_limits<real_type>::epsilon() >
         fabs((std::max)(static_cast<real_type>(lhs), static_cast<real_type>(rhs)) -
              (std::min)(static_cast<real_type>(lhs), static_cast<real_type>(rhs)));
}
}  // namespace test_detail

template <typename T, int m = 3>
using test_constants_t = test_detail::test_constants_t<T, boost::mp11::mp_int<m>>;

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
  if (cp == call) {
    return S * Phi(d1) - exp(-r * tau) * K * Phi(d2);
  } else
    return exp(-r * tau) * K * Phi(-d2) - S * Phi(-d1);
}

template <typename T>
T uncast_return(const T& x) {
  return x == 0 ? 0 : 1;
}

#endif  // BOOST_MATH_TEST_AUTODIFF_HPP