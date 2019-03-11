//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_AUTODIFF_HPP
#define BOOST_MATH_TEST_AUTODIFF_HPP

#include <boost/config.hpp>
#include <boost/math/differentiation/autodiff.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <boost/mp11/integral.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/range/irange.hpp>

#include <algorithm>
#include <cfenv>
#include <cstdlib>
#include <random>

#define BOOST_TEST_MODULE test_autodiff
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#if defined(_MSC_VER) || defined(BOOST_MSVC)
#define NOMINMAX
#endif

namespace mp11 = boost::mp11;
namespace bmp = boost::multiprecision;

// using bin_float_types = mp_list<float,double,long
// double,bmp::cpp_bin_float_50>;
using bin_float_types = mp11::mp_list<float, double, long double>;
//  cpp_bin_float_50 is fixed in boost 1.70
// float blows up in unchecked_factorial

// cpp_dec_float_50 cannot be used with close_at_tolerance
/*using multiprecision_float_types =
    mp_list<bmp::cpp_dec_float_50, bmp::cpp_bin_float_50>;*/
using multiprecision_float_types = mp11::mp_list<>;

using all_float_types = mp11::mp_append<bin_float_types, multiprecision_float_types>;

using namespace boost::math::differentiation;

namespace test_detail {

/**
 * Simple struct to hold constants that are used in each test
 * since BOOST_AUTO_TEST_CASE_TEMPLATE doesn't support fixtures.
 */
template <typename T, typename Order> struct test_constants_t;

template <typename T, typename Order, Order val> struct test_constants_t<T, std::integral_constant<Order, val>> {
  static constexpr int n_samples = mp11::mp_if<mp11::mp_or<bmp::is_number<T>, bmp::is_number_expression<T>>,
                                               mp11::mp_int<10>, mp11::mp_int<25>>::value;
  static constexpr Order order = val;
  static constexpr T pct_epsilon() { return 50 * std::numeric_limits<T>::epsilon() * 100; }
};

/**
 * struct to emit pseudo-random values from a given interval.
 * Endpoints are closed or open depending on whether or not they're infinite).
 */

template <typename T> struct RandomSample {
  using is_multiprecision_t = mp11::mp_or<bmp::is_number<T>, bmp::is_number_expression<T>>;
  using numeric_limits_t = std::numeric_limits<T>;
  using is_integer_t = mp11::mp_bool<std::numeric_limits<T>::is_integer>;

  using distribution_param_t =
      mp11::mp_if<is_multiprecision_t,
                  mp11::mp_if<is_integer_t, mp11::mp_if_c<numeric_limits_t::is_signed, int64_t, uint64_t>, long double>,
                  T>;
  static_assert((std::numeric_limits<T>::is_integer && std::numeric_limits<distribution_param_t>::is_integer) ||
                    (!std::numeric_limits<T>::is_integer && !std::numeric_limits<distribution_param_t>::is_integer),
                "T and distribution_param_t must either both be integral or both be not integral");

  using dist_t = mp11::mp_if<is_integer_t, std::uniform_int_distribution<distribution_param_t>,
                             std::uniform_real_distribution<distribution_param_t>>;

  struct get_integral_endpoint {
    template <typename V> constexpr distribution_param_t operator()(V finish) const noexcept {
      return static_cast<distribution_param_t>(finish);
    }
  };

  struct get_real_endpoint {
    template <typename V> constexpr distribution_param_t operator()(V finish) const noexcept {
      return ((std::nextafter))(static_cast<distribution_param_t>(finish),
                                ((std::numeric_limits<distribution_param_t>::max))());
    }
  };

  using get_endpoint_t = mp11::mp_if<is_integer_t, get_integral_endpoint, get_real_endpoint>;

  template <typename U, typename V>
  RandomSample(U start, V finish)
      : rng_(std::random_device{}()), dist_(static_cast<distribution_param_t>(start), get_endpoint_t{}(finish)) {}

  T next() noexcept { return static_cast<T>(dist_(rng_)); }
  T normalize(const T& x) noexcept { return x / (dist_.max() - dist_.min()); }

  std::mt19937 rng_;
  dist_t dist_;
};
static_assert(std::is_same<typename RandomSample<float>::dist_t, std::uniform_real_distribution<float>>::value, "");
static_assert(std::is_same<typename RandomSample<int64_t>::dist_t, std::uniform_int_distribution<int64_t>>::value, "");
static_assert(
    std::is_same<typename RandomSample<bmp::uint512_t>::dist_t, std::uniform_int_distribution<uint64_t>>::value, "");
static_assert(std::is_same<typename RandomSample<bmp::cpp_bin_float_50>::dist_t,
                           std::uniform_real_distribution<long double>>::value,
              "");

}  // namespace test_detail

template <typename T, int m = 3> using test_constants_t = test_detail::test_constants_t<T, mp11::mp_int<m>>;

template <typename W, typename X, typename Y, typename Z>
promote<W, X, Y, Z> mixed_partials_f(const W& w, const X& x, const Y& y, const Z& z) {
  return exp(w * sin(x * log(y) / z) + sqrt(w * z / (x * y))) + w * w / tan(z);
}

// Equations and function/variable names are from
// https://en.wikipedia.org/wiki/Greeks_(finance)#Formulas_for_European_option_Greeks
//
// Standard normal probability density function
template <typename T> T phi(const T& x) { return boost::math::constants::one_div_root_two_pi<T>() * exp(-0.5 * x * x); }

// Standard normal cumulative distribution function
template <typename T> T Phi(const T& x) { return 0.5 * erfc(-boost::math::constants::one_div_root_two<T>() * x); }

enum class CP { call, put };

// Assume zero annual dividend yield (q=0).
template <typename Price, typename Sigma, typename Tau, typename Rate>
promote<Price, Sigma, Tau, Rate> black_scholes_option_price(CP cp, double K, const Price& S, const Sigma& sigma,
                                                            const Tau& tau, const Rate& r) {
  const auto d1 = (log(S / K) + (r + sigma * sigma / 2) * tau) / (sigma * sqrt(tau));
  const auto d2 = (log(S / K) + (r - sigma * sigma / 2) * tau) / (sigma * sqrt(tau));
  if (cp == CP::call) {
    return S * Phi(d1) - exp(-r * tau) * K * Phi(d2);
  }
  return exp(-r * tau) * K * Phi(-d2) - S * Phi(-d1);
}

template <typename T> T uncast_return(const T& x) { return x == 0 ? 0 : 1; }

#endif  // BOOST_MATH_TEST_AUTODIFF_HPP
