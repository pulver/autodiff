//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

#include <boost/range/irange.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <random>
#include <type_traits>

using namespace boost::math::differentiation;

/*********************************************************************************************************************
 * special functions tests
 *********************************************************************************************************************/

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
  RandomSample(T start, T finish)
      : start_(start),
        finish_(finish),
        rng_(std::random_device{}()),
        dist_(start_, std::nextafter(finish_, boost::math::tools::max_value<T>())) {}

  T next() noexcept { return dist_(rng_); }

  T start_;
  T finish_;
  std::mt19937 rng_;
  dist_t dist_;
};
static_assert(std::is_same<typename RandomSample<float>::dist_t, std::uniform_real_distribution<float>>::value, "");
static_assert(std::is_same<typename RandomSample<long>::dist_t, std::uniform_int_distribution<long>>::value, "");

/**
 * Simple struct to hold constants that are used in each test
 * since BOOST_AUTO_TEST_CASE_TEMPLATE doesn't support fixtures.
 */
template <typename T, typename Order>
struct test_constants_t;

template <typename T, typename Order, Order val>
struct test_constants_t<T, std::integral_constant<Order, val>> {
  static constexpr T pct_epsilon = 50 * std::numeric_limits<T>::epsilon() * 100;
  static constexpr int n_samples = 25;
  static constexpr Order order = val;
};

template <typename T, typename U>
constexpr bool check_if_small(const T &lhs, const U &rhs) noexcept {
  using boost::math::differentiation::detail::get_root_type;
  using boost::math::differentiation::detail::is_fvar;
  using real_type = typename std::common_type<boost::mp11::mp_if<is_fvar<T>, typename get_root_type<T>::type, T>,
                                              boost::mp11::mp_if<is_fvar<U>, typename get_root_type<U>::type, U>>::type;

  return std::numeric_limits<real_type>::epsilon() >
         fabs((std::max)(static_cast<real_type>(lhs), static_cast<real_type>(rhs)) -
              (std::min)(static_cast<real_type>(lhs), static_cast<real_type>(rhs)));
}
}  // namespace detail

template <typename T, int m = 3>
using test_constants_t = test_detail::test_constants_t<T, boost::mp11::mp_int<m>>;

using testing_types = boost::mp11::mp_list<double>;

BOOST_AUTO_TEST_SUITE(test_autodiff_5)

BOOST_AUTO_TEST_CASE_TEMPLATE(airy_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-100, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::airy_ai(make_fvar<T, m>(x));
      auto anchor_v = boost::math::airy_ai(x);
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_ai(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_ai(static_cast<T>(x)), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_ai(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_ai(static_cast<T>(x)), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::airy_ai_prime(make_fvar<T, m>(x));
      auto anchor_v = boost::math::airy_ai_prime(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_ai_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_ai_prime(static_cast<T>(x)), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_ai_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_ai_prime(static_cast<T>(x)), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::airy_bi(make_fvar<T, m>(x));
      auto anchor_v = boost::math::airy_bi(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_bi(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_bi(static_cast<T>(x)), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_bi(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_bi(static_cast<T>(x)), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::airy_bi_prime(make_fvar<T, m>(x));
      auto anchor_v = boost::math::airy_bi_prime(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_bi_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_bi_prime(static_cast<T>(x)), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::airy_bi_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::airy_bi_prime(static_cast<T>(x)), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    if (i > 0) {
      try {
        auto autodiff_v = boost::math::airy_ai_zero<autodiff_fvar<T, m>>(i);
        auto anchor_v = boost::math::airy_ai_zero<T>(i);
        if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
          BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
        } else {
          BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
        }
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(((boost::math::airy_ai_zero<autodiff_fvar<T, m>>(i))),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::airy_ai_zero<T>(i), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        BOOST_REQUIRE_THROW(((boost::math::airy_ai_zero<autodiff_fvar<T, m>>(i))),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::airy_ai_zero<T>(i), boost::wrapexcept<std::overflow_error>);
      } catch (...) {
        std::cout << "Input: x: " << x << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }

      try {
        auto autodiff_v = boost::math::airy_bi_zero<autodiff_fvar<T, m>>(i);
        auto anchor_v = boost::math::airy_bi_zero<T>(i);
        if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
          BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
        } else {
          BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
        }
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(((boost::math::airy_bi_zero<autodiff_fvar<T, m>>(i))),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::airy_bi_zero<T>(i), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        BOOST_REQUIRE_THROW(((boost::math::airy_bi_zero<autodiff_fvar<T, m>>(i))),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::airy_bi_zero<T>(i), boost::wrapexcept<std::overflow_error>);
      } catch (...) {
        std::cout << "Input: x: " << x << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(acosh_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::acosh(make_fvar<T, m>(x));
      auto anchor_v = boost::math::acosh(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((acosh(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      BOOST_REQUIRE_THROW(boost::math::acosh(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((acosh(make_fvar<T, m>(x)), std::fetestexcept(FE_OVERFLOW)));
      BOOST_REQUIRE_THROW(boost::math::acosh(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asinh_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::asinh(make_fvar<T, m>(x));
      auto anchor_v = boost::math::asinh(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((asinh(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      BOOST_REQUIRE_THROW(boost::math::asinh(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((asinh(make_fvar<T, m>(x)), std::fetestexcept(FE_OVERFLOW)));
      BOOST_REQUIRE_THROW(boost::math::asinh(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atanh_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::atanh(make_fvar<T, m>(x));
      auto anchor_v = boost::math::atanh(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((atanh(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      BOOST_REQUIRE_THROW(boost::math::atanh(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((atanh(make_fvar<T, m>(x)), std::fetestexcept(FE_OVERFLOW)));
      BOOST_REQUIRE_THROW(boost::math::atanh(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bernoulli_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<int> x_sampler{0, 2000};

  for (auto i : boost::irange(test_constants::n_samples)) {
    try {
      BOOST_REQUIRE_EQUAL(((boost::math::bernoulli_b2n<autodiff_fvar<T, m>>(i))), boost::math::bernoulli_b2n<T>(i));
    } catch (const std::domain_error &e) {
      BOOST_REQUIRE_THROW(((boost::math::bernoulli_b2n<autodiff_fvar<T, m>>(i))), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::bernoulli_b2n<T>(i), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &e) {
      BOOST_REQUIRE_THROW(((boost::math::bernoulli_b2n<autodiff_fvar<T, m>>(i))),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::bernoulli_b2n<T>(i), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: i: " << i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_EQUAL(((boost::math::tangent_t2n<autodiff_fvar<T, m>>(i))), boost::math::tangent_t2n<T>(i));
    } catch (const std::domain_error &e) {
      BOOST_REQUIRE_THROW(((boost::math::tangent_t2n<autodiff_fvar<T, m>>(i))), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::tangent_t2n<T>(i), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &e) {
      BOOST_REQUIRE_THROW(((boost::math::tangent_t2n<autodiff_fvar<T, m>>(i))), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::tangent_t2n<T>(i), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: i: " << i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bessel_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> v_sampler{-100, 100};
  test_detail::RandomSample<T> x_sampler{-boost::math::tools::log_max_value<T>() + 1,
                                    boost::math::tools::log_max_value<T>() - 1};
  for (auto i : boost::irange(test_constants::n_samples)) {
    auto v = v_sampler.next();
    auto x = x_sampler.next();
    if (v == 0) {
      continue;
    }

    try {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      auto autodiff_v = boost::math::cyl_bessel_i(make_fvar<T, m>(v), make_fvar<T, m>(x_i));
      auto anchor_v = boost::math::cyl_bessel_i(v, x_i);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i(make_fvar<T, m>(v), make_fvar<T, m>(x_i)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i(v, x_i), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i(make_fvar<T, m>(v), make_fvar<T, m>(x_i)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i(v, x_i), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      std::cout << "Input: v: " << v << " x_i: " << x_i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto x_j = abs(x) + 1;
      auto autodiff_v = boost::math::cyl_bessel_j(make_fvar<T, m>(v), make_fvar<T, m>(x_j));
      auto anchor_v = boost::math::cyl_bessel_j(v, x_j);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_j = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j(make_fvar<T, m>(v), make_fvar<T, m>(x_j)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j(v, x_j), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_j = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j(make_fvar<T, m>(v), make_fvar<T, m>(x_j)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j(v, x_j), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_j = abs(x) + 1;
      std::cout << "Input: v: " << v << " x_j: " << x_j << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::cyl_bessel_j_zero(make_fvar<T, m>(v), i + 1);
      auto anchor_v = boost::math::cyl_bessel_j_zero(v, i + 1);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_zero(make_fvar<T, m>(v), i + 1),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_zero(v, i), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_zero(make_fvar<T, m>(v), i + 1),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_zero(v, i + 1), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: v: " << v << " i+1: " << i + 1 << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto x_k = abs(x) + 1;
      auto autodiff_v = boost::math::cyl_bessel_k(make_fvar<T, m>(v), make_fvar<T, m>(x_k));
      auto anchor_v = boost::math::cyl_bessel_k(v, x_k);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_k = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k(make_fvar<T, m>(v), make_fvar<T, m>(x_k)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k(v, x_k), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_k = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k(make_fvar<T, m>(v), make_fvar<T, m>(x_k)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k(v, x_k), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_k = abs(x) + 1;
      std::cout << "Input: v: " << v << " x_k: " << x_k << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto x_neumann = abs(x);
      auto autodiff_v = boost::math::cyl_neumann(make_fvar<T, m>(v), make_fvar<T, m>(x_neumann));
      auto anchor_v = boost::math::cyl_neumann(v, x_neumann);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_neumann = abs(x);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann(make_fvar<T, m>(v), make_fvar<T, m>(x_neumann)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann(v, x_neumann), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_neumann = abs(x);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann(make_fvar<T, m>(v), make_fvar<T, m>(x_neumann)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann(v, x_neumann), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_neumann = abs(x);
      std::cout << "Input: v: " << v << " x_neumann: " << x_neumann << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::cyl_neumann_zero(make_fvar<T, m>(v), i + 1);
      auto anchor_v = boost::math::cyl_neumann_zero(v, i + 1);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_zero(make_fvar<T, m>(v), i + 1),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_zero(v, i), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_zero(make_fvar<T, m>(v), i + 1),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_zero(v, i + 1), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: v: " << v << " i+1: " << i + 1 << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::sph_bessel<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v));
      auto anchor_v = boost::math::sph_bessel<T>(i, v);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_bessel<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_bessel<T>(i, v), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_bessel<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_bessel<T>(i, v), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: i: " << i << " v: " << v << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::sph_neumann<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v));
      auto anchor_v = boost::math::sph_neumann<T>(i, v);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_neumann<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_neumann<T>(i, v), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_neumann<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_neumann<T>(i, v), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: i: " << i << " v: " << v << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      auto autodiff_v = boost::math::cyl_bessel_i_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_i));
      auto anchor_v = boost::math::cyl_bessel_i_prime(v, x_i);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_i)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i_prime(v, x_i), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_i)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_i_prime(v, x_i), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_i = x < 0 ? boost::math::itrunc(x) : x;
      std::cout << "Input: v: " << v << " x_i: " << x_i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto x_j = abs(x) + 1;
      auto autodiff_v = boost::math::cyl_bessel_j_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_j));
      auto anchor_v = boost::math::cyl_bessel_j_prime(v, x_j);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_j = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_j)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_prime(v, x_j), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_j = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_j)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_j_prime(v, x_j), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_j = abs(x) + 1;
      std::cout << "Input: v: " << v << " x_j: " << x_j << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto x_k = abs(x) + 1;
      auto autodiff_v = boost::math::cyl_bessel_k_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_k));
      auto anchor_v = boost::math::cyl_bessel_k_prime(v, x_k);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_k = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_k)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k_prime(v, x_k), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_k = abs(x) + 1;
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_k)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_bessel_k_prime(v, x_k), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_k = abs(x) + 1;
      std::cout << "Input: v: " << v << " x_k: " << x_k << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto x_neumann = abs(x);
      auto autodiff_v = boost::math::cyl_neumann_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_neumann));
      auto anchor_v = boost::math::cyl_neumann_prime(v, x_neumann);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      auto x_neumann = abs(x);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_neumann)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_prime(v, x_neumann), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      auto x_neumann = abs(x);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_prime(make_fvar<T, m>(v), make_fvar<T, m>(x_neumann)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cyl_neumann_prime(v, x_neumann), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      auto x_neumann = abs(x);
      std::cout << "Input: v: " << v << " x_neumann: " << x_neumann << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::sph_bessel_prime<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v));
      auto anchor_v = boost::math::sph_bessel_prime<T>(i, v);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_bessel_prime<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_bessel_prime<T>(i, v), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_bessel_prime<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_bessel_prime<T>(i, v), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: i: " << i << " v: " << v << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::sph_neumann_prime<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v));
      auto anchor_v = boost::math::sph_neumann_prime<T>(i, v);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_neumann_prime<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_neumann_prime<T>(i, v), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(((boost::math::sph_neumann_prime<autodiff_fvar<T, m>>(i, make_fvar<T, m>(v)))),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::sph_neumann_prime<T>(i, v), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: i: " << i << " v: " << v << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(beta_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> a_sampler{-2000, 2000};
  test_detail::RandomSample<T> b_sampler{-2000, 2000};
  test_detail::RandomSample<T> z_sampler{0, 1};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto a = a_sampler.next();
    auto b = b_sampler.next();
    try {
      auto autodiff_v = boost::math::beta(fabs(make_fvar<T, m>(a)), fabs(make_fvar<T, m>(b)));
      auto anchor_v = boost::math::beta(fabs(a), fabs(b));
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::beta(fabs(make_fvar<T, m>(a)), fabs(make_fvar<T, m>(b))),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::beta(fabs(a), fabs(b)), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::beta(fabs(make_fvar<T, m>(a)), fabs(make_fvar<T, m>(b))),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::beta(fabs(a), fabs(b)), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << fabs(a) << "  b: " << fabs(b) << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    auto z = z_sampler.next();
    try {
      auto autodiff_v = boost::math::beta(fabs(make_fvar<T, m>(a)), fabs(make_fvar<T, m>(b)));
      auto anchor_v = boost::math::beta(fabs(a), fabs(b));
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(
          boost::math::betac(fabs(make_fvar<T, m>(a)), fabs(make_fvar<T, m>(b)), fabs(make_fvar<T, m>(z))),
          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::betac(fabs(a), fabs(b), fabs(z)), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(
          boost::math::betac(fabs(make_fvar<T, m>(a)), fabs(make_fvar<T, m>(b)), fabs(make_fvar<T, m>(z))),
          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::betac(fabs(a), fabs(b), fabs(z)), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << fabs(a) << "  b: " << fabs(b) << "  z: " << fabs(z) << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibeta(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibeta(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::beta(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::beta(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibeta_derivative(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibeta_derivative(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_derivative(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_derivative(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_derivative(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_derivative(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibeta_inv(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibeta_inv<T>(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_inv(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_inv(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_inv(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_inv(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibetac_inv(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibetac_inv<T>(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac_inv(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac_inv(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac_inv(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac_inv(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibeta_inva(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibeta_inva(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_inva(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_inva(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_inva(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_inva(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibetac_inva(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibetac_inva(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac_inva(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac_inva(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac_inva(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac_inva(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibeta_invb(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibeta_invb(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_invb(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_invb(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibeta_invb(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibeta_invb(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::ibetac_invb(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z));
      auto anchor_v = boost::math::ibetac_invb(a, b, z);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac_invb(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac_invb(a, b, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ibetac_invb(make_fvar<T, m>(a), make_fvar<T, m>(b), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ibetac_invb(a, b, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: a: " << a << "  b: " << b << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(binomial_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<unsigned> n_sampler{0u, 100};
  test_detail::RandomSample<unsigned> r_sampler{0u, 100};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto n = n_sampler.next();
    r_sampler.dist_.param(typename test_detail::RandomSample<unsigned>::dist_t::param_type(0, n));
    auto r = r_sampler.next();
    try {
      auto autodiff_v = boost::math::binomial_coefficient<autodiff_fvar<T, m>>(n, r);
      auto anchor_v = boost::math::binomial_coefficient<T>(n, r);
      if (std::isfinite(static_cast<T>(autodiff_v)) && std::isfinite(anchor_v)) {
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, test_constants::pct_epsilon);
      } else {
        BOOST_REQUIRE(!(std::isfinite(static_cast<T>(autodiff_v)) || std::isfinite(anchor_v)));
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(((boost::math::binomial_coefficient<autodiff_fvar<T, m>>(n, r))),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::binomial_coefficient<T>(n, r), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(((boost::math::binomial_coefficient<autodiff_fvar<T, m>>(n, r))),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::binomial_coefficient<T>(n, r), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: n: " << n << "  r: " << r << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(cbrt_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      if (boost::math::isinf(x) || x == 0) {
        BOOST_REQUIRE_EQUAL(boost::math::cbrt(make_fvar<T, m>(x)), x);
      } else {
        BOOST_REQUIRE_CLOSE(boost::math::cbrt(make_fvar<T, m>(x)), boost::math::cbrt(x), test_constants::pct_epsilon);
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::cbrt(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::cbrt(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::cbrt(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::cbrt(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(chebyshev_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  {
    test_detail::RandomSample<T> x_sampler{-2, 2};
    T t_0 = 1;
    T x = x_sampler.next();
    T t_1 = x;
    for (auto i : boost::irange(test_constants::n_samples)) {
      std::ignore = i;
      try {
        std::swap(t_0, t_1);
        auto tmp = boost::math::chebyshev_next(x, t_0, t_1);
        BOOST_REQUIRE_EQUAL(boost::math::chebyshev_next(make_fvar<T, m>(x), make_fvar<T, m>(t_0), make_fvar<T, m>(t_1)),
                            tmp);
        t_1 = tmp;
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_next(make_fvar<T, m>(x), make_fvar<T, m>(t_0), make_fvar<T, m>(t_1)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_next(x, t_0, t_1), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_next(make_fvar<T, m>(x), make_fvar<T, m>(t_0), make_fvar<T, m>(t_1)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_next(x, t_0, t_1), boost::wrapexcept<std::overflow_error>);
      } catch (...) {
        std::cout << "Input: x: " << x << "  t_0: " << t_0 << "  t_1: " << t_1 << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
  }
  {
    test_detail::RandomSample<unsigned> n_sampler{0, 10};
    test_detail::RandomSample<T> x_sampler{-2, 2};
    for (auto i : boost::irange(test_constants::n_samples)) {
      std::ignore = i;
      auto n = n_sampler.next();
      auto x = x_sampler.next();
      try {
        BOOST_REQUIRE_CLOSE_FRACTION(boost::math::chebyshev_t(n, make_fvar<T, m>(x)), boost::math::chebyshev_t(n, x),
                                     4000 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t(n, make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t(n, x), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t(n, make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t(n, x), boost::wrapexcept<std::overflow_error>);
      } catch (...) {
        std::cout << "Inputs: n: " << n << "  x: " << x << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }

      try {
        BOOST_REQUIRE_CLOSE_FRACTION(boost::math::chebyshev_u(n, make_fvar<T, m>(x)), boost::math::chebyshev_u(n, x),
                                     4000 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_u(n, make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_u(n, x), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_u(n, make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_u(n, x), boost::wrapexcept<std::overflow_error>);
      } catch (...) {
        std::cout << "Inputs: n: " << n << "  x: " << x << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }

      try {
        BOOST_REQUIRE_CLOSE_FRACTION(boost::math::chebyshev_t_prime(n, make_fvar<T, m>(x)),
                                     boost::math::chebyshev_t_prime(n, x), 4000 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t_prime(n, make_fvar<T, m>(x)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t_prime(n, x), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t_prime(n, make_fvar<T, m>(x)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::chebyshev_t_prime(n, x), boost::wrapexcept<std::overflow_error>);
      } catch (...) {
        std::cout << "Inputs: n: " << n << "  x: " << x << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }

      // /usr/include/boost/math/special_functions/chebyshev.hpp:164:40: error:
      // cannot convert
      // ‘boost::boost::math::differentiation::autodiff_v1::detail::fvar<double,
      // 3>’ to ‘double’ in return
      // BOOST_REQUIRE_EQUAL(boost::math::chebyshev_clenshaw_recurrence(c.data(),c.size(),make_fvar<T,m>(0.20))
      // ,
      // boost::math::chebyshev_clenshaw_recurrence(c.data(),c.size(),static_cast<T>(0.20)));
      /*try {
        std::array<T, 4> c0{{14.2, -13.7, 82.3, 96}};
        BOOST_REQUIRE_CLOSE_FRACTION(boost::math::chebyshev_clenshaw_recurrence(c0.data(),
      c0.size(), make_fvar<T,m>(x)),
                                     boost::math::chebyshev_clenshaw_recurrence(c0.data(),
      c0.size(), x), 10*std::numeric_limits<T>::epsilon()); } catch (...) {
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }*/
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(cospi_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::cos_pi(make_fvar<T, m>(x)), boost::math::cos_pi(x), test_constants::pct_epsilon);
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(boost::math::cos_pi(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(boost::math::cos_pi(x), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(digamma_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::digamma(make_fvar<T, m>(x));
      auto anchor_v = boost::math::digamma(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::digamma(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::digamma(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::digamma(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::digamma(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_1_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.2, 1.2};
  test_detail::RandomSample<T> phi_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_1(make_fvar<T, m>(k)), boost::math::ellint_1(k),
                          50 * test_constants::pct_epsilon);
      BOOST_REQUIRE_CLOSE(boost::math::ellint_1(make_fvar<T, m>(k), make_fvar<T, m>(phi)),
                          boost::math::ellint_1(k, phi), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_1(make_fvar<T, m>(k)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_1(k), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_1(make_fvar<T, m>(k)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_1(k), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_2_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.2, 1.2};
  test_detail::RandomSample<T> phi_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_2(make_fvar<T, m>(k)), boost::math::ellint_2(k),
                          50 * test_constants::pct_epsilon);
      BOOST_REQUIRE_CLOSE(boost::math::ellint_2(make_fvar<T, m>(k), make_fvar<T, m>(phi)),
                          boost::math::ellint_2(k, phi), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_2(make_fvar<T, m>(k)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_2(k), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_2(make_fvar<T, m>(k)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_2(k), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_3_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.2, 1.2};
  test_detail::RandomSample<T> n_sampler{-2000, 2000};
  test_detail::RandomSample<T> phi_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto n = n_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_3(make_fvar<T, m>(k), make_fvar<T, m>(n)), boost::math::ellint_3(k, n),
                          50 * test_constants::pct_epsilon);
      BOOST_REQUIRE_CLOSE(boost::math::ellint_3(make_fvar<T, m>(k), make_fvar<T, m>(n), make_fvar<T, m>(phi)),
                          boost::math::ellint_3(k, n, phi), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_3(make_fvar<T, m>(k), make_fvar<T, m>(n)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_3(k, n), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_3(make_fvar<T, m>(k), make_fvar<T, m>(n)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_3(k, n), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Inputs: k: " << k << "  n: " << n << "  phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_d_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.20, 1.20};
  test_detail::RandomSample<T> phi_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_d(make_fvar<T, m>(k)), boost::math::ellint_d(k),
                          50 * test_constants::pct_epsilon);
      BOOST_REQUIRE_CLOSE(boost::math::ellint_d(make_fvar<T, m>(k), make_fvar<T, m>(phi)),
                          boost::math::ellint_d(k, phi), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_d(make_fvar<T, m>(k)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_d(k), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_d(make_fvar<T, m>(k)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_d(k), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Inputs: k: " << k << "  phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rf_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};
  test_detail::RandomSample<T> z_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    auto z = z_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_rf(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::math::ellint_rf(x, y, z), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rf(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rf(x, y, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rf(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rf(x, y, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cerr << "Inputs: x: " << x << "  y: " << y << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rc_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_rc(make_fvar<T, m>(x), make_fvar<T, m>(y)), boost::math::ellint_rc(x, y),
                          50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rc(make_fvar<T, m>(x), make_fvar<T, m>(y)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rc(x, y), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rc(make_fvar<T, m>(x), make_fvar<T, m>(y)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rc(x, y), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cerr << "Inputs: x: " << x << "  y: " << y << std::endl;
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rj_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};
  test_detail::RandomSample<T> z_sampler{-2000, 2000};
  test_detail::RandomSample<T> p_sampler{-2000, 2000};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    auto z = z_sampler.next();
    auto p = p_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(
          boost::math::ellint_rj(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z), make_fvar<T, m>(p)),
          boost::math::ellint_rj(x, y, z, p), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &e) {
      BOOST_REQUIRE_THROW(
          boost::math::ellint_rj(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z), make_fvar<T, m>(p)),
          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rj(x, y, z, p), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &e) {
      BOOST_REQUIRE_THROW(
          boost::math::ellint_rj(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z), make_fvar<T, m>(p)),
          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rj(x, y, z, p), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << "  y: " << y << "  z: " << z << "  p: " << p << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rd_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};
  test_detail::RandomSample<T> z_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    auto z = z_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_rd(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::math::ellint_rd(x, y, z), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rd(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rd(x, y, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rd(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rd(x, y, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << "  y: " << y << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rg_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};
  test_detail::RandomSample<T> z_sampler{-2000, 2000};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    auto z = z_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_rg(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::math::ellint_rg(x, y, z), 50 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rg(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rg(x, y, z), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::ellint_rg(make_fvar<T, m>(x), make_fvar<T, m>(y), make_fvar<T, m>(z)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::ellint_rg(x, y, z), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << "  y: " << y << "  z: " << z << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(erf_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(erf(make_fvar<T, m>(x)), boost::math::erf(x), 200 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((erf(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      BOOST_REQUIRE_THROW(boost::math::erf(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((erf(make_fvar<T, m>(x)), std::fetestexcept(FE_OVERFLOW)));
      BOOST_REQUIRE_THROW(boost::math::erf(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(erfc(make_fvar<T, m>(x)), boost::math::erfc(x), 200 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((erfc(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      BOOST_REQUIRE_THROW(boost::math::erfc(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      std::feclearexcept(FE_ALL_EXCEPT);
      BOOST_REQUIRE((erfc(make_fvar<T, m>(x)), std::fetestexcept(FE_OVERFLOW)));
      BOOST_REQUIRE_THROW(boost::math::erfc(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(expint_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{1, 500};
  for (auto n : boost::irange<unsigned>(test_constants::n_samples)) {
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::expint(n, make_fvar<T, m>(x)), boost::math::expint(n, x),
                          200 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::expint(n, make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::expint(n, x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::expint(n, make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::expint(n, x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: n: " << n << " x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    for (auto y : {-1, 1}) {
      try {
        BOOST_REQUIRE_CLOSE(boost::math::expint(make_fvar<T, m>(x * y)), boost::math::expint(x * y),
                            200 * test_constants::pct_epsilon);
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::expint(make_fvar<T, m>(x * y)), boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::expint(x * y), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        BOOST_REQUIRE_THROW(boost::math::expint(make_fvar<T, m>(x * y)), boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::expint(x * y), boost::wrapexcept<std::overflow_error>);
      } catch (...) {
        std::cout << "Input: x: " << x << " y: " << y << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(expm1_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-boost::math::log1p(2000), boost::math::log1p(2000)};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::expm1(make_fvar<T, m>(x)), boost::math::expm1(x), test_constants::pct_epsilon);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::expm1(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::expm1(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(factorials_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 1000};
  for (auto i : boost::irange<unsigned>(test_constants::n_samples)) {
    try {
      auto fact_i = boost::math::factorial<T>(i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_EQUAL(autodiff_v, fact_i);
    } catch (...) {
      std::cout << "Input: i: " << i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto fact_i = boost::math::unchecked_factorial<T>(i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_EQUAL(autodiff_v, fact_i);
    } catch (...) {
      std::cout << "Input: i: " << i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto fact_i = boost::math::unchecked_factorial<T>(i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_EQUAL(autodiff_v, fact_i);
    } catch (...) {
      std::cout << "Input: i: " << i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto fact_i = boost::math::double_factorial<T>(i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_EQUAL(autodiff_v, fact_i);
    } catch (...) {
      std::cout << "Input: i: " << i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    auto x = x_sampler.next();
    try {
      auto fact_i = boost::math::rising_factorial<T>(x, i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_EQUAL(autodiff_v, fact_i);
    } catch (...) {
      std::cout << "Input: x: " << x << " i: " << i << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto fact_i = boost::math::falling_factorial<T>(x, test_constants::n_samples - i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_EQUAL(autodiff_v, fact_i);
    } catch (...) {
      std::cout << "Input: x: " << x << " i: " << (test_constants::n_samples - i) << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fpclassify_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-1000, 1000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(0)), static_cast<T>(FP_ZERO));
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(10)), static_cast<T>(FP_NORMAL));
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(std::numeric_limits<T>::infinity())),
                        static_cast<T>(FP_INFINITE));
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(std::numeric_limits<T>::quiet_NaN())),
                        static_cast<T>(FP_NAN));
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(std::numeric_limits<T>::denorm_min())),
                        static_cast<T>(FP_SUBNORMAL));

    BOOST_REQUIRE(boost::math::isfinite(make_fvar<T, m>(0)));
    BOOST_REQUIRE(boost::math::isnormal(make_fvar<T, m>((std::numeric_limits<T>::min)())));
    BOOST_REQUIRE(!boost::math::isnormal(make_fvar<T, m>(std::numeric_limits<T>::denorm_min())));
    BOOST_REQUIRE(boost::math::isinf(make_fvar<T, m>(std::numeric_limits<T>::infinity())));
    BOOST_REQUIRE(boost::math::isnan(make_fvar<T, m>(std::numeric_limits<T>::quiet_NaN())));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(gamma_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 1000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::tgamma(make_fvar<T, m>(x));
      auto anchor_v = boost::math::tgamma(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma(x), boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma(x), boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::tgamma1pm1(make_fvar<T, m>(x));
      auto anchor_v = boost::math::tgamma1pm1(x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(x), boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(make_fvar<T, m>(x)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(x), boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    int s1 = 0;
    int s2 = 0;

    try {
      BOOST_REQUIRE_CLOSE(boost::math::lgamma(make_fvar<T, m>(x), std::addressof(s1)),
                          boost::math::lgamma(x, std::addressof(s2)), 50 * test_constants::pct_epsilon);
      BOOST_REQUIRE((std::addressof(s1) == nullptr && std::addressof(s2) == nullptr) || (s1 == s2));
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lgamma(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lgamma(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::lgamma(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::lgamma(x), boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::lgamma(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::lgamma(x), boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    auto x2 = x_sampler.next();
    try {
      auto autodiff_v = boost::math::tgamma_lower(make_fvar<T, m>(x), make_fvar<T, m>(x2));
      auto anchor_v = boost::math::tgamma_lower(x, x2);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma_lower(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma_lower(x, x2), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma_lower(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma_lower(x, x2), boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::tgamma_lower(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::tgamma_lower(x, x2), boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << " x2: " << x2 << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::gamma_q(make_fvar<T, m>(x), make_fvar<T, m>(x2));
      auto anchor_v = boost::math::gamma_q(x, x2);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q(x, x2), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q(x, x2), boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q(x, x2), boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << " x2: " << x2 << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::gamma_p(make_fvar<T, m>(x), make_fvar<T, m>(x2));
      auto anchor_v = boost::math::gamma_p(x, x2);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p(x, x2), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p(x, x2), boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p(make_fvar<T, m>(x), make_fvar<T, m>(x2)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p(x, x2), boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << " x2: " << x2 << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    auto x_normalized = x / (x_sampler.dist_.max() - x_sampler.dist_.min());
    auto x2_normalized = x2 / (x_sampler.dist_.max() - x_sampler.dist_.min());
    try {
      auto autodiff_v = boost::math::gamma_p_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized));
      auto anchor_v = boost::math::gamma_p_inv(x_normalized, x2_normalized);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(x_normalized, x2_normalized), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(x_normalized, x2_normalized),
                          boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(x_normalized, x2_normalized),
                          boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x_normalized << " x2: " << x2_normalized << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::gamma_q_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized));
      auto anchor_v = boost::math::gamma_q_inv(x_normalized, x2_normalized);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(x_normalized, x2_normalized), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(x_normalized, x2_normalized),
                          boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(x_normalized, x2_normalized),
                          boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x_normalized << " x2: " << x2_normalized << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::gamma_p_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized));
      auto anchor_v = boost::math::gamma_p_inva(x_normalized, x2_normalized);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(x_normalized, x2_normalized), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(x_normalized, x2_normalized),
                          boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(x_normalized, x2_normalized),
                          boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x_normalized << " x2: " << x2_normalized << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::gamma_q_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized));
      auto anchor_v = boost::math::gamma_q_inva(x_normalized, x2_normalized);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(x_normalized, x2_normalized), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(x_normalized, x2_normalized),
                          boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
                          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(x_normalized, x2_normalized),
                          boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x_normalized << " x2: " << x2_normalized << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::gamma_p_derivative(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized));
      auto anchor_v = boost::math::gamma_p_derivative(x_normalized, x2_normalized);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(
          boost::math::gamma_p_derivative(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_derivative(x_normalized, x2_normalized),
                          boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(
          boost::math::gamma_p_derivative(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_derivative(x_normalized, x2_normalized),
                          boost::wrapexcept<std::overflow_error>);
    } catch (const boost::math::evaluation_error &) {
      BOOST_REQUIRE_THROW(
          boost::math::gamma_p_derivative(make_fvar<T, m>(x_normalized), make_fvar<T, m>(x2_normalized)),
          boost::wrapexcept<boost::math::evaluation_error>);
      BOOST_REQUIRE_THROW(boost::math::gamma_p_derivative(x_normalized, x2_normalized),
                          boost::wrapexcept<boost::math::evaluation_error>);
    } catch (...) {
      std::cout << "Input: x: " << x_normalized << " x2: " << x2_normalized << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(heuman_lambda_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-1.01, 1.01};
  test_detail::RandomSample<T> phi_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::heuman_lambda(make_fvar<T, m>(x), make_fvar<T, m>(phi)),
                          boost::math::heuman_lambda(x, phi), 10000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::heuman_lambda(make_fvar<T, m>(x), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::heuman_lambda(x, phi), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << "  "
                << "phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(hermite_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-200, 200};
  for (auto i : boost::irange(test_constants::n_samples)) {
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE_FRACTION(boost::math::hermite(i, make_fvar<T, m>(x)), boost::math::hermite(i, x),
                                   10000 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::hermite(i, make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::hermite(i, x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::hermite(i, make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::hermite(i, x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(hypot_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::hypot(make_fvar<T, m>(x), make_fvar<T, m>(y)), boost::math::hypot(x, y),
                          2 * test_constants::pct_epsilon);
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(boost::math::hypot(make_fvar<T, m>(x), make_fvar<T, m>(y)),
                          boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(boost::math::hypot(x, y), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << "  y: " << y << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(jacobi_zeta_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2, 2};
  test_detail::RandomSample<T> phi_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE_FRACTION(boost::math::jacobi_zeta(make_fvar<T, m>(x), make_fvar<T, m>(phi)),
                                   boost::math::jacobi_zeta(x, phi), 100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_zeta(make_fvar<T, m>(x), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_zeta(x, phi), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << "  "
                << "phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(laguerre_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<unsigned> n_sampler{0, 50};
  test_detail::RandomSample<unsigned> r_sampler{0, 50};
  test_detail::RandomSample<T> x_sampler{-50, 50};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto n = n_sampler.next();
    n_sampler.dist_.param(typename test_detail::RandomSample<unsigned>::dist_t::param_type(0, m));
    auto r = r_sampler.next();
    auto x = x_sampler.next();

    try {
      auto autodiff_v = boost::math::laguerre(n, make_fvar<T, m>(x));
      auto anchor_v = boost::math::laguerre(n, x);
      if (!std::isfinite(static_cast<T>(autodiff_v)) || !std::isfinite(anchor_v)) {
        BOOST_REQUIRE(!std::isfinite(static_cast<T>(autodiff_v)) && !std::isfinite(anchor_v));
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(((boost::math::laguerre(n, make_fvar<T, m>(x)))), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::laguerre(n, x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(((boost::math::laguerre(n, make_fvar<T, m>(x)))), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::laguerre(n, x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: n: " << n << " x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::laguerre(n, r, make_fvar<T, m>(x));
      auto anchor_v = boost::math::laguerre(n, r, x);
      if (!std::isfinite(static_cast<T>(autodiff_v)) || !std::isfinite(anchor_v)) {
        BOOST_REQUIRE(!std::isfinite(static_cast<T>(autodiff_v)) && !std::isfinite(anchor_v));
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(((boost::math::laguerre(n, r, make_fvar<T, m>(x)))), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::laguerre(n, r, x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(((boost::math::laguerre(n, r, make_fvar<T, m>(x)))), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::laguerre(n, r, x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: n: " << n << " r: " << r << " x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(lambert_w_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{static_cast<T>(-1 / std::exp(-1)), std::numeric_limits<T>::max()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE_FRACTION(boost::math::lambert_w0(make_fvar<T, m>(x)), boost::math::lambert_w0(x),
                                   100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_w0(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_w0(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE_FRACTION(boost::math::lambert_wm1(make_fvar<T, m>(x)), boost::math::lambert_wm1(x),
                                   100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE_FRACTION(boost::math::lambert_w0_prime(make_fvar<T, m>(x)), boost::math::lambert_w0_prime(x),
                                   100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_w0_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_w0_prime(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE_FRACTION(boost::math::lambert_wm1_prime(make_fvar<T, m>(x)),
                                   boost::math::lambert_wm1_prime(x), 100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1_prime(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(log1p_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-1, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE_FRACTION(boost::math::log1p(make_fvar<T, m>(x)), boost::math::log1p(x),
                                   10 * std::numeric_limits<T>::epsilon());
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(boost::math::log1p(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(boost::math::log1p(x), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(owens_t_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> h_sampler{-2000, 2000};
  test_detail::RandomSample<T> a_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto h = h_sampler.next();
    auto a = a_sampler.next();
    try {
      auto autodiff_v = boost::math::owens_t(make_fvar<T, m>(h), make_fvar<T, m>(a));
      auto anchor_v = boost::math::owens_t(h, a);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::owens_t(make_fvar<T, m>(h), make_fvar<T, m>(a)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::owens_t(h, a), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::owens_t(make_fvar<T, m>(h), make_fvar<T, m>(a)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::owens_t(h, a), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: h: " << h << "\ta: " << a << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pow_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  for (auto i : boost::irange(10)) {
    BOOST_REQUIRE_CLOSE(boost::math::pow<0>(make_fvar<T, m>(i)), boost::math::pow<0>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<1>(make_fvar<T, m>(i)), boost::math::pow<1>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<2>(make_fvar<T, m>(i)), boost::math::pow<2>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<3>(make_fvar<T, m>(i)), boost::math::pow<3>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<4>(make_fvar<T, m>(i)), boost::math::pow<4>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<5>(make_fvar<T, m>(i)), boost::math::pow<5>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<6>(make_fvar<T, m>(i)), boost::math::pow<6>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<7>(make_fvar<T, m>(i)), boost::math::pow<7>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<8>(make_fvar<T, m>(i)), boost::math::pow<8>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
    BOOST_REQUIRE_CLOSE(boost::math::pow<9>(make_fvar<T, m>(i)), boost::math::pow<9>(static_cast<T>(i)),
                        test_constants::pct_epsilon);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(polygamma_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::polygamma(i, make_fvar<T, m>(x));
      auto anchor_v = boost::math::polygamma(i, x);
      if  (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(autodiff_v - anchor_v), std::numeric_limits<T>::epsilon());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * std::numeric_limits<T>::epsilon());
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::polygamma(i, make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::polygamma(i, x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::polygamma(i, make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::polygamma(i, x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: i: " << i << "\tx: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(powm1_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 20};
  test_detail::RandomSample<T> y_sampler{-200, 200};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    try {
      auto autodiff_v = boost::math::powm1(make_fvar<T, m>(x), make_fvar<T, m>(y));
      auto anchor_v = boost::math::powm1(x, y);
      if (!std::isfinite(static_cast<T>(autodiff_v)) || !std::isfinite(anchor_v)) {
        BOOST_REQUIRE(!std::isfinite(static_cast<T>(autodiff_v)) && !std::isfinite(anchor_v));
      } else {
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 25 * test_constants::pct_epsilon);
      }
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::powm1(make_fvar<T, m>(x), make_fvar<T, m>(y)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::powm1(x, y), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::powm1(make_fvar<T, m>(x), make_fvar<T, m>(y)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::powm1(x, y), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << "  y: " << y << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sin_pi_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::sin_pi(make_fvar<T, m>(x)), boost::math::sin_pi(x), test_constants::pct_epsilon);
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(boost::math::sin_pi(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(boost::math::sin_pi(x), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sinhc_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::sinhc_pi(make_fvar<T, m>(x));
      auto anchor_v = boost::math::sinhc_pi(x);
      if (!std::isfinite(static_cast<T>(autodiff_v)) || !std::isfinite(anchor_v)) {
        BOOST_REQUIRE(!std::isfinite(static_cast<T>(autodiff_v)) && !std::isfinite(anchor_v));
      } else {
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, test_constants::pct_epsilon);
      }
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(boost::math::sinhc_pi(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(boost::math::sinhc_pi(x), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sqrt1pm1_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-1, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::sqrt1pm1(make_fvar<T, m>(x)), boost::math::sqrt1pm1(x),
                          test_constants::pct_epsilon);
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(boost::math::sqrt1pm1(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(boost::math::sqrt1pm1(x), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(trigamma_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::trigamma(make_fvar<T, m>(x)), boost::math::trigamma(x),
                          (x < static_cast<T>(0) ? 220 : 20) * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::trigamma(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::trigamma(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::trigamma(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::trigamma(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(zeta_hpp, T, testing_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::zeta(make_fvar<T, m>(x)), boost::math::zeta(x),
                          (x < static_cast<T>(1) ? 220 : 20) * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::zeta(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::zeta(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::zeta(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::zeta(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()