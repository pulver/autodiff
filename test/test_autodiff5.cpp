//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

using namespace boost::math::differentiation;

/*********************************************************************************************************************
 * special functions tests
 *********************************************************************************************************************/

BOOST_AUTO_TEST_SUITE(test_autodiff_5)

BOOST_AUTO_TEST_CASE_TEMPLATE(airy_hpp, T, multiprecision_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler(-100, 100);
  for (auto i : boost::irange(test_constants::n_samples)) {
    const auto &x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::airy_ai(make_fvar<T, m>(x));
      auto anchor_v = boost::math::airy_ai(x);
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
        if (test_detail::check_if_small(autodiff_v, anchor_v)) {
          BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
        } else {
          BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
        if (test_detail::check_if_small(autodiff_v, anchor_v)) {
          BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
        } else {
          BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(acosh_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-100, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = T(x_sampler.next());
    try {
      auto autodiff_v = acosh(make_fvar<T, m>(x));
      auto anchor_v = acosh(x);
      if (!isfinite(static_cast<T>(autodiff_v)) || !isfinite(anchor_v)) {
        BOOST_REQUIRE(!isfinite(static_cast<T>(autodiff_v)) && !isfinite(anchor_v));
      } else if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
      }
    } catch (const std::domain_error &) {
      if (bmp::is_number<T>::value || bmp::is_number_expression<T>::value) {
        BOOST_REQUIRE_THROW(acosh(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      } else {
        std::feclearexcept(FE_ALL_EXCEPT);
        BOOST_REQUIRE((acosh(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      }
      BOOST_REQUIRE_THROW(acosh(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      if (bmp::is_number<T>::value || bmp::is_number_expression<T>::value) {
        BOOST_REQUIRE_THROW(acosh(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      } else {
        std::feclearexcept(FE_ALL_EXCEPT);
        BOOST_REQUIRE((acosh(make_fvar<T, m>(x)), std::fetestexcept(FE_OVERFLOW)));
      }
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asinh_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-100, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = asinh(make_fvar<T, m>(x));
      auto anchor_v = asinh(x);
      if (!isfinite(static_cast<T>(autodiff_v)) || !isfinite(anchor_v)) {
        BOOST_REQUIRE(!isfinite(static_cast<T>(autodiff_v)) && !isfinite(anchor_v));
      } else if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
      }
    } catch (const std::domain_error &) {
      if (bmp::is_number<T>::value || bmp::is_number_expression<T>::value) {
        BOOST_REQUIRE_THROW(asinh(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      } else {
        std::feclearexcept(FE_ALL_EXCEPT);
        BOOST_REQUIRE((asinh(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      }
      BOOST_REQUIRE_THROW(asinh(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      if (bmp::is_number<T>::value || bmp::is_number_expression<T>::value) {
        BOOST_REQUIRE_THROW(asinh(make_fvar<T, m>(x)), boost::wrapexcept<std::overflow_error>);
      } else {
        std::feclearexcept(FE_ALL_EXCEPT);
        BOOST_REQUIRE((asinh(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      }
      BOOST_REQUIRE_THROW(asinh(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atanh_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-100, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = atanh(make_fvar<T, m>(x));
      auto anchor_v = atanh(x);
      if (!isfinite(static_cast<T>(autodiff_v)) || !isfinite(anchor_v)) {
        BOOST_REQUIRE(!isfinite(static_cast<T>(autodiff_v)) && !isfinite(anchor_v));
      } else if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
      }
    } catch (const std::domain_error &) {
      if (bmp::is_number<T>::value || bmp::is_number_expression<T>::value) {
        BOOST_REQUIRE_THROW(atanh(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      } else {
        std::feclearexcept(FE_ALL_EXCEPT);
        BOOST_REQUIRE((atanh(make_fvar<T, m>(x)), std::fetestexcept(FE_INVALID)));
      }
      BOOST_REQUIRE_THROW(atanh(x), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      if (bmp::is_number<T>::value || bmp::is_number_expression<T>::value) {
        BOOST_REQUIRE_THROW(atanh(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      } else {
        std::feclearexcept(FE_ALL_EXCEPT);
        BOOST_REQUIRE((atanh(make_fvar<T, m>(x)), std::fetestexcept(FE_OVERFLOW)));
      }
      BOOST_REQUIRE_THROW(atanh(x), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atan2_function, T, all_float_types) {
  BOOST_MATH_STD_USING;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    try {
      auto autodiff_v = atan2(make_fvar<T, m>(x), make_fvar<T, m>(y));
      auto anchor_v = atan2(x, y);
      BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, boost::math::tools::epsilon<T>());
    } catch (...) {
      std::cout << "Input: x: " << x << " y: " << y << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bernoulli_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  for (auto i : boost::irange(test_constants::n_samples)) {
    try {
      auto autodiff_v = boost::math::bernoulli_b2n<autodiff_fvar<T, m>>(i);
      auto anchor_v = boost::math::bernoulli_b2n<T>(i);
      if (!isfinite(static_cast<T>(autodiff_v)) || !isfinite(anchor_v)) {
        BOOST_REQUIRE(!isfinite(static_cast<T>(autodiff_v)) && !isfinite(anchor_v));
      } else {
        BOOST_REQUIRE_EQUAL(autodiff_v, anchor_v);
      }
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
      auto autodiff_v = boost::math::tangent_t2n<autodiff_fvar<T, m>>(i);
      auto anchor_v = boost::math::tangent_t2n<T>(i);
      if (!isfinite(static_cast<T>(autodiff_v)) || !isfinite(anchor_v)) {
        BOOST_REQUIRE(!isfinite(static_cast<T>(autodiff_v)) && !isfinite(anchor_v));
      } else {
        BOOST_REQUIRE_EQUAL(autodiff_v, anchor_v);
      }
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

// TODO(kbhat): Something in here is very slow with boost::multiprecision
BOOST_AUTO_TEST_CASE_TEMPLATE(bessel_hpp, T, bin_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> v_sampler{-20, 20};
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(beta_hpp, T, all_float_types) {
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 100000 * boost::math::tools::epsilon<T>());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(binomial_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<unsigned> n_sampler{0u, 30};
  test_detail::RandomSample<unsigned> r_sampler{0u, 30};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto n = n_sampler.next();
    auto r = (std::min)(r_sampler.next(), n - 1);
    try {
      auto autodiff_v = boost::math::binomial_coefficient<autodiff_fvar<T, m>>(n, r);
      auto anchor_v = boost::math::binomial_coefficient<T>(n, r);
      if (isfinite(static_cast<T>(autodiff_v)) && isfinite(anchor_v)) {
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, test_constants::pct_epsilon());
      } else {
        BOOST_REQUIRE(!(isfinite(static_cast<T>(autodiff_v)) || isfinite(anchor_v)));
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

BOOST_AUTO_TEST_CASE_TEMPLATE(cbrt_hpp, T, all_float_types) {
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
        BOOST_REQUIRE_CLOSE(boost::math::cbrt(make_fvar<T, m>(x)), boost::math::cbrt(x), test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(chebyshev_hpp, T, all_float_types) {
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
                                     4000 * boost::math::tools::epsilon<T>());
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
                                     4000 * boost::math::tools::epsilon<T>());
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
                                     boost::math::chebyshev_t_prime(n, x), 4000 * boost::math::tools::epsilon<T>());
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
      c0.size(), x), 10*boost::math::tools::epsilon<T>()); } catch (...) {
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }*/
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(cospi_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::cos_pi(make_fvar<T, m>(x)), boost::math::cos_pi(x), test_constants::pct_epsilon());
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(boost::math::cos_pi(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(boost::math::cos_pi(x), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(digamma_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::digamma(make_fvar<T, m>(x));
      auto anchor_v = boost::math::digamma(x);
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
        BOOST_REQUIRE_SMALL(static_cast<T>(fabs(autodiff_v - anchor_v)), boost::math::tools::epsilon<T>());
      } else {
        BOOST_REQUIRE_CLOSE_FRACTION(autodiff_v, anchor_v, 200000 * boost::math::tools::epsilon<T>());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_1_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.2, 1.2};
  test_detail::RandomSample<T> phi_sampler{-boost::math::constants::two_pi<T>(), boost::math::constants::two_pi<T>()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_1(make_fvar<T, m>(k)), boost::math::ellint_1(k),
                          50 * test_constants::pct_epsilon());
      BOOST_REQUIRE_CLOSE(boost::math::ellint_1(make_fvar<T, m>(k), make_fvar<T, m>(phi)),
                          boost::math::ellint_1(k, phi), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_2_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.2, 1.2};
  test_detail::RandomSample<T> phi_sampler{-boost::math::constants::two_pi<T>(), boost::math::constants::two_pi<T>()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_2(make_fvar<T, m>(k)), boost::math::ellint_2(k),
                          50 * test_constants::pct_epsilon());
      BOOST_REQUIRE_CLOSE(boost::math::ellint_2(make_fvar<T, m>(k), make_fvar<T, m>(phi)),
                          boost::math::ellint_2(k, phi), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_3_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.2, 1.2};
  test_detail::RandomSample<T> n_sampler{-2000, 2000};
  test_detail::RandomSample<T> phi_sampler{-boost::math::constants::two_pi<T>(), boost::math::constants::two_pi<T>()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto n = n_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_3(make_fvar<T, m>(k), make_fvar<T, m>(n)), boost::math::ellint_3(k, n),
                          50 * test_constants::pct_epsilon());
      BOOST_REQUIRE_CLOSE(boost::math::ellint_3(make_fvar<T, m>(k), make_fvar<T, m>(n), make_fvar<T, m>(phi)),
                          boost::math::ellint_3(k, n, phi), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_d_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-1.20, 1.20};
  test_detail::RandomSample<T> phi_sampler{-boost::math::constants::two_pi<T>(), boost::math::constants::two_pi<T>()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto phi = phi_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::ellint_d(make_fvar<T, m>(k)), boost::math::ellint_d(k),
                          50 * test_constants::pct_epsilon());
      BOOST_REQUIRE_CLOSE(boost::math::ellint_d(make_fvar<T, m>(k), make_fvar<T, m>(phi)),
                          boost::math::ellint_d(k, phi), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rf_hpp, T, all_float_types) {
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
                          boost::math::ellint_rf(x, y, z), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rc_hpp, T, all_float_types) {
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
                          50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rj_hpp, T, all_float_types) {
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
          boost::math::ellint_rj(x, y, z, p), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rd_hpp, T, all_float_types) {
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
                          boost::math::ellint_rd(x, y, z), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(ellint_rg_hpp, T, all_float_types) {
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
                          boost::math::ellint_rg(x, y, z), 50 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(erf_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(erf(make_fvar<T, m>(x)), boost::math::erf(x), 200 * test_constants::pct_epsilon());
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
      BOOST_REQUIRE_CLOSE(erfc(make_fvar<T, m>(x)), boost::math::erfc(x), 200 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_CASE_TEMPLATE(expint_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{1, 500};
  for (auto n : boost::irange<unsigned>(test_constants::n_samples)) {
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::expint(n, make_fvar<T, m>(x)), boost::math::expint(n, x),
                          200 * test_constants::pct_epsilon());
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
                            200 * test_constants::pct_epsilon());
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

BOOST_AUTO_TEST_SUITE_END()