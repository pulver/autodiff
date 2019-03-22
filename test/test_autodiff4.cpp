//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"
#include <boost/test/tools/floating_point_comparison.hpp>

using namespace boost::math::differentiation;

BOOST_AUTO_TEST_SUITE(test_autodiff_4)

BOOST_AUTO_TEST_CASE_TEMPLATE(lround_llround_lltrunc_truncl, T, all_float_types) {
  using boost::math::lltrunc;
  using std::llround;
  using std::lround;
  using std::truncl;

  constexpr std::size_t m = 3;
  const T& cx = static_cast<T>(3.25);
  auto x = make_fvar<T, m>(cx);
  auto yl = lround(x);
  BOOST_REQUIRE_EQUAL(yl, lround(cx));
  auto yll = llround(x);
  BOOST_REQUIRE_EQUAL(yll, llround(cx));
  BOOST_REQUIRE_EQUAL(lltrunc(cx), lltrunc(x));

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
  if constexpr (!boost::multiprecision::is_number<T>::value && !boost::multiprecision::is_number_expression<T>::value) {
    BOOST_REQUIRE_EQUAL(truncl(x), truncl(cx));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(equality, T, all_float_types) {
  BOOST_MATH_STD_USING
  using boost::math::fpclassify;
  using std::fpclassify;
  using boost::math::ulp;
  using boost::math::epsilon_difference;

  constexpr std::size_t m = 3;
  // check zeros
  {
    auto x = make_fvar<T, m>(0.0);
    auto y = T(-0.0);
    BOOST_REQUIRE_EQUAL(x, y);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multiprecision, T, multiprecision_float_types) {
  BOOST_MATH_STD_USING

  const T eps = 30 * std::numeric_limits<T>::epsilon();
  constexpr std::size_t Nw = 3;
  constexpr std::size_t Nx = 2;
  constexpr std::size_t Ny = 4;
  constexpr std::size_t Nz = 3;
  const auto w = make_fvar<T, Nw>(11);
  const auto x = make_fvar<T, 0, Nx>(12);
  const auto y = make_fvar<T, 0, 0, Ny>(13);
  const auto z = make_fvar<T, 0, 0, 0, Nz>(14);
  const auto v = mixed_partials_f(w, x, y, z);  // auto = autodiff_fvar<T,Nw,Nx,Ny,Nz>
  // Calculated from Mathematica symbolic differentiation.
  const T answer = boost::lexical_cast<T>(
      "1976.3196007477977177798818752904187209081211892187"
      "5499076582535951111845769110560421820940516423255314");
  // BOOST_REQUIRE_CLOSE(v.derivative(Nw,Nx,Ny,Nz), answer, eps); // Doesn't work for cpp_dec_float
  const T relative_error = static_cast<T>(fabs(v.derivative(Nw, Nx, Ny, Nz) / answer - 1));
  BOOST_REQUIRE_LT(relative_error, eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(airy_hpp, T, all_float_types) {

  using boost::multiprecision::min;
  using std::min;

  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler(-100, 100);
  for (auto i : boost::irange(test_constants::n_samples)) {
    const auto& x = x_sampler.next();
    {
      auto autodiff_v = boost::math::airy_ai(make_fvar<T, m>(x));
      auto anchor_v = boost::math::airy_ai(x);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                          1e4 * test_constants::pct_epsilon());
    }

    {
      auto autodiff_v = boost::math::airy_ai_prime(make_fvar<T, m>(x));
      auto anchor_v = boost::math::airy_ai_prime(x);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                          1e4 * test_constants::pct_epsilon());
    }

    {
      auto x_ = (min)(x, T(26));
      auto autodiff_v = boost::math::airy_bi(make_fvar<T, m>(x_));
      auto anchor_v = boost::math::airy_bi(x_);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                          1e4 * test_constants::pct_epsilon());
    }

    {
      auto x_ = ((min))(x, T(26));
      auto autodiff_v = boost::math::airy_bi_prime(make_fvar<T, m>(x_));
      auto anchor_v = boost::math::airy_bi_prime(x_);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                          1e4 * test_constants::pct_epsilon());
    }

    if (i > 0) {
      {
        auto autodiff_v = boost::math::airy_ai_zero<autodiff_fvar<T, m>>(i);
        auto anchor_v = boost::math::airy_ai_zero<T>(i);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                            1e4 * test_constants::pct_epsilon());
      }

      {
        auto autodiff_v = boost::math::airy_bi_zero<autodiff_fvar<T, m>>(i);
        auto anchor_v = boost::math::airy_bi_zero<T>(i);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                            1e4 * test_constants::pct_epsilon());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(acosh_hpp, T, all_float_types) {
  using boost::math::acosh;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{1, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto autodiff_v = acosh(make_fvar<T, m>(x));
    auto anchor_v = acosh(x);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                        1e3 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asinh_hpp, T, all_float_types) {
  using boost::math::asinh;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-100, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();

    auto autodiff_v = asinh(make_fvar<T, m>(x));
    auto anchor_v = asinh(x);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                        1e3 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atanh_hpp, T, all_float_types) {
  using boost::math::nextafter;
  using std::nextafter;

  using boost::math::atanh;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-1, 1};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = nextafter(x_sampler.next(), T(0));

    auto autodiff_v = atanh(make_fvar<T, m>(x));
    auto anchor_v = atanh(x);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                        1e3 * test_constants::pct_epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atan_hpp, T, all_float_types) {
  using std::atan;
  using boost::multiprecision::atan;
  using boost::math::differentiation::detail::atan;
  using boost::math::signbit;
  using boost::multiprecision::signbit;
  using boost::math::fpclassify;
  using boost::multiprecision::fpclassify;
  using boost::math::float_prior;

  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-1, 1};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = T(1);
    while(fpclassify(T(abs(x)-1)) == FP_ZERO) {
      x = x_sampler.next();
    }

    auto autodiff_v = atan(make_fvar<T, m>(x));
    auto anchor_v = atan(x);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                        1e3 * test_constants::pct_epsilon());
  }
}

/*
BOOST_AUTO_TEST_CASE_TEMPLATE(atan2_function, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;

  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();

    auto autodiff_v = atan2(make_fvar<T, m>(x), make_fvar<T, m>(y));
    auto anchor_v = atan2(x, y);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 200 * test_constants::pct_epsilon());
  }
}
*/

BOOST_AUTO_TEST_SUITE_END()
