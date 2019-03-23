//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

using namespace boost::math::differentiation;

BOOST_AUTO_TEST_SUITE(test_autodiff_2)

BOOST_AUTO_TEST_CASE_TEMPLATE(one_over_one_plus_x_squared, T, all_float_types) {
  constexpr unsigned m = 4;
  const T cx(1);
  auto f = make_fvar<T, m>(cx);
  // f = 1 / ((f *= f) += 1);
  f *= f;
  f += T(1);
  f = f.inverse();
  BOOST_REQUIRE_EQUAL(f.derivative(0u), 0.5);
  BOOST_REQUIRE_EQUAL(f.derivative(1u), -0.5);
  BOOST_REQUIRE_EQUAL(f.derivative(2u), 0.5);
  BOOST_REQUIRE_EQUAL(f.derivative(3u), 0);
  BOOST_REQUIRE_EQUAL(f.derivative(4u), -3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(exp_test, T, all_float_types) {
  using std::exp;
  constexpr unsigned m = 4;
  const T cx = 2.0;
  const auto x = make_fvar<T, m>(cx);
  auto y = exp(x);
  for (auto i : boost::irange(m + 1)) {
    // std::cout.precision(100);
    // std::cout << "y.derivative("<<i<<") = " << y.derivative(i) << ",
    // std::exp(cx) = " << std::exp(cx) << std::endl;
    BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(i), exp(cx), std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pow, T, bin_float_types) {
  const T eps = 201 * std::numeric_limits<T>::epsilon();  // percent
  using std::exp;
  using std::log;
  using std::pow;
  constexpr unsigned m = 5;
  constexpr unsigned n = 4;
  const T cx = 2.0;
  const T cy = 3.0;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, m, n>(cy);
  auto z0 = pow(x, cy);
  BOOST_REQUIRE_EQUAL(z0.derivative(0u), pow(cx, cy));
  BOOST_REQUIRE_EQUAL(z0.derivative(1u), cy * pow(cx, cy - 1));
  BOOST_REQUIRE_EQUAL(z0.derivative(2u), cy * (cy - 1) * pow(cx, cy - 2));
  BOOST_REQUIRE_EQUAL(z0.derivative(3u), cy * (cy - 1) * (cy - 2) * pow(cx, cy - 3));
  BOOST_REQUIRE_EQUAL(z0.derivative(4u), 0u);
  BOOST_REQUIRE_EQUAL(z0.derivative(5u), 0u);
  auto z1 = pow(cx, y);
  BOOST_REQUIRE_CLOSE(z1.derivative(0u, 0u), pow(cx, cy), eps);
  for (auto j : boost::irange(1u, n + 1)) {
    BOOST_REQUIRE_CLOSE(z1.derivative(0u, j), pow(log(cx), j) * exp(cy * log(cx)), eps);
  }

  for (auto i : boost::irange(1u, m + 1)) {
    for (auto j : boost::irange(n + 1)) {
      BOOST_REQUIRE_EQUAL(z1.derivative(i, j), 0);
    }
  }

  auto z2 = pow(x, y);
  for (auto j : boost::irange(n + 1)) {
    BOOST_REQUIRE_CLOSE(z2.derivative(0u, j), pow(cx, cy) * pow(log(cx), j), eps);
  }
  for (auto j : boost::irange(n + 1)) {
    BOOST_REQUIRE_CLOSE(z2.derivative(1u, j),
                        pow(cx, cy - 1) * pow(log(cx), static_cast<int>(j) - 1) * (cy * log(cx) + j), eps);
  }
  BOOST_REQUIRE_CLOSE(z2.derivative(2u, 0u), pow(cx, cy - 2) * cy * (cy - 1), eps);
  BOOST_REQUIRE_CLOSE(z2.derivative(2u, 1u), pow(cx, cy - 2) * (cy * (cy - 1) * log(cx) + 2 * cy - 1), eps);
  for (auto j : boost::irange(2u, n + 1)) {
    BOOST_REQUIRE_CLOSE(z2.derivative(2u, j),
                        pow(cx, cy - 2) * pow(log(cx), j - 2) *
                            (j * (2 * cy - 1) * log(cx) + (j - 1) * j + (cy - 1) * cy * pow(log(cx), 2)),
                        eps);
  }
  BOOST_REQUIRE_CLOSE(
      z2.derivative(2u, 4u),
      pow(cx, cy - 2) * pow(log(cx), 2) * (4 * (2 * cy - 1) * log(cx) + (4 - 1) * 4 + (cy - 1) * cy * pow(log(cx), 2)),
      eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sqrt_test, T, all_float_types) {
  using std::pow;
  using std::sqrt;
  constexpr unsigned m = 5;
  const T cx = 4.0;
  auto x = make_fvar<T, m>(cx);
  auto y = sqrt(x);
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(0u), sqrt(cx), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(1u), 0.5 * pow(cx, -0.5), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(2u), -0.5 * 0.5 * pow(cx, -1.5), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(3u), 0.5 * 0.5 * 1.5 * pow(cx, -2.5), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(4u), -0.5 * 0.5 * 1.5 * 2.5 * pow(cx, -3.5),
                               std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(5u), 0.5 * 0.5 * 1.5 * 2.5 * 3.5 * pow(cx, -4.5),
                               std::numeric_limits<T>::epsilon());
  x = make_fvar<T, m>(0);
  y = sqrt(x);
  // std::cout << "sqrt(0) = " << y << std::endl; // (0,inf,-inf,inf,-inf,inf)
  BOOST_REQUIRE_EQUAL(y.derivative(0u), 0);
  for (auto i : boost::irange(1u, m + 1)) {
    BOOST_REQUIRE_EQUAL(y.derivative(i), (i % 2 == 1 ? 1 : -1) * std::numeric_limits<T>::infinity());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(log_test, T, all_float_types) {
  using std::log;
  using std::pow;
  constexpr unsigned m = 5;
  const T cx = 2.0;
  auto x = make_fvar<T, m>(cx);
  auto y = log(x);
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(0u), log(cx), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(1u), 1 / cx, std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(2u), -1 / pow(cx, 2), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(3u), 2 / pow(cx, 3), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(4u), -6 / pow(cx, 4), std::numeric_limits<T>::epsilon());
  BOOST_REQUIRE_CLOSE_FRACTION(y.derivative(5u), 24 / pow(cx, 5), std::numeric_limits<T>::epsilon());
  x = make_fvar<T, m>(0);
  y = log(x);
  // std::cout << "log(0) = " << y << std::endl; // log(0) =
  // depth(1)(-inf,inf,-inf,inf,-inf,inf)
  for (auto i : boost::irange(m + 1)) {
    BOOST_REQUIRE_EQUAL(y.derivative(i), (i % 2 == 1 ? 1 : -1) * std::numeric_limits<T>::infinity());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ylogx, T, all_float_types) {
  using std::log;
  using std::pow;
  const T eps = 100 * std::numeric_limits<T>::epsilon();  // percent
  constexpr unsigned m = 5;
  constexpr unsigned n = 4;
  const T cx = 2.0;
  const T cy = 3.0;
  const auto x = make_fvar<T, m>(cx);
  const auto y = make_fvar<T, m, n>(cy);
  auto z = y * log(x);
  BOOST_REQUIRE_EQUAL(z.derivative(0u, 0u), cy * log(cx));
  BOOST_REQUIRE_EQUAL(z.derivative(0u, 1u), log(cx));
  BOOST_REQUIRE_EQUAL(z.derivative(0u, 2u), 0);
  BOOST_REQUIRE_EQUAL(z.derivative(0u, 3u), 0);
  BOOST_REQUIRE_EQUAL(z.derivative(0u, 4u), 0);
  for (auto i : boost::irange(1u, m + 1)) {
    BOOST_REQUIRE_CLOSE(z.derivative(i, 0u), pow(-1, i - 1) * boost::math::factorial<T>(i - 1) * cy / pow(cx, i), eps);
    BOOST_REQUIRE_CLOSE(z.derivative(i, 1u), pow(-1, i - 1) * boost::math::factorial<T>(i - 1) / pow(cx, i), eps);
    for (auto j : boost::irange(2u, n + 1)) {
      BOOST_REQUIRE_EQUAL(z.derivative(i, j), 0u);
    }
  }
  auto z1 = exp(z);
  // RHS is confirmed by
  // https://www.wolframalpha.com/input/?i=D%5Bx%5Ey,%7Bx,2%7D,%7By,4%7D%5D+%2F.+%7Bx-%3E2.0,+y-%3E3.0%7D
  BOOST_REQUIRE_CLOSE(
      z1.derivative(2u, 4u),
      pow(cx, cy - 2) * pow(log(cx), 2) * (4 * (2 * cy - 1) * log(cx) + (4 - 1) * 4 + (cy - 1) * cy * pow(log(cx), 2)),
      eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(frexp_test, T, all_float_types) {
  using std::exp2;
  using std::frexp;
  constexpr unsigned m = 3;
  const T cx = 3.5;
  const auto x = make_fvar<T, m>(cx);
  int exp, testexp;
  auto y = frexp(x, &exp);
  BOOST_REQUIRE_EQUAL(y.derivative(0u), frexp(cx, &testexp));
  BOOST_REQUIRE_EQUAL(exp, testexp);
  BOOST_REQUIRE_EQUAL(y.derivative(1u), exp2(-exp));
  BOOST_REQUIRE_EQUAL(y.derivative(2u), 0);
  BOOST_REQUIRE_EQUAL(y.derivative(3u), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ldexp_test, T, all_float_types) {
  BOOST_MATH_STD_USING
  using boost::multiprecision::ldexp;
  constexpr auto m = 3u;
  const T cx = 3.5;
  const auto x = make_fvar<T, m>(cx);
  constexpr auto exponent = 3;
  auto y = ldexp(x, exponent);
  BOOST_REQUIRE_EQUAL(y.derivative(0u), ldexp(cx, exponent));
  BOOST_REQUIRE_EQUAL(y.derivative(1u), exp2(exponent));
  BOOST_REQUIRE_EQUAL(y.derivative(2u), 0);
  BOOST_REQUIRE_EQUAL(y.derivative(3u), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(cos_and_sin, T, bin_float_types) {
  using std::cos;
  using std::sin;
  const T eps = 200 * std::numeric_limits<T>::epsilon();  // percent
  constexpr unsigned m = 5;
  const T cx = boost::math::constants::third_pi<T>();
  const auto x = make_fvar<T, m>(cx);
  auto cos5 = cos(x);
  BOOST_REQUIRE_CLOSE(cos5.derivative(0u), cos(cx), eps);
  BOOST_REQUIRE_CLOSE(cos5.derivative(1u), -sin(cx), eps);
  BOOST_REQUIRE_CLOSE(cos5.derivative(2u), -cos(cx), eps);
  BOOST_REQUIRE_CLOSE(cos5.derivative(3u), sin(cx), eps);
  BOOST_REQUIRE_CLOSE(cos5.derivative(4u), cos(cx), eps);
  BOOST_REQUIRE_CLOSE(cos5.derivative(5u), -sin(cx), eps);
  auto sin5 = sin(x);
  BOOST_REQUIRE_CLOSE(sin5.derivative(0u), sin(cx), eps);
  BOOST_REQUIRE_CLOSE(sin5.derivative(1u), cos(cx), eps);
  BOOST_REQUIRE_CLOSE(sin5.derivative(2u), -sin(cx), eps);
  BOOST_REQUIRE_CLOSE(sin5.derivative(3u), -cos(cx), eps);
  BOOST_REQUIRE_CLOSE(sin5.derivative(4u), sin(cx), eps);
  BOOST_REQUIRE_CLOSE(sin5.derivative(5u), cos(cx), eps);
  // Test Order = 0 for codecov
  auto cos0 = cos(make_fvar<T, 0>(cx));
  BOOST_REQUIRE_CLOSE(cos0.derivative(0u), cos(cx), eps);
  auto sin0 = sin(make_fvar<T, 0>(cx));
  BOOST_REQUIRE_CLOSE(sin0.derivative(0u), sin(cx), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(acos_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using std::acos;
  using std::pow;
  using std::sqrt;
  constexpr unsigned m = 5;
  const T cx = 0.5;
  auto x = make_fvar<T, m>(cx);
  auto y = acos(x);
  BOOST_REQUIRE_CLOSE(y.derivative(0u), acos(cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1u), -1 / sqrt(1 - cx * cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2u), -cx / pow(1 - cx * cx, 1.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3u), -(2 * cx * cx + 1) / pow(1 - cx * cx, 2.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4u), -3 * cx * (2 * cx * cx + 3) / pow(1 - cx * cx, 3.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5u), -(24 * (cx * cx + 3) * cx * cx + 9) / pow(1 - cx * cx, 4.5), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(acosh_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using std::acosh;
  constexpr unsigned m = 5;
  const T cx = 2;
  auto x = make_fvar<T, m>(cx);
  auto y = acosh(x);
  // BOOST_REQUIRE_EQUAL(y.derivative(0) == acosh(cx)); // FAILS! acosh(2) is
  // overloaded for integral types
  BOOST_REQUIRE_CLOSE(y.derivative(0u), acosh(static_cast<T>(x)), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1u), 1 / boost::math::constants::root_three<T>(), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2u), -2 / (3 * boost::math::constants::root_three<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3u), 1 / boost::math::constants::root_three<T>(), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4u), -22 / (9 * boost::math::constants::root_three<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5u), 227 / (27 * boost::math::constants::root_three<T>()), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asin_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using std::asin;
  using std::pow;
  using std::sqrt;
  constexpr unsigned m = 5;
  const T cx = 0.5;
  auto x = make_fvar<T, m>(cx);
  auto y = asin(x);
  BOOST_REQUIRE_CLOSE(y.derivative(0u), asin(static_cast<T>(x)), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1u), 1 / sqrt(1 - cx * cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2u), cx / pow(1 - cx * cx, 1.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3u), (2 * cx * cx + 1) / pow(1 - cx * cx, 2.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4u), 3 * cx * (2 * cx * cx + 3) / pow(1 - cx * cx, 3.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5u), (24 * (cx * cx + 3) * cx * cx + 9) / pow(1 - cx * cx, 4.5), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asin_infinity, T, all_float_types) {
  const T eps = 100 * std::numeric_limits<T>::epsilon();  // percent
  constexpr unsigned m = 5;
  auto x = make_fvar<T, m>(1);
  auto y = asin(x);
  // std::cout << "asin(1) = " << y << std::endl; //
  // depth(1)(1.5707963267949,inf,inf,-nan,-nan,-nan)
  BOOST_REQUIRE_CLOSE(y.derivative(0u), boost::math::constants::half_pi<T>(),
                      eps);  // MacOS is not exact
  BOOST_REQUIRE_EQUAL(y.derivative(1u), std::numeric_limits<T>::infinity());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asin_derivative, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using std::pow;
  using std::sqrt;
  constexpr unsigned m = 4;
  const T cx(0.5);
  auto x = make_fvar<T, m>(cx);
  auto y = T(1) - x * x;
  BOOST_REQUIRE_EQUAL(y.derivative(0u), 1 - cx * cx);
  BOOST_REQUIRE_EQUAL(y.derivative(1u), -2 * cx);
  BOOST_REQUIRE_EQUAL(y.derivative(2u), -2);
  BOOST_REQUIRE_EQUAL(y.derivative(3u), 0);
  BOOST_REQUIRE_EQUAL(y.derivative(4u), 0);
  y = sqrt(y);
  BOOST_REQUIRE_EQUAL(y.derivative(0u), sqrt(1 - cx * cx));
  BOOST_REQUIRE_CLOSE(y.derivative(1u), -cx / sqrt(1 - cx * cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2u), -1 / pow(1 - cx * cx, 1.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3u), -3 * cx / pow(1 - cx * cx, 2.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4u), -(12 * cx * cx + 3) / pow(1 - cx * cx, 3.5), eps);
  y = y.inverse();  // asin'(x) = 1 / sqrt(1-x*x).
  BOOST_REQUIRE_CLOSE(y.derivative(0u), 1 / sqrt(1 - cx * cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1u), cx / pow(1 - cx * cx, 1.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2u), (2 * cx * cx + 1) / pow(1 - cx * cx, 2.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3u), 3 * cx * (2 * cx * cx + 3) / pow(1 - cx * cx, 3.5), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4u), (24 * (cx * cx + 3) * cx * cx + 9) / pow(1 - cx * cx, 4.5), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(asinh_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using boost::math::asinh;
  constexpr unsigned m = 5;
  const T cx = 1;
  auto x = make_fvar<T, m>(cx);
  auto y = asinh(x);
  BOOST_REQUIRE_CLOSE(y.derivative(0u), asinh(static_cast<T>(x)), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1u), 1 / boost::math::constants::root_two<T>(), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2u), -1 / (2 * boost::math::constants::root_two<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3u), 1 / (4 * boost::math::constants::root_two<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4u), 3 / (8 * boost::math::constants::root_two<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5u), -39 / (16 * boost::math::constants::root_two<T>()), eps);
}

BOOST_AUTO_TEST_SUITE_END()
