//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

BOOST_AUTO_TEST_SUITE(test_autodiff_3)

BOOST_AUTO_TEST_CASE_TEMPLATE(atanh_test, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using std::atanh;
  constexpr int m = 5;
  const T cx = 0.5;
  auto x = make_fvar<T, m>(cx);
  auto y = atanh(x);
  // BOOST_REQUIRE(y.derivative(0) == atanh(cx)); // fails due to overload
  BOOST_REQUIRE(y.derivative(0) == atanh(static_cast<T>(x)));
  BOOST_REQUIRE_CLOSE(y.derivative(1), static_cast<T>(4) / 3, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2), static_cast<T>(16) / 9, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3), static_cast<T>(224) / 27, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4), static_cast<T>(1280) / 27, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5), static_cast<T>(31232) / 81, eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atan_test, T, all_float_types) {
  constexpr int m = 5;
  constexpr float cx = 1.0;
  const auto x = make_fvar<T, m>(cx);
  auto y = atan(x);
  const auto eps = test_constants_t<T>::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon();
  BOOST_REQUIRE_CLOSE(y.derivative(0), boost::math::constants::pi<T>() / 4, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1), 0.5, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2), -0.5, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3), 0.5, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4), 0.0, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5), -3.0, eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(erf_test, T, all_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using std::erf;
  using namespace boost;
  constexpr int m = 5;
  constexpr float cx = 1.0;
  const auto x = make_fvar<T, m>(cx);
  auto y = erf(x);
  BOOST_REQUIRE(y.derivative(0) == erf(static_cast<T>(x)));
  BOOST_REQUIRE_CLOSE(y.derivative(1), 2 / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2), -4 / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3), 4 / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4), 8 / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5), -40 / (math::constants::e<T>() * math::constants::root_pi<T>()), eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sinc_test, T, bin_float_types) {
  const T eps = 20000 * std::numeric_limits<T>::epsilon();  // percent
  using std::cos;
  using std::sin;
  constexpr int m = 5;
  const T cx = 1;
  auto x = make_fvar<T, m>(cx);
  auto y = sinc(x);
  BOOST_REQUIRE_CLOSE(y.derivative(0), sin(cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1), cos(cx) - sin(cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2), sin(cx) - 2 * cos(cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3), 5 * cos(cx) - 3 * sin(cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4), 13 * sin(cx) - 20 * cos(cx), eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5), 101 * cos(cx) - 65 * sin(cx), eps);
  // Test at x = 0
  auto y2 = sinc(make_fvar<T, 10>(0));
  BOOST_REQUIRE_CLOSE(y2.derivative(0), 1, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(1), 0, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(2), -cx / 3, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(3), 0, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(4), cx / 5, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(5), 0, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(6), -cx / 7, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(7), 0, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(8), cx / 9, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(9), 0, eps);
  BOOST_REQUIRE_CLOSE(y2.derivative(10), -cx / 11, eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sinh_and_cosh, T, bin_float_types) {
  const T eps = 300 * std::numeric_limits<T>::epsilon();  // percent
  using std::cosh;
  using std::sinh;
  constexpr int m = 5;
  const T cx = 1;
  auto x = make_fvar<T, m>(cx);
  auto s = sinh(x);
  auto c = cosh(x);
  BOOST_REQUIRE_CLOSE(s.derivative(0), sinh(static_cast<T>(x)), eps);
  BOOST_REQUIRE_CLOSE(c.derivative(0), cosh(static_cast<T>(x)), eps);
  for (size_t i = 0; i <= m; ++i) {
    BOOST_REQUIRE_CLOSE(s.derivative(i), static_cast<T>(i & 1 ? c : s), eps);
    BOOST_REQUIRE_CLOSE(c.derivative(i), static_cast<T>(i & 1 ? s : c), eps);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(tan_test, T, bin_float_types) {
  const T eps = 800 * std::numeric_limits<T>::epsilon();  // percent
  using std::sqrt;
  constexpr int m = 5;
  const T cx = boost::math::constants::third_pi<T>();
  const T root_three = boost::math::constants::root_three<T>();
  const auto x = make_fvar<T, m>(cx);
  auto y = tan(x);
  BOOST_REQUIRE_CLOSE(y.derivative(0), root_three, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(1), 4.0, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(2), 8 * root_three, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(3), 80.0, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(4), 352 * root_three, eps);
  BOOST_REQUIRE_CLOSE(y.derivative(5), 5824.0, eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fmod_test, T, bin_float_types) {
  constexpr int m = 3;
  constexpr float cx = 3.25;
  const T cy = 0.5;
  auto x = make_fvar<T, m>(cx);
  auto y = fmod(x, autodiff_fvar<T, m>(cy));
  BOOST_REQUIRE(y.derivative(0) == 0.25);
  BOOST_REQUIRE(y.derivative(1) == 1.0);
  BOOST_REQUIRE(y.derivative(2) == 0.0);
  BOOST_REQUIRE(y.derivative(3) == 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(round_and_trunc, T, all_float_types) {
  using std::round;
  using std::trunc;
  constexpr int m = 3;
  constexpr float cx = 3.25;
  auto x = make_fvar<T, m>(cx);
  auto y = round(x);
  BOOST_REQUIRE(y.derivative(0) == round(cx));
  BOOST_REQUIRE(y.derivative(1) == 0.0);
  BOOST_REQUIRE(y.derivative(2) == 0.0);
  BOOST_REQUIRE(y.derivative(3) == 0.0);
  y = trunc(x);
  BOOST_REQUIRE(y.derivative(0) == trunc(cx));
  BOOST_REQUIRE(y.derivative(1) == 0.0);
  BOOST_REQUIRE(y.derivative(2) == 0.0);
  BOOST_REQUIRE(y.derivative(3) == 0.0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(iround_and_itrunc, T, all_float_types) {
  using namespace boost::math;
  constexpr int m = 3;
  constexpr float cx = 3.25;
  auto x = make_fvar<T, m>(cx);
  int y = iround(x);
  BOOST_REQUIRE(y == iround(cx));
  y = itrunc(x);
  BOOST_REQUIRE(y == itrunc(cx));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(lambert_w0_test, T, all_float_types) {
  const T eps = 1000 * std::numeric_limits<T>::epsilon();  // percent
  constexpr int m = 10;
  const T cx = 3;
  // Mathematica: N[Table[D[ProductLog[x], {x, n}], {n, 0, 10}] /. x -> 3, 52]
  const char *const answers[m + 1]{"1.049908894964039959988697070552897904589466943706341",
                                   "0.1707244807388472968312949774415522047470762509741737",
                                   "-0.04336545501146252734105411312976167858858970875797718",
                                   "0.02321456264324789334313200360870492961288748451791104",
                                   "-0.01909049778427783072663170526188353869136655225133878",
                                   "0.02122935002563637629500975949987796094687564718834156",
                                   "-0.02979093848448877259041971538394953658978044986784643",
                                   "0.05051290266216717699803334605370337985567016837482099",
                                   "-0.1004503154972645060971099914384090562800544486549660",
                                   "0.2292464437392250211967939182075930820454464472006425",
                                   "-0.5905839053125614593682763387470654123192290838719517"};
  auto x = make_fvar<T, m>(cx);
  auto y = lambert_w0(x);
  for (int i = 0; i <= m; ++i) {
    const T answer = boost::lexical_cast<T>(answers[i]);
    BOOST_REQUIRE_CLOSE(y.derivative(i), answer, eps);
  }
  // const T cx0 = -1 / boost::math::constants::e<T>();
  // auto edge = lambert_w0(make_fvar<T,m>(cx0));
  // std::cout << "edge = " << edge << std::endl;
  // edge = depth(1)(-1,inf,-inf,inf,-inf,inf,-inf,inf,-inf,inf,-inf)
  // edge = depth(1)(-1,inf,-inf,inf,-inf,inf,-inf,inf,-inf,inf,-inf)
  // edge =
  // depth(1)(-1,3.68935e+19,-9.23687e+57,4.62519e+96,-2.89497e+135,2.02945e+174,-1.52431e+213,1.19943e+252,-9.75959e+290,8.14489e+329,-6.93329e+368)
}

BOOST_AUTO_TEST_SUITE_END()