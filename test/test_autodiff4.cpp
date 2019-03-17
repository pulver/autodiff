//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

#include "test_autodiff.hpp"

BOOST_AUTO_TEST_SUITE(test_autodiff_4)

BOOST_AUTO_TEST_CASE_TEMPLATE(lround_llround_lltrunc_truncl, T, all_float_types) {
  using std::llround;
  using std::lround;
  using std::truncl;
  using boost::math::lltrunc;

  constexpr int m = 3;
  const T cx = 3.25;
  auto x = make_fvar<T, m>(cx);
  long yl = lround(x);
  BOOST_REQUIRE_EQUAL(yl, lround(cx));
  long long yll = llround(x);
  BOOST_REQUIRE_EQUAL(yll, llround(cx));
  BOOST_REQUIRE_EQUAL(lltrunc(cx), lltrunc(x));

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
  if constexpr (!boost::multiprecision::is_number<T>::value && !boost::multiprecision::is_number_expression<T>::value) {
    BOOST_REQUIRE_EQUAL(truncl(x), truncl(cx));
  }
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multiprecision, T, multiprecision_float_types) {
  const T eps = 30 * std::numeric_limits<T>::epsilon();
  constexpr int Nw = 3;
  constexpr int Nx = 2;
  constexpr int Ny = 4;
  constexpr int Nz = 3;
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
  using std::fabs;
  const T relative_error = static_cast<T>(fabs(v.derivative(Nw, Nx, Ny, Nz) / answer - 1));
  BOOST_REQUIRE_LT(relative_error, eps);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(autodiff_tanh, T, all_float_types) {
  using std::tanh;
  using boost::multiprecision::tanh;
  using boost::math::differentiation::detail::tanh;

  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-10, 10};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(tanh(make_fvar<T, m>(x)), tanh(x),
                          test_constants::pct_epsilon());
    } catch (const boost::math::rounding_error &) {
      BOOST_REQUIRE_THROW(tanh(make_fvar<T, m>(x)), boost::wrapexcept<boost::math::rounding_error>);
      BOOST_REQUIRE_THROW(tanh(x), boost::wrapexcept<boost::math::rounding_error>);
    } catch (...) {
      std::cout << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
