#include "test_autodiff.hpp"

BOOST_AUTO_TEST_SUITE(test_autodiff_6)

BOOST_AUTO_TEST_CASE_TEMPLATE(jacobi_elliptic_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{-500, 500};
  test_detail::RandomSample<T> u_sampler{0, 500};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto u = u_sampler.next();

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_cd(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_cd(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_cd(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_cd(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_cd(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_cd(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_cn(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_cn(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_cn(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_cn(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_cn(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_cn(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_cs(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_cs(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_cs(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_cs(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_cs(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_cs(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_dc(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_dc(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_dc(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_dc(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_dc(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_dc(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_dn(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_dn(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_dn(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_dn(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_dn(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_dn(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_ds(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_ds(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_ds(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_ds(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_ds(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_ds(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_nc(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_nc(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_nc(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_nc(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_nc(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_nc(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_nd(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_nd(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_nd(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_nd(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_nd(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_nd(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_ns(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_ns(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_ns(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_ns(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_ns(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_ns(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_sc(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_sc(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_sc(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_sc(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_sc(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_sc(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_sd(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_sd(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_sd(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_sd(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_sd(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_sd(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::jacobi_sn(make_fvar<T, m>(k), make_fvar<T, m>(u)), boost::math::jacobi_sn(k, u),
                          50000 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_sn(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_sn(k, u), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::jacobi_sn(make_fvar<T, m>(k), make_fvar<T, m>(u)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::jacobi_sn(k, u), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: k: " << k << "  "
                << "u: " << u << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(jacobi_zeta_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(laguerre_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(lambert_w_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(log1p_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(next_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  for (auto i : boost::irange(test_constants::n_samples)) {
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_next(make_fvar<T, m>(i)),
                                 boost::math::float_next(static_cast<T>(i)),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_prior(make_fvar<T, m>(i)),
                                 boost::math::float_prior(static_cast<T>(i)),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());

    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(1)),
                                 boost::math::nextafter(static_cast<T>(i), static_cast<T>(1)),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(i + 2)),
                                 boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(i + 2)),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(i + 1)),
                                 boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(i + 2)),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(-1)),
                                 boost::math::nextafter(static_cast<T>(i), static_cast<T>(-1)),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(-1 * (i + 2))),
                                 boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(-1 * (i + 2))),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(-1 * (i + 1))),
                                 boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(-1 * (i + 2))),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(i)), ((make_fvar<T, m>(i))),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());

    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_advance(make_fvar<T, m>(i), 1),
                                 boost::math::float_advance(static_cast<T>(i), 1),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_advance(make_fvar<T, m>(i), i + 2),
                                 boost::math::float_advance(make_fvar<T, m>(i), i + 2),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_advance(make_fvar<T, m>(i), i + 1),
                                 boost::math::float_advance(boost::math::float_advance(make_fvar<T, m>(i), i + 2), -1),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_advance(make_fvar<T, m>(i), -1),
                                 boost::math::float_advance(static_cast<T>(i), -1),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_advance(make_fvar<T, m>(i), -i - 2),
                                 boost::math::float_advance(static_cast<T>(i), -i - 2),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_advance(make_fvar<T, m>(i), -i - 1),
                                 boost::math::float_advance(boost::math::float_advance(make_fvar<T, m>(i), -i - 2), 1),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_advance(make_fvar<T, m>(i), 0), ((make_fvar<T, m>(i))),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());

    BOOST_REQUIRE_CLOSE_FRACTION(boost::math::float_distance(make_fvar<T, m>(i), static_cast<T>(i)), static_cast<T>(0),
                                 test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(
        boost::math::float_distance(boost::math::float_next(make_fvar<T, m>(i)), make_fvar<T, m>(i)),
        ((make_fvar<T, m>(-1))), test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE_FRACTION(
        boost::math::float_distance(boost::math::float_prior(make_fvar<T, m>(i)), make_fvar<T, m>(i)),
        ((make_fvar<T, m>(1))), test_constants::mp_epsilon_multiplier * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(owens_t_hpp, T, all_float_types) {
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
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(pow_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(polygamma_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    auto x = x_sampler.next();
    try {
      auto autodiff_v = boost::math::polygamma(i, make_fvar<T, m>(x));
      auto anchor_v = boost::math::polygamma(i, x);
      if (test_detail::check_if_small(autodiff_v, anchor_v)) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(powm1_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(sin_pi_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(sinhc_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(spherical_harmonic_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> theta_sampler{0, boost::math::constants::pi<T>()};
  test_detail::RandomSample<T> phi_sampler{0, boost::math::constants::two_pi<T>()};
  test_detail::RandomSample<int> r_sampler{0, test_constants::n_samples};
  for (auto n : boost::irange(test_constants::n_samples)) {
    auto theta = theta_sampler.next();
    auto phi = phi_sampler.next();
    auto r = (std::min<unsigned>)(n - 1, r_sampler.next());
    try {
      auto autodiff_v = boost::math::spherical_harmonic(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi));
      auto anchor_v = boost::math::spherical_harmonic(n, r, theta, phi);
      BOOST_REQUIRE_CLOSE(autodiff_v.real(), anchor_v.real(), 5 * test_constants::pct_epsilon);
      BOOST_REQUIRE_CLOSE(autodiff_v.imag(), anchor_v.imag(), 5 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic(n, r, theta, phi), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic(n, r, theta, phi), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: n: " << n << " r: " << r << " theta: " << theta << " phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::spherical_harmonic_r(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi));
      auto anchor_v = boost::math::spherical_harmonic_r(n, r, theta, phi);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_r(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_r(n, r, theta, phi), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_r(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_r(n, r, theta, phi), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: n: " << n << " r: " << r << " theta: " << theta << " phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::spherical_harmonic_i(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi));
      auto anchor_v = boost::math::spherical_harmonic_i(n, r, theta, phi);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5 * test_constants::pct_epsilon);
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_i(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_i(n, r, theta, phi), boost::wrapexcept<std::domain_error>);
    } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_i(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi)),
                          boost::wrapexcept<std::overflow_error>);
      BOOST_REQUIRE_THROW(boost::math::spherical_harmonic_i(n, r, theta, phi), boost::wrapexcept<std::overflow_error>);
    } catch (...) {
      std::cout << "Input: n: " << n << " r: " << r << " theta: " << theta << " phi: " << phi << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sqrt1pm1_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(trigamma_hpp, T, all_float_types) {
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

BOOST_AUTO_TEST_CASE_TEMPLATE(zeta_hpp, T, all_float_types) {
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
