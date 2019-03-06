#include "test_autodiff.hpp"

BOOST_AUTO_TEST_SUITE(test_autodiff_6)

BOOST_AUTO_TEST_CASE_TEMPLATE(expm1_hpp, T, all_float_types) {
  using boost::multiprecision::log;
  using std::log;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-log(T(2000)), log(T(2000))};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::expm1(make_fvar<T, m>(x)), boost::math::expm1(x),
                        5 * 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(factorials_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 28};
  for (auto i : boost::irange<unsigned>(test_constants::n_samples)) {
    {
      auto fact_i = boost::math::factorial<T>(i);
      auto autodiff_v = boost::math::factorial<autodiff_fvar<T, m>>(i);
      BOOST_REQUIRE_CLOSE(autodiff_v, fact_i, 100 * std::numeric_limits<T>::epsilon());
    }

    {
      auto fact_i = boost::math::unchecked_factorial<T>(i);
      auto autodiff_v = boost::math::unchecked_factorial<autodiff_fvar<T, m>>(i);
      BOOST_REQUIRE_CLOSE(autodiff_v, fact_i, 100 * std::numeric_limits<T>::epsilon());
    }

    {
      auto fact_i = boost::math::unchecked_factorial<T>(i);
      auto autodiff_v = boost::math::unchecked_factorial<autodiff_fvar<T, m>>(i);
      BOOST_REQUIRE_CLOSE(autodiff_v, fact_i, 100 * std::numeric_limits<T>::epsilon());
    }

    {
      auto fact_i = boost::math::double_factorial<T>(i);
      auto autodiff_v = boost::math::double_factorial<autodiff_fvar<T, m>>(i);
      BOOST_REQUIRE_CLOSE(autodiff_v, fact_i, 100 * std::numeric_limits<T>::epsilon());
    }

    auto x = x_sampler.next();
    {
      auto fact_i = boost::math::rising_factorial<T>(x, i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_CLOSE(autodiff_v, fact_i, 100 * std::numeric_limits<T>::epsilon());
    }

    {
      auto fact_i = boost::math::falling_factorial<T>(x, test_constants::n_samples - i);
      auto autodiff_v = make_fvar<T, m>(fact_i);
      BOOST_REQUIRE_CLOSE(autodiff_v, fact_i, 100 * std::numeric_limits<T>::epsilon());
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fpclassify_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-1000, 1000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;

    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(0)), FP_ZERO);
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(10)), FP_NORMAL);
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(std::numeric_limits<T>::infinity())), FP_INFINITE);
    BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(std::numeric_limits<T>::quiet_NaN())), FP_NAN);
    if (std::numeric_limits<T>::has_denorm != std::denorm_absent) {
      BOOST_REQUIRE_EQUAL(boost::math::fpclassify(make_fvar<T, m>(std::numeric_limits<T>::denorm_min())), FP_SUBNORMAL);
    }

    BOOST_REQUIRE(boost::math::isfinite(make_fvar<T, m>(0)));
    BOOST_REQUIRE(boost::math::isnormal(make_fvar<T, m>((std::numeric_limits<T>::min)())));
    BOOST_REQUIRE(!boost::math::isnormal(make_fvar<T, m>(std::numeric_limits<T>::denorm_min())));
    BOOST_REQUIRE(boost::math::isinf(make_fvar<T, m>(std::numeric_limits<T>::infinity())));
    BOOST_REQUIRE(boost::math::isnan(make_fvar<T, m>(std::numeric_limits<T>::quiet_NaN())));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(gamma_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> z_sampler{0, 34};
  test_detail::RandomSample<T> a_sampler{0, 34};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto z = z_sampler.next();
    {
      try {
        auto autodiff_v = boost::math::tgamma(make_fvar<T, m>(z));
        auto anchor_v = boost::math::tgamma(z);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::tgamma(make_fvar<T, m>(z)), boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma(z), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow z: " << z << std::endl;
        BOOST_REQUIRE_THROW(boost::math::tgamma(make_fvar<T, m>(z)), boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma(z), boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::tgamma(make_fvar<T, m>(z)), boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma(z), boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: z: " << z << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      try {
        auto autodiff_v = boost::math::tgamma1pm1(make_fvar<T, m>(z));
        auto anchor_v = boost::math::tgamma1pm1(z);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());

      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(make_fvar<T, m>(z)), boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(z), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow z: " << z << std::endl;
        BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(make_fvar<T, m>(z)), boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(z), boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(make_fvar<T, m>(z)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma1pm1(z), boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: z: " << z << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }

    {
      int s1 = 0;
      int s2 = 0;
      try {
        BOOST_REQUIRE_CLOSE(boost::math::lgamma(make_fvar<T, m>(z), std::addressof(s1)),
                            boost::math::lgamma(z, std::addressof(s2)), 5000 * 100 * std::numeric_limits<T>::epsilon());
        BOOST_REQUIRE((std::addressof(s1) == nullptr && std::addressof(s2) == nullptr) || (s1 == s2));
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::lgamma(make_fvar<T, m>(z)), boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::lgamma(z), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow z: " << z << std::endl;
        BOOST_REQUIRE_THROW(boost::math::lgamma(make_fvar<T, m>(z)), boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::lgamma(z), boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::lgamma(make_fvar<T, m>(z)), boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::lgamma(z), boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: z: " << z << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      auto a = boost::math::nextafter(a_sampler.next(), ((std::numeric_limits<T>::max))());
      try {
        auto autodiff_v = boost::math::tgamma_lower(make_fvar<T, m>(a), make_fvar<T, m>(z));
        auto anchor_v = boost::math::tgamma_lower(a, z);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());

      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::tgamma_lower(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma_lower(a, z), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a << " z: " << z << std::endl;
        BOOST_REQUIRE_THROW(boost::math::tgamma_lower(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma_lower(a, z), boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::tgamma_lower(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::tgamma_lower(a, z), boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a << " z: " << z << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      auto a = boost::math::nextafter(a_sampler.next(), ((std::numeric_limits<T>::max))());
      try {
        auto autodiff_v = boost::math::gamma_q(make_fvar<T, m>(a), make_fvar<T, m>(z));
        auto anchor_v = boost::math::gamma_q(a, z);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());

      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_q(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q(a, z), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a << " z: " << z << std::endl;
        BOOST_REQUIRE_THROW(boost::math::gamma_q(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q(a, z), boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_q(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q(a, z), boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a << " z: " << z << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      auto a = boost::math::nextafter(a_sampler.next(), ((std::numeric_limits<T>::max))());
      try {
        auto autodiff_v = boost::math::gamma_p(make_fvar<T, m>(a), make_fvar<T, m>(z));
        auto anchor_v = boost::math::gamma_p(a, z);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());

      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_p(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p(a, z), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a << " z: " << z << std::endl;
        BOOST_REQUIRE_THROW(boost::math::gamma_p(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p(a, z), boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_p(make_fvar<T, m>(a), make_fvar<T, m>(z)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p(a, z), boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a << " z: " << z << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    auto z_normalized = z / (((z_sampler.dist_.max))() - ((z_sampler.dist_.min))());
    {
      auto a_normalized = a_sampler.next() / (((a_sampler.dist_.max))() - ((a_sampler.dist_.min))());
      try {
        auto autodiff_v = boost::math::gamma_p_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized));
        auto anchor_v = boost::math::gamma_p_inv(a_normalized, z_normalized);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());

      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(a_normalized, z_normalized), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a_normalized << " z: " << z_normalized << std::endl;
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(a_normalized, z_normalized),
                            boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inv(a_normalized, z_normalized),
                            boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a_normalized << " z: " << z_normalized << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      auto a_normalized = a_sampler.next() / (((a_sampler.dist_.max))() - ((a_sampler.dist_.min))());
      try {
        auto autodiff_v = boost::math::gamma_q_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized));
        auto anchor_v = boost::math::gamma_q_inv(a_normalized, z_normalized);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(a_normalized, z_normalized), boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a_normalized << " z: " << z_normalized << std::endl;
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(a_normalized, z_normalized),
                            boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inv(a_normalized, z_normalized),
                            boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a_normalized << " z: " << z_normalized << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      auto a_normalized = a_sampler.next() / (((a_sampler.dist_.max))() - ((a_sampler.dist_.min))());
      try {
        auto autodiff_v = boost::math::gamma_p_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized));
        auto anchor_v = boost::math::gamma_p_inva(a_normalized, z_normalized);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(a_normalized, z_normalized),
                            boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a_normalized << " z: " << z_normalized << std::endl;
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(a_normalized, z_normalized),
                            boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_inva(a_normalized, z_normalized),
                            boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a_normalized << " z: " << z_normalized << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      auto a_normalized = a_sampler.next() / (((a_sampler.dist_.max))() - ((a_sampler.dist_.min))());
      try {
        auto autodiff_v = boost::math::gamma_q_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized));
        auto anchor_v = boost::math::gamma_q_inva(a_normalized, z_normalized);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(a_normalized, z_normalized),
                            boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a_normalized << " z: " << z_normalized << std::endl;
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(a_normalized, z_normalized),
                            boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
                            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_q_inva(a_normalized, z_normalized),
                            boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a_normalized << " z: " << z_normalized << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
    {
      auto a_normalized = a_sampler.next() / (((a_sampler.dist_.max))() - ((a_sampler.dist_.min))());
      try {
        auto autodiff_v = boost::math::gamma_p_derivative(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized));
        auto anchor_v = boost::math::gamma_p_derivative(a_normalized, z_normalized);
        BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 5000 * 100 * std::numeric_limits<T>::epsilon());
      } catch (const std::domain_error &) {
        BOOST_REQUIRE_THROW(
            boost::math::gamma_p_derivative(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
            boost::wrapexcept<std::domain_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_derivative(a_normalized, z_normalized),
                            boost::wrapexcept<std::domain_error>);
      } catch (const std::overflow_error &) {
        std::cout << "Overflow a: " << a_normalized << " z: " << z_normalized << std::endl;
        BOOST_REQUIRE_THROW(
            boost::math::gamma_p_derivative(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
            boost::wrapexcept<std::overflow_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_derivative(a_normalized, z_normalized),
                            boost::wrapexcept<std::overflow_error>);
      } catch (const boost::math::evaluation_error &) {
        std::cout << "Overflow a: " << a_normalized << " z: " << z_normalized << std::endl;
        BOOST_REQUIRE_THROW(
            boost::math::gamma_p_derivative(make_fvar<T, m>(a_normalized), make_fvar<T, m>(z_normalized)),
            boost::wrapexcept<boost::math::evaluation_error>);
        BOOST_REQUIRE_THROW(boost::math::gamma_p_derivative(a_normalized, z_normalized),
                            boost::wrapexcept<boost::math::evaluation_error>);
      } catch (...) {
        std::cout << std::setprecision(20) << "Input: a: " << a_normalized << " z: " << z_normalized << std::endl;
        std::rethrow_exception(std::exception_ptr(std::current_exception()));
      }
    }
  }
}

// Requires pow(complex<autodiff_fvar<T,m>>, T)
/* BOOST_AUTO_TEST_CASE_TEMPLATE(hankel_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> v_sampler{-200, 200};
  test_detail::RandomSample<T> x_sampler{-200, 200};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto v = v_sampler.next();
    auto x = x_sampler.next();

    try {
      auto autodiff_v = boost::math::cyl_hankel_1(make_fvar<T,m>(v), make_fvar<T,m>(x));
      auto anchor_v = boost::math::cyl_hankel_1(v, x);
      BOOST_REQUIRE_CLOSE(autodiff_v.real(), anchor_v.real(),
                                   100 * std::numeric_limits<T>::epsilon());
      BOOST_REQUIRE_CLOSE(autodiff_v.imag(), anchor_v.imag(),
                                   100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_hankel_1(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::domain_error>); BOOST_REQUIRE_THROW(boost::math::cyl_hankel_1(v, x),
boost::wrapexcept<std::domain_error>); } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_hankel_1(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::overflow_error>); BOOST_REQUIRE_THROW(boost::math::cyl_hankel_1(v, x),
boost::wrapexcept<std::overflow_error>); } catch (...) { std::cout << std::setprecision(20) << "Input: x: " << x<< "
max: "<< std::numeric_limits<T>::max() << std::endl;
std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::cyl_hankel_2(make_fvar<T,m>(v), make_fvar<T,m>(x));
      auto anchor_v = boost::math::cyl_hankel_2(v, x);
      BOOST_REQUIRE_CLOSE(autodiff_v.real(), anchor_v.real(),
                                   100 * std::numeric_limits<T>::epsilon());
      BOOST_REQUIRE_CLOSE(autodiff_v.imag(), anchor_v.imag(),
                                   100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_hankel_2(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::domain_error>); BOOST_REQUIRE_THROW(boost::math::cyl_hankel_2(v, x),
boost::wrapexcept<std::domain_error>); } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::cyl_hankel_2(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::overflow_error>); BOOST_REQUIRE_THROW(boost::math::cyl_hankel_2(v, x),
boost::wrapexcept<std::overflow_error>); } catch (...) { std::cout << std::setprecision(20) << "Input: x: " << x<< "
max: "<< std::numeric_limits<T>::max() << std::endl;
std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::sph_hankel_1(make_fvar<T,m>(v), make_fvar<T,m>(x));
      auto anchor_v = boost::math::sph_hankel_1(v, x);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                                   100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::sph_hankel_1(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::domain_error>); BOOST_REQUIRE_THROW(boost::math::sph_hankel_1(v, x),
boost::wrapexcept<std::domain_error>); } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::sph_hankel_1(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::overflow_error>); BOOST_REQUIRE_THROW(boost::math::sph_hankel_1(v, x),
boost::wrapexcept<std::overflow_error>); } catch (...) { std::cout << std::setprecision(20) << "Input: x: " << x<< "
max: "<< std::numeric_limits<T>::max() << std::endl;
std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      auto autodiff_v = boost::math::sph_hankel_2(make_fvar<T,m>(v), make_fvar<T,m>(x));
      auto anchor_v = boost::math::sph_hankel_2(v, x);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v,
                                   100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::sph_hankel_2(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::domain_error>); BOOST_REQUIRE_THROW(boost::math::sph_hankel_2(v, x),
boost::wrapexcept<std::domain_error>); } catch (const std::overflow_error &) {
      BOOST_REQUIRE_THROW(boost::math::sph_hankel_2(make_fvar<T,m>(v), make_fvar<T,m>(x)),
boost::wrapexcept<std::overflow_error>); BOOST_REQUIRE_THROW(boost::math::sph_hankel_2(v, x),
boost::wrapexcept<std::overflow_error>); } catch (...) { std::cout << std::setprecision(20) << "Input: x: " << x<< "
max: "<< std::numeric_limits<T>::max() << std::endl;
std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }
  }
} */

BOOST_AUTO_TEST_CASE_TEMPLATE(hermite_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-200, 200};
  for (auto i : boost::irange(14)) {
    auto x = x_sampler.next();
    auto autodiff_v = boost::math::hermite(i, make_fvar<T, m>(x));
    auto anchor_v = boost::math::hermite(i, x);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(heuman_lambda_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-1, 1};
  test_detail::RandomSample<T> phi_sampler{-boost::math::constants::two_pi<T>(), boost::math::constants::two_pi<T>()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto phi = phi_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::heuman_lambda(make_fvar<T, m>(x), make_fvar<T, m>(phi)),
                        boost::math::heuman_lambda(x, phi), 5 * 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(hypot_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  test_detail::RandomSample<T> y_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto y = y_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::hypot(make_fvar<T, m>(x), make_fvar<T, m>(y)), boost::math::hypot(x, y),
                        100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(jacobi_elliptic_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> k_sampler{0, 1};
  test_detail::RandomSample<T> theta_sampler{-100, 100};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto k = k_sampler.next();
    auto theta = theta_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_cd(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_cd(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_cn(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_cn(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_cs(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_cs(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_dc(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_dc(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_dn(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_dn(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_ds(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_ds(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_nc(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_nc(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_nd(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_nd(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_ns(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_ns(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_sc(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_sc(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_sd(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_sd(k, theta), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_sn(make_fvar<T, m>(k), make_fvar<T, m>(theta)),
                        boost::math::jacobi_sn(k, theta), 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(jacobi_zeta_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-1, 1};
  test_detail::RandomSample<T> phi_sampler{-boost::math::constants::two_pi<T>(), boost::math::constants::two_pi<T>()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    auto phi = phi_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::jacobi_zeta(make_fvar<T, m>(x), make_fvar<T, m>(phi)),
                        boost::math::jacobi_zeta(x, phi), 5 * 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(laguerre_hpp, T, all_float_types) {
  BOOST_MATH_STD_USING
  using boost::multiprecision::min;
  using std::min;

  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<unsigned> n_sampler{1, 50};
  test_detail::RandomSample<unsigned> r_sampler{0, 50};
  test_detail::RandomSample<T> x_sampler{0, 50};

  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto n = n_sampler.next();
    auto r = ((min))(n - 1, r_sampler.next());
    auto x = x_sampler.next();

    {
      auto autodiff_v = boost::math::laguerre(n, make_fvar<T, m>(x));
      auto anchor_v = boost::math::laguerre(n, x);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 100 * std::numeric_limits<T>::epsilon());
    }
    {
      auto autodiff_v = boost::math::laguerre(n, r, make_fvar<T, m>(x));
      auto anchor_v = boost::math::laguerre(n, r, x);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 100 * std::numeric_limits<T>::epsilon());
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(lambert_w_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{static_cast<T>(-1 / std::exp(-1)), ((std::numeric_limits<T>::max))()};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    try {
      BOOST_REQUIRE_CLOSE(boost::math::lambert_w0(make_fvar<T, m>(x)), boost::math::lambert_w0(x),
                          100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_w0(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_w0(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << std::setprecision(20) << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::lambert_wm1(make_fvar<T, m>(x)), boost::math::lambert_wm1(x),
                          100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << std::setprecision(20) << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::lambert_w0_prime(make_fvar<T, m>(x)), boost::math::lambert_w0_prime(x),
                          100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_w0_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_w0_prime(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << std::setprecision(20) << "Input: x: " << x << std::endl;
      std::rethrow_exception(std::exception_ptr(std::current_exception()));
    }

    try {
      BOOST_REQUIRE_CLOSE(boost::math::lambert_wm1_prime(make_fvar<T, m>(x)), boost::math::lambert_wm1_prime(x),
                          100 * std::numeric_limits<T>::epsilon());
    } catch (const std::domain_error &) {
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1_prime(make_fvar<T, m>(x)), boost::wrapexcept<std::domain_error>);
      BOOST_REQUIRE_THROW(boost::math::lambert_wm1_prime(x), boost::wrapexcept<std::domain_error>);
    } catch (...) {
      std::cout << std::setprecision(20) << "Input: x: " << x << std::endl;
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
    BOOST_REQUIRE_CLOSE(boost::math::log1p(make_fvar<T, m>(x)), boost::math::log1p(x),
                        100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(next_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  for (auto i : boost::irange(test_constants::n_samples)) {
    BOOST_REQUIRE_CLOSE(boost::math::float_next(make_fvar<T, m>(i)), boost::math::float_next(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_prior(make_fvar<T, m>(i)), boost::math::float_prior(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());

    BOOST_REQUIRE_CLOSE(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(1)),
                        boost::math::nextafter(static_cast<T>(i), static_cast<T>(1)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(i + 2)),
                        boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(i + 2)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(i + 1)),
                        boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(i + 2)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(-1)),
                        boost::math::nextafter(static_cast<T>(i), static_cast<T>(-1)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(-1 * (i + 2))),
                        boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(-1 * (i + 2))),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(-1 * (i + 1))),
                        boost::math::nextafter(make_fvar<T, m>(i), static_cast<T>(-1 * (i + 2))),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::nextafter(make_fvar<T, m>(i), make_fvar<T, m>(i)), ((make_fvar<T, m>(i))),
                        100 * std::numeric_limits<T>::epsilon());

    BOOST_REQUIRE_CLOSE(boost::math::float_advance(make_fvar<T, m>(i), 1),
                        boost::math::float_advance(static_cast<T>(i), 1), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_advance(make_fvar<T, m>(i), i + 2),
                        boost::math::float_advance(make_fvar<T, m>(i), i + 2), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_advance(make_fvar<T, m>(i), i + 1),
                        boost::math::float_advance(boost::math::float_advance(make_fvar<T, m>(i), i + 2), -1),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_advance(make_fvar<T, m>(i), -1),
                        boost::math::float_advance(static_cast<T>(i), -1), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_advance(make_fvar<T, m>(i), -i - 2),
                        boost::math::float_advance(static_cast<T>(i), -i - 2), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_advance(make_fvar<T, m>(i), -i - 1),
                        boost::math::float_advance(boost::math::float_advance(make_fvar<T, m>(i), -i - 2), 1),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_advance(make_fvar<T, m>(i), 0), ((make_fvar<T, m>(i))),
                        100 * std::numeric_limits<T>::epsilon());

    BOOST_REQUIRE_CLOSE(boost::math::float_distance(make_fvar<T, m>(i), static_cast<T>(i)), static_cast<T>(0),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_distance(boost::math::float_next(make_fvar<T, m>(i)), make_fvar<T, m>(i)),
                        ((make_fvar<T, m>(-1))), 100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::float_distance(boost::math::float_prior(make_fvar<T, m>(i)), make_fvar<T, m>(i)),
                        ((make_fvar<T, m>(1))), 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(owens_t_hpp, T, all_float_types) {
  BOOST_MATH_STD_USING;
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> h_sampler{-2000, 2000};
  test_detail::RandomSample<T> a_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto h = h_sampler.next();
    auto a = a_sampler.next();
    auto autodiff_v = boost::math::owens_t(make_fvar<T, m>(h), make_fvar<T, m>(a));
    auto anchor_v = boost::math::owens_t(h, a);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 20 * 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pow_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  for (auto i : boost::irange(10)) {
    BOOST_REQUIRE_CLOSE(boost::math::pow<0>(make_fvar<T, m>(i)), boost::math::pow<0>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<1>(make_fvar<T, m>(i)), boost::math::pow<1>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<2>(make_fvar<T, m>(i)), boost::math::pow<2>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<3>(make_fvar<T, m>(i)), boost::math::pow<3>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<4>(make_fvar<T, m>(i)), boost::math::pow<4>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<5>(make_fvar<T, m>(i)), boost::math::pow<5>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<6>(make_fvar<T, m>(i)), boost::math::pow<6>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<7>(make_fvar<T, m>(i)), boost::math::pow<7>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<8>(make_fvar<T, m>(i)), boost::math::pow<8>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
    BOOST_REQUIRE_CLOSE(boost::math::pow<9>(make_fvar<T, m>(i)), boost::math::pow<9>(static_cast<T>(i)),
                        100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(polygamma_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    auto x = x_sampler.next();
    auto autodiff_v = boost::math::polygamma(i, make_fvar<T, m>(x));
    auto anchor_v = boost::math::polygamma(i, x);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(powm1_hpp, T, all_float_types) {
  BOOST_MATH_STD_USING
  using boost::math::tools::max;
  using boost::multiprecision::log;
  using boost::multiprecision::min;
  using boost::multiprecision::sqrt;
  using std::log;
  using std::max;
  using std::min;
  using std::sqrt;

  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, log(((std::numeric_limits<T>::max))())};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = ((max))(x_sampler.next(), boost::math::nextafter(static_cast<T>(0), ((std::numeric_limits<T>::max))()));

    auto y = ((min))(x_sampler.next(), log(sqrt(((std::numeric_limits<T>::max))()) + 1) / log(x + 1));
    auto autodiff_v = boost::math::powm1(make_fvar<T, m>(x), make_fvar<T, m>(y));
    auto anchor_v = boost::math::powm1(x, y);
    BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 10000 * 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sin_pi_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-2000, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::sin_pi(make_fvar<T, m>(x)), boost::math::sin_pi(x),
                        100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sinhc_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-80, 80};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    if (x != 0) {
      auto autodiff_v = boost::math::sinhc_pi(make_fvar<T, m>(x));
      auto anchor_v = boost::math::sinhc_pi(x);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 100 * std::numeric_limits<T>::epsilon());
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
    auto r = (std::min)(n - 1, r_sampler.next());
    {
      auto autodiff_v = boost::math::spherical_harmonic(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi));
      auto anchor_v = boost::math::spherical_harmonic(n, r, theta, phi);
      BOOST_REQUIRE_CLOSE(autodiff_v.real(), anchor_v.real(), 500 * 100 * std::numeric_limits<T>::epsilon());
      BOOST_REQUIRE_CLOSE(autodiff_v.imag(), anchor_v.imag(), 500 * 100 * std::numeric_limits<T>::epsilon());
    }

    {
      auto autodiff_v = boost::math::spherical_harmonic_r(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi));
      auto anchor_v = boost::math::spherical_harmonic_r(n, r, theta, phi);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 500 * 100 * std::numeric_limits<T>::epsilon());
    }

    {
      auto autodiff_v = boost::math::spherical_harmonic_i(n, r, make_fvar<T, m>(theta), make_fvar<T, m>(phi));
      auto anchor_v = boost::math::spherical_harmonic_i(n, r, theta, phi);
      BOOST_REQUIRE_CLOSE(autodiff_v, anchor_v, 500 * 100 * std::numeric_limits<T>::epsilon());
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
    BOOST_REQUIRE_CLOSE(boost::math::sqrt1pm1(make_fvar<T, m>(x)), boost::math::sqrt1pm1(x),
                        100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(trigamma_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{0, 2000};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::trigamma(make_fvar<T, m>(x)), boost::math::trigamma(x),
                        100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(zeta_hpp, T, all_float_types) {
  using test_constants = test_constants_t<T>;
  static constexpr auto m = test_constants::order;
  test_detail::RandomSample<T> x_sampler{-30, 30};
  for (auto i : boost::irange(test_constants::n_samples)) {
    std::ignore = i;
    auto x = x_sampler.next();
    BOOST_REQUIRE_CLOSE(boost::math::zeta(make_fvar<T, m>(x)), boost::math::zeta(x),
                        100 * 100 * std::numeric_limits<T>::epsilon());
  }
}

BOOST_AUTO_TEST_SUITE_END()
