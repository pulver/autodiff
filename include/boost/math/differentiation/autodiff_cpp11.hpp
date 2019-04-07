//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

// Contributors:
//  * Kedar R. Bhat - C++11 compatibility.

// Notes:
//  * Any changes to this file should always be downstream from autodiff.cpp.
//    C++17 is a higher-level language and is easier to maintain. For example, a number of functions which are
//    lucidly read in autodiff.cpp are forced to be split into multiple structs/functions in this file for C++11.
//  * Use of typename RootType and SizeType is a hack to prevent Visual Studio 2015 from compiling functions
//    that are never called, that would otherwise produce compiler errors. Also forces functions to be inline.

#ifndef BOOST_MATH_DIFFERENTIATION_AUTODIFF_HPP
#   error "Do not #include this file directly. This should only be #included by autodiff.hpp for C++11 compatibility."
#endif

namespace boost { namespace math { namespace differentiation { inline namespace autodiff_v1 {

namespace detail {

template<typename RealType, size_t Order>
fvar<RealType,Order>::fvar(const root_type& ca, const bool is_variable)
{
  fvar_cpp11(is_fvar<RealType>{}, ca, is_variable);
}

template<typename RealType, size_t Order>
template<typename RootType>
void fvar<RealType,Order>::fvar_cpp11(std::true_type /* is_fvar */, const RootType& ca, const bool is_variable)
{
  v.front() = RealType(ca, is_variable);
  if (0 < Order)
    std::fill(v.begin()+1, v.end(), static_cast<RealType>(0));
}

template<typename RealType, size_t Order>
template<typename RootType>
void fvar<RealType,Order>::fvar_cpp11(std::false_type /* !is_fvar */, const RootType& ca, const bool is_variable)
{
  v.front() = ca;
  if (0 < Order)
  {
    v[1] = static_cast<root_type>(static_cast<int>(is_variable));
    if (1 < Order)
      std::fill(v.begin()+2, v.end(), static_cast<RealType>(0));
  }
}

template<typename RealType, size_t Order>
template<typename... Orders>
get_type_at<RealType, sizeof...(Orders)>
fvar<RealType,Order>::at_cpp11(std::true_type /* sizeof...(orders) == 0 */, size_t order, Orders... /*orders*/) const
{
  return v.at(order);
}

template<typename RealType, size_t Order>
template<typename... Orders>
get_type_at<RealType, sizeof...(Orders)>
fvar<RealType,Order>::at_cpp11(std::false_type /* sizeof...(orders) > 0 */, size_t order, Orders... orders) const
{
  return v.at(order).at(orders...);
}

// Can throw "std::out_of_range: array::at: __n (which is 7) >= _Nm (which is 7)"
template<typename RealType, size_t Order>
template<typename... Orders>
get_type_at<RealType,sizeof...(Orders)> fvar<RealType,Order>::at(size_t order, Orders... orders) const
{
  return at_cpp11(std::integral_constant<bool,sizeof...(orders)==0>{}, order, orders...);
}

template<typename T, typename... Ts>
constexpr T product(Ts... /*factors*/)
{
  return static_cast<T>(1);
}

template<typename T, typename... Ts>
constexpr T product(T factor, Ts... factors)
{
  return factor * product<T>(factors...);
}

// Can throw "std::out_of_range: array::at: __n (which is 7) >= _Nm (which is 7)"
template<typename RealType, size_t Order>
template<typename... Orders>
get_type_at<fvar<RealType,Order>,sizeof...(Orders)> fvar<RealType,Order>::derivative(Orders... orders) const
{
  static_assert(sizeof...(Orders) <= depth, "Number of parameters to derivative(...) cannot exceed fvar::depth.");
  return at(orders...) * product(boost::math::factorial<root_type>(static_cast<unsigned>(orders))...);
}

template<typename RootType, typename Func>
class Curry
{
  const Func& f;
  const size_t i;
 public:
  template <typename SizeType> // typename SizeType to force inline constructor.
  Curry(const Func& f, SizeType i):f(f),i(i) { }
  template <typename... Indices>
  RootType operator()(Indices... indices) const { return f(i,indices...); }
};

// f : order -> derivative(order)/factorial(order)
// Use this when you have the polynomial coefficients, rather than just the derivatives. E.g. See atan2().
template<typename RealType, size_t Order>
template<typename Func, typename Fvar, typename... Fvars>
promote<fvar<RealType,Order>,Fvar,Fvars...> fvar<RealType,Order>::apply_coefficients(
    const size_t order, const Func& f, const Fvar& cr, Fvars&&... fvars) const
{
  const fvar<RealType,Order> epsilon = fvar<RealType,Order>(*this).set_root(0);
  size_t i = order < order_sum ? order : order_sum;
  using return_type = promote<fvar<RealType,Order>,Fvar,Fvars...>;
  return_type accumulator = cr.apply_coefficients(
      order-i, Curry<typename return_type::root_type,Func>(f,i), std::forward<Fvars>(fvars)...);
  while (i--)
    (accumulator *= epsilon) += cr.apply_coefficients(
        order-i, Curry<typename return_type::root_type,Func>(f,i), std::forward<Fvars>(fvars)...);
  return accumulator;
}

// f : order -> derivative(order)/factorial(order)
// Use this when you have the polynomial coefficients, rather than just the derivatives. E.g. See atan2().
template<typename RealType, size_t Order>
template<typename Func, typename Fvar, typename... Fvars>
promote<fvar<RealType,Order>,Fvar,Fvars...> fvar<RealType,Order>::apply_derivatives(
    const size_t order, const Func& f, const Fvar& cr, Fvars&&... fvars) const
{
  const fvar<RealType,Order> epsilon = fvar<RealType,Order>(*this).set_root(0);
  size_t i = order < order_sum ? order : order_sum;
  using return_type = promote<fvar<RealType,Order>,Fvar,Fvars...>;
  return_type accumulator = cr.apply_derivatives(
      order-i, Curry<typename return_type::root_type,Func>(f,i), std::forward<Fvars>(fvars)...)
      / factorial<root_type>(static_cast<unsigned>(i));
  while (i--)
    (accumulator *= epsilon) += cr.apply_derivatives(
        order-i, Curry<typename return_type::root_type,Func>(f,i), std::forward<Fvars>(fvars)...)
        / factorial<root_type>(static_cast<unsigned>(i));
  return accumulator;
}

// f : order -> derivative(order)
template<typename RealType, size_t Order>
template<typename Func, typename Fvar, typename... Fvars>
promote<fvar<RealType,Order>,Fvar,Fvars...> fvar<RealType,Order>::apply_derivatives_nonhorner(
    const size_t order, const Func& f, const Fvar& cr, Fvars&&... fvars) const
{
    const fvar<RealType,Order> epsilon = fvar<RealType,Order>(*this).set_root(0);
    fvar<RealType,Order> epsilon_i = fvar<RealType,Order>(1); // epsilon to the power of i
    using return_type = promote<fvar<RealType,Order>,Fvar,Fvars...>;
    return_type accumulator = cr.apply_derivatives_nonhorner(
        order, Curry<typename return_type::root_type,Func>(f,0), std::forward<Fvars>(fvars)...);
    const size_t i_max = order < order_sum ? order : order_sum;
    for (size_t i=1 ; i<=i_max ; ++i)
    { // accumulator += (epsilon_i *= epsilon) * (f(i) / factorial<root_type>(i));
      epsilon_i = epsilon_i.epsilon_multiply(i-1, 0, epsilon, 1, 0);
      accumulator += epsilon_i.epsilon_multiply(i, 0, cr.apply_derivatives_nonhorner(
            order-i, Curry<typename return_type::root_type,Func>(f,i), std::forward<Fvars>(fvars)...)
            / factorial<root_type>(static_cast<unsigned>(i)), 0, 0);
    }
    return accumulator;
}

template<typename RealType, size_t Order>
template<typename SizeType>
fvar<RealType,Order> fvar<RealType,Order>::epsilon_multiply_cpp11(std::true_type /* is_fvar */,
                                                                  SizeType z0, size_t isum0, const fvar<RealType,Order>& cr, size_t z1, size_t isum1) const
{
  const size_t m0 = order_sum + isum0 < Order + z0 ? Order + z0 - (order_sum + isum0) : 0;
  const size_t m1 = order_sum + isum1 < Order + z1 ? Order + z1 - (order_sum + isum1) : 0;
  const size_t i_max = m0 + m1 < Order ? Order - (m0 + m1) : 0;
  fvar<RealType,Order> retval = fvar<RealType,Order>();
  for (size_t i=0, j=Order ; i<=i_max ; ++i, --j)
    retval.v[j] = epsilon_inner_product(z0, isum0, m0, cr, z1, isum1, m1, j);
  return retval;
}

template<typename RealType, size_t Order>
template<typename SizeType>
fvar<RealType,Order> fvar<RealType,Order>::epsilon_multiply_cpp11(std::false_type /* !is_fvar */,
                                                                  SizeType z0, size_t isum0, const fvar<RealType,Order>& cr, size_t z1, size_t isum1) const
{
  const RealType zero(0);
  const size_t m0 = order_sum + isum0 < Order + z0 ? Order + z0 - (order_sum + isum0) : 0;
  const size_t m1 = order_sum + isum1 < Order + z1 ? Order + z1 - (order_sum + isum1) : 0;
  const size_t i_max = m0 + m1 < Order ? Order - (m0 + m1) : 0;
  fvar<RealType,Order> retval = fvar<RealType,Order>();
  for (size_t i=0, j=Order ; i<=i_max ; ++i, --j)
    retval.v[j] = std::inner_product(v.cbegin()+m0, v.cend()-(i+m1), cr.v.crbegin()+(i+m0), zero);
  return retval;
}

template<typename RealType, size_t Order>
fvar<RealType,Order> fvar<RealType,Order>::epsilon_multiply(size_t z0, size_t isum0,
                                                            const fvar<RealType,Order>& cr, size_t z1, size_t isum1) const
{
  return epsilon_multiply_cpp11(is_fvar<RealType>{},
                                z0, isum0, cr, z1, isum1);
}

template<typename RealType, size_t Order>
template<typename SizeType>
fvar<RealType,Order> fvar<RealType,Order>::epsilon_multiply_cpp11(std::true_type /* is_fvar */,
                                                                  SizeType z0, size_t isum0, const root_type& ca) const
{
  fvar<RealType,Order> retval(*this);
  const size_t m0 = order_sum + isum0 < Order + z0 ? Order + z0 - (order_sum + isum0) : 0;
  for (size_t i=m0 ; i<=Order ; ++i)
    retval.v[i] = retval.v[i].epsilon_multiply(z0, isum0+i, ca);
  return retval;
}

template<typename RealType, size_t Order>
template<typename SizeType>
fvar<RealType,Order> fvar<RealType,Order>::epsilon_multiply_cpp11(std::false_type /* !is_fvar */,
                                                                  SizeType z0, size_t isum0, const root_type& ca) const
{
  fvar<RealType,Order> retval(*this);
  const size_t m0 = order_sum + isum0 < Order + z0 ? Order + z0 - (order_sum + isum0) : 0;
  for (size_t i=m0 ; i<=Order ; ++i)
    if (retval.v[i] != static_cast<RealType>(0))
      retval.v[i] *= ca;
  return retval;
}

template<typename RealType, size_t Order>
fvar<RealType,Order> fvar<RealType,Order>::epsilon_multiply(size_t z0, size_t isum0,
                                                            const root_type& ca) const
{
  return epsilon_multiply_cpp11(is_fvar<RealType>{}, z0, isum0, ca);
}

template<typename RealType, size_t Order>
template<typename RootType>
fvar<RealType,Order>& fvar<RealType,Order>::multiply_assign_by_root_type_cpp11(std::true_type  /* is_fvar */,
                                                                               bool is_root, const RootType& ca)
{
  auto itr = v.begin();
  itr->multiply_assign_by_root_type(is_root, ca);
  for (++itr ; itr!=v.end() ; ++itr)
    itr->multiply_assign_by_root_type(false, ca);
  return *this;
}

template<typename RealType, size_t Order>
template<typename RootType>
fvar<RealType,Order>& fvar<RealType,Order>::multiply_assign_by_root_type_cpp11(std::false_type /* !is_fvar */,
                                                                               bool is_root, const RootType& ca)
{
  auto itr = v.begin();
  if (is_root || *itr != 0)
    *itr *= ca; // Skip multiplication of 0 by ca=inf to avoid nan. Exception: root value is always multiplied.
  for (++itr ; itr!=v.end() ; ++itr)
    if (*itr != 0)
      *itr *= ca;
  return *this;
}

template<typename RealType, size_t Order>
fvar<RealType,Order>& fvar<RealType,Order>::multiply_assign_by_root_type(bool is_root, const root_type& ca)
{
  return multiply_assign_by_root_type_cpp11(is_fvar<RealType>{}, is_root, ca);
}

template<typename RealType, size_t Order>
template<typename RootType>
fvar<RealType,Order>& fvar<RealType,Order>::negate_cpp11(std::true_type /* is_fvar */, const RootType&)
{
  std::for_each(v.begin(), v.end(), [](RealType& r) { r.negate(); });
  return *this;
}

template<typename RealType, size_t Order>
template<typename RootType>
fvar<RealType,Order>& fvar<RealType,Order>::negate_cpp11(std::false_type /* !is_fvar */, const RootType&)
{
  std::for_each(v.begin(), v.end(), [](RealType& a) { a = -a; });
  return *this;
}

template<typename RealType, size_t Order>
fvar<RealType,Order>& fvar<RealType,Order>::negate()
{
  return negate_cpp11(is_fvar<RealType>{}, static_cast<root_type>(*this));
}

template<typename RealType, size_t Order>
template<typename RootType>
fvar<RealType,Order>& fvar<RealType,Order>::set_root_cpp11(std::true_type /* is_fvar */, const RootType& root)
{
  v.front().set_root(root);
  return *this;
}

template<typename RealType, size_t Order>
template<typename RootType>
fvar<RealType,Order>& fvar<RealType,Order>::set_root_cpp11(std::false_type /* !is_fvar */, const RootType& root)
{
  v.front() = root;
  return *this;
}

template<typename RealType, size_t Order>
fvar<RealType,Order>& fvar<RealType,Order>::set_root(const root_type& root)
{
  return set_root_cpp11(is_fvar<RealType>{}, root);
}

} // namespace detail

} } } } // namespace boost::math::differentiation::autodiff_v1
