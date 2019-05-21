//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

// THIS IS NOT MEANT TO BE COMPILED - ONLY FOR INCLUSION IN DOCUMENTATION.

#include <boost/math/differentiation/autodiff.hpp>

namespace boost {
namespace math {
namespace differentiation {

// Function returning a single variable of differentiation. Recommended: Use auto for type.
template <typename RealType, size_t Order, size_t... Orders>
autodiff_fvar<RealType, Order, Orders...> make_fvar(RealType const& ca);

// Function returning multiple independent variables of differentiation.
template<typename RealType, size_t... Orders, typename... RealTypes>
auto make_ftuple(RealTypes const&... ca);

// Type of combined autodiff types. Recommended: Use auto for return type (C++14).
template <typename RealType, typename... RealTypes>
using promote = typename detail::promote_args_n<RealType, RealTypes...>::type;

namespace detail {

// Single autodiff variable. Independent variables are created by make_ftuple.
template <typename RealType, size_t Order>
class fvar {
 public:
  // Query return value of function to get the derivatives.
  template <typename... Orders>
  get_type_at<RealType, sizeof...(Orders) - 1> derivative(Orders... orders) const;

  // All of the arithmetic and comparison operators are overloaded.
  template <typename RealType2, size_t Order2>
  fvar& operator+=(fvar<RealType2, Order2> const&);

  fvar& operator+=(root_type const&);

  // ...
};

// Standard math functions are overloaded and called via argument-dependent lookup (ADL).
template <typename RealType, size_t Order>
fvar<RealType, Order> floor(fvar<RealType, Order> const&);

template <typename RealType, size_t Order>
fvar<RealType, Order> exp(fvar<RealType, Order> const&);

// ...

}  // namespace detail

}  // namespace differentiation
}  // namespace math
}  // namespace boost
/**/
