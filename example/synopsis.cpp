//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

// THIS IS NOT MEANT TO BE COMPILED - ONLY FOR INCLUSION IN DOCUMENTATION.

#include <boost/math/differentiation/autodiff.hpp>

namespace boost { namespace math { namespace differentiation {

// Use to determine combined autodiff type.
template<typename RealType, typename... RealTypes>
using promote = typename detail::promote_args_n<RealType,RealTypes...>::type;

// Single-parameter constructor initializes a constant.
template<typename RealType, size_t Order, size_t... Orders>
using autodiff_fvar = typename detail::nest_fvar<RealType,Order,Orders...>::type;

// Returns a variable of differentiation.
template<typename RealType, size_t Order, size_t... Orders>
autodiff_fvar<RealType,Order,Orders...> make_fvar(const RealType& ca);

namespace detail {

// Each nested level corresponds to an independent variable.
template<typename RealType, size_t Order>
class fvar
{
  public:

    // Query return value of function to get the derivatives.
    template<typename... Orders>
    get_type_at<RealType, sizeof...(Orders)-1> derivative(Orders... orders) const;

    // All of the arithmetic operators are overloaded.
    template<typename RealType2, size_t Order2>
    fvar& operator+=(const fvar<RealType2,Order2>&);

    fvar& operator+=(const root_type&);

    // ...
};

// Standard math functions are overloaded and called via argument-dependent lookup (ADL).
template<typename RealType, size_t Order>
fvar<RealType,Order> floor(const fvar<RealType,Order>&);

template<typename RealType, size_t Order>
fvar<RealType,Order> exp(const fvar<RealType,Order>&);

// ...

} // namespace detail

} } } // namespace boost::math::differentiation
/**/
