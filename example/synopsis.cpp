//           Copyright Matthew Pulver 2018 - 2019.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

// THIS IS NOT MEANT TO BE COMPILED - ONLY FOR INCLUSION IN DOCUMENTATION.

#include <boost/math/differentiation/autodiff.hpp>

namespace boost { namespace math { namespace differentiation { namespace autodiff {

// The primary template alias for instantiating autodiff variables.
template<typename RealType,size_t Order,size_t... Orders>
using variable = typename nested_dimensions<RealType,Order,Orders...>::type;

// Satisfies Boost's Conceptual Requirements for Real Number Types.
// Don't use this dimension<> class directly. Instead, use the variable<> alias.
template<typename RealType,size_t Order>
class dimension
{
  public:

    // Initialize a variable of differentiation.
    explicit dimension(const root_type&);

    // The root data type chosen when variable<> is declared. E.g. float, double, etc.
    explicit operator root_type() const;

    // Query return value of computation to get the derivatives.
    template<typename... Orders>
    typename type_at<RealType,sizeof...(Orders)-1>::type derivative(Orders... orders) const;

    // All of the arithmetic operators are overloaded.
    template<typename RealType2,size_t Order2>
    dimension<RealType,Order>& operator+=(const dimension<RealType2,Order2>&);

    dimension<RealType,Order>& operator+=(const root_type&);

    // ...
};

// Standard math functions are overloaded and called via argument-dependent lookup (ADL).
template<typename RealType,size_t Order>
dimension<RealType,Order> floor(const dimension<RealType,Order>&);

template<typename RealType,size_t Order>
dimension<RealType,Order> exp(const dimension<RealType,Order>&);

// ...

} } } } // namespace boost::math::differentiation::autodiff
/**/
