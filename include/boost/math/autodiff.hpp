//               Copyright Matthew Pulver 2018.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

/*!
 * \mainpage
 * \tableofcontents
 *
 * \section quick-start Quick Start Examples
 *
 * <hr>
 *
 * \subsection single-variable Example 1: Single-variable derivatives
 * <h3>Calculate derivatives of \f$f(x)=x^4\f$ at \f$x=2.0\f$.</h3>
 * \dontinclude fourth_power.cpp \skip #include
 * \until **
 * The above calculates \f{array}
{lrrl}
{\tt y.derivative(0)} =& f(2) =& \left.x^4\right|_{x=2} &= 16\\
{\tt y.derivative(1)} =& f'(2) =& \left.4\cdot x^3\right|_{x=2} &= 32\\
{\tt y.derivative(2)} =& f''(2) =& \left.4\cdot 3\cdot x^2\right|_{x=2} &= 48\\
{\tt y.derivative(3)} =& f'''(2) =& \left.4\cdot 3\cdot2\cdot x\right|_{x=2} &= 48\\
{\tt y.derivative(4)} =& f^{(4)}(2) =& 4\cdot 3\cdot2\cdot1 &= 24\\
{\tt y.derivative(5)} =& f^{(5)}(2) =& 0 &
\f}
 *
 * <hr>
 *
 * \subsection multiprecision Example 2: Multi-variable mixed partial derivatives with multi-precision data type
 * <h3>Calculate \f$\frac{\partial^{12}f}{\partial w^{3}\partial x^{2}\partial y^{4}\partial z^{3}}(11,12,13,14)\f$
 * with a precision of about 100 decimal digits, where
\f$f(w,x,y,z)=\exp\left(w\sin\left(\frac{x\log(y)}{z}\right)+\sqrt{\frac{wz}{xy}}\right)+\frac{w^2}{\tan(z)}\f$.</h3>
 * \dontinclude multiprecision.cpp \skip #include
 * \until **
 *
 * <hr>
 * \subsection black-scholes Example 3: Black-Scholes option pricing
 * <h3>Using the standard Black-Scholes model for pricing European options, calculate call/put prices and greeks.</h3>
 * https://en.wikipedia.org/wiki/Greeks_(finance)#Formulas_for_European_option_Greeks
 *
 * One of the primary benefits of using automatic differentiation is the elimination of additional functions to
calculate derivatives, which is a form of code redundancy.
 * \dontinclude black_scholes.cpp \skip #include
 * \until **
 *
 * <hr>
 *
 * \subsection multi-variable Example 4: Multi-variable mixed partial derivatives
 * <h3>Calculate mixed partial derivatives of
\f$f(w,x,y,z)=\exp\left(w\sin\left(\frac{x\log(y)}{z}\right)+\sqrt{\frac{wz}{xy}}\right)+\frac{w^2}{\tan(z)}\f$
at \f$(w,x,y,z)=(11,12,13,14)\f$.</h3>
 * \dontinclude mixed_partials.cpp \skip #include
 * \until **
 * The above calculates \f[{\tt v.derivative(iw,ix,iy,iz)} =
\frac{\partial^{i_w+i_x+i_y+i_z}f}{\partial w^{i_w}\partial x^{i_x}\partial y^{i_y}\partial z^{i_z}}(11,12,13,14)
\qquad\text{for each}\qquad
(i_w,i_x,i_y,i_z)\in \mathbb{N}_4\times\mathbb{N}_3\times\mathbb{N}_5\times\mathbb{N}_4
\f]
where \f$\mathbb{N}_i=\{0,1,2,...,i-1\}\f$.
For testing purposes, the \f$4\times3\times5\times4=240\f$-element <code>answers[]</code>
array was calculated independently by Mathematica (example/mixed_partials.nb) in 2 steps:

 1. <a href="https://reference.wolfram.com/language/tutorial/SymbolicComputation.html">Symbolic differentiation</a>
 2. <a href="https://reference.wolfram.com/language/tutorial/ArbitraryPrecisionNumbers.html">Arbitrary-precision computation</a> to calculate the answers to 20 decimal places.

Out of the 240 calculated values, the maximum relative error between the values calculated by Mathematica vs.
the Boost Autodiff library is found to be about \f$6.82\times10^{-13}\f$ using the standard IEEE double precision
floating point data type. Since the data type is a template variable, the error can be reduced arbitrarily by
using a data type with greater precision.

 *
 * <hr>
 *
 * \copyright
 * <center>
 *               Copyright &copy; Matthew Pulver 2018.<br/>
 * Distributed under the Boost Software License, Version 1.0.<br/>
 *      (See accompanying file LICENSE_1_0.txt or copy at<br/>
 *           https://www.boost.org/LICENSE_1_0.txt)
 * </center>
 *
 */

#ifndef BOOST_MATH_AUTODIFF_HPP
#define BOOST_MATH_AUTODIFF_HPP

#include <boost/config.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/tools/promotion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <ostream>
#include <type_traits>

// Automatic Differentiation v1
namespace boost { namespace math { namespace autodiff { inline namespace v1 {

// Use variable<> instead of dimension<> or nested_dimensions<>.
template<typename RealType,size_t Order>
class dimension;
template<typename RealType,size_t Order,size_t... Orders> // specialized for dimension<> below.
struct nested_dimensions { using type = dimension<typename nested_dimensions<RealType,Orders...>::type,Order>; };

// Satisfies Conceptual Requirements for Real Number Types
// https://www.boost.org/doc/libs/1_69_0/libs/math/doc/html/math_toolkit/real_concepts.html
template<typename RealType,size_t Order,size_t... Orders>
using variable = typename nested_dimensions<RealType,Order,Orders...>::type;

////////////////////////////////////////////////////////////////////////////////

template<typename... RealTypes>
using promote = typename boost::math::tools::promote_args<RealTypes...>::type;

// Get non-dimension<> root type T of variable<T,O0,O1,O2,...>.
template<typename RealType>
struct root_type_finder { using type = RealType; }; // specialized for dimension<> below.

template<typename RealType,size_t Depth>
struct type_at { using type = RealType; }; // specialized for dimension<> below.

template<typename RealType,size_t Order>
class dimension
{
    std::array<RealType,Order+1> v;
  public:
    using root_type = typename root_type_finder<RealType>::type; // RealType in the root dimension<RealType,Order>.
    dimension() = default;
    // RealType(cr) | RealType | RealType is copy constructible.
    dimension(const dimension<RealType,Order>&) = default;
    // Be aware of implicit casting from one dimension<> type to another by this copy constructor.
    template<typename RealType2,size_t Order2>
    dimension<RealType,Order>(const dimension<RealType2,Order2>&);
    // RealType(ca) | RealType | RealType is copy constructible from the arithmetic types.
    explicit dimension(const root_type&); // Initialize a variable of differentiation.
    explicit dimension(const std::initializer_list<root_type>&); // Initialize a constant.
    // r = cr | RealType& | Assignment operator.
    dimension<RealType,Order>& operator=(const dimension<RealType,Order>&) = default;
    // r = ca | RealType& | Assignment operator from the arithmetic types.
    dimension<RealType,Order>& operator=(const root_type&); // Set a constant.
    // r += cr | RealType& | Adds cr to r.
    template<typename RealType2,size_t Order2>
    dimension<RealType,Order>& operator+=(const dimension<RealType2,Order2>&);
    // r += ca | RealType& | Adds ar to r.
    dimension<RealType,Order>& operator+=(const root_type&);
    // r -= cr | RealType& | Subtracts cr from r.
    template<typename RealType2,size_t Order2>
    dimension<RealType,Order>& operator-=(const dimension<RealType2,Order2>&);
    // r -= ca | RealType& | Subtracts ca from r.
    dimension<RealType,Order>& operator-=(const root_type&);
    // r *= cr | RealType& | Multiplies r by cr.
    template<typename RealType2,size_t Order2>
    dimension<RealType,Order>& operator*=(const dimension<RealType2,Order2>&);
    // r *= ca | RealType& | Multiplies r by ca.
    dimension<RealType,Order>& operator*=(const root_type&);
    // r /= cr | RealType& | Divides r by cr.
    template<typename RealType2,size_t Order2>
    dimension<RealType,Order>& operator/=(const dimension<RealType2,Order2>&);
    // r /= ca | RealType& | Divides r by ca.
    dimension<RealType,Order>& operator/=(const root_type&);
    // -r | RealType | Unary Negation.
    dimension<RealType,Order> operator-() const;
    // +r | RealType& | Identity Operation.
    const dimension<RealType,Order>& operator+() const;
    // cr + cr2 | RealType | Binary Addition
    template<typename RealType2,size_t Order2>
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> operator+(const dimension<RealType2,Order2>&) const;
    // cr + ca | RealType | Binary Addition
    dimension<RealType,Order> operator+(const root_type&) const;
    // ca + cr | RealType | Binary Addition
    template<typename RealType2,size_t Order2>
    friend dimension<RealType2,Order2>
        operator+(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr - cr2 | RealType | Binary Subtraction
    template<typename RealType2,size_t Order2>
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> operator-(const dimension<RealType2,Order2>&) const;
    // cr - ca | RealType | Binary Subtraction
    dimension<RealType,Order> operator-(const root_type&) const;
    // ca - cr | RealType | Binary Subtraction
    template<typename RealType2,size_t Order2>
    friend dimension<RealType2,Order2>
        operator-(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr * cr2 | RealType | Binary Multiplication
    template<typename RealType2,size_t Order2>
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> operator*(const dimension<RealType2,Order2>&) const;
    // cr * ca | RealType | Binary Multiplication
    dimension<RealType,Order> operator*(const root_type&) const;
    // ca * cr | RealType | Binary Multiplication
    template<typename RealType2,size_t Order2>
    friend dimension<RealType2,Order2>
        operator*(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr / cr2 | RealType | Binary Subtraction
    template<typename RealType2,size_t Order2>
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> operator/(const dimension<RealType2,Order2>&) const;
    // cr / ca | RealType | Binary Subtraction
    dimension<RealType,Order> operator/(const root_type&) const;
    // ca / cr | RealType | Binary Subtraction
    template<typename RealType2,size_t Order2>
    friend dimension<RealType2,Order2>
        operator/(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr == cr2 | bool | Equality Comparison
    template<typename RealType2,size_t Order2> // This only compares the root term. All other terms are ignored.
    bool operator==(const dimension<RealType2,Order2>&) const;
    // cr == ca | bool | Equality Comparison
    bool operator==(const root_type&) const;
    // ca == cr | bool | Equality Comparison
    template<typename RealType2,size_t Order2> // This only compares the root term. All other terms are ignored.
    friend bool operator==(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr != cr2 | bool | Inequality Comparison
    template<typename RealType2,size_t Order2>
    bool operator!=(const dimension<RealType2,Order2>&) const;
    // cr != ca | bool | Inequality Comparison
    bool operator!=(const root_type&) const;
    // ca != cr | bool | Inequality Comparison
    template<typename RealType2,size_t Order2>
    friend bool operator!=(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr <= cr2 | bool | Less than equal to.
    template<typename RealType2,size_t Order2>
    bool operator<=(const dimension<RealType2,Order2>&) const;
    // cr <= ca | bool | Less than equal to.
    bool operator<=(const root_type&) const;
    // ca <= cr | bool | Less than equal to.
    template<typename RealType2,size_t Order2>
    friend bool operator<=(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr >= cr2 | bool | Greater than equal to.
    template<typename RealType2,size_t Order2>
    bool operator>=(const dimension<RealType2,Order2>&) const;
    // cr >= ca | bool | Greater than equal to.
    bool operator>=(const root_type&) const;
    // ca >= cr | bool | Greater than equal to.
    template<typename RealType2,size_t Order2>
    friend bool operator>=(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr < cr2 | bool | Less than comparison.
    template<typename RealType2,size_t Order2>
    bool operator<(const dimension<RealType2,Order2>&) const;
    // cr < ca | bool | Less than comparison.
    bool operator<(const root_type&) const;
    // ca < cr | bool | Less than comparison.
    template<typename RealType2,size_t Order2>
    friend bool operator<(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);
    // cr > cr2 | bool | Greater than comparison.
    template<typename RealType2,size_t Order2>
    bool operator>(const dimension<RealType2,Order2>&) const;
    // cr > ca | bool | Greater than comparison.
    bool operator>(const root_type&) const;
    // ca > cr | bool | Greater than comparison.
    template<typename RealType2,size_t Order2>
    friend bool operator>(const typename dimension<RealType2,Order2>::root_type&, const dimension<RealType2,Order2>&);

    // Will throw std::out_of_range if Order < order.
    template<typename... Orders>
    typename type_at<RealType,sizeof...(Orders)>::type at(size_t order, Orders... orders) const;
    template<typename... Orders>
    typename type_at<RealType,sizeof...(Orders)-1>::type derivative(Orders... orders) const;

    dimension<RealType,Order> inverse() const; // Multiplicative inverse.

    static constexpr size_t depth(); // = sizeof...(Orders)
    static constexpr size_t order_sum();

    explicit operator root_type() const;
    dimension<RealType,Order>& set_root(const root_type&);

    // Use when function returns derivatives.
    dimension<RealType,Order> apply(const std::function<root_type(size_t)>&) const;
    // Use when function returns derivative(i)/factorial(i) (slightly more efficient than apply().)
    dimension<RealType,Order> apply_with_factorials(const std::function<root_type(size_t)>&) const;
    // Same as apply() but uses horner method. May be more accurate in some cases but not as good with inf derivatives.
    dimension<RealType,Order> apply_with_horner(const std::function<root_type(size_t)>&) const;
    // Same as apply_with_factorials() buy uses horner method.
    dimension<RealType,Order> apply_with_horner_factorials(const std::function<root_type(size_t)>&) const;

private:
    RealType epsilon_inner_product(size_t z0, size_t isum0, size_t m0,
        const dimension<RealType,Order>& cr, size_t z1, size_t isum1, size_t m1, size_t j) const;
    dimension<RealType,Order> epsilon_multiply(size_t z0, size_t isum0,
        const dimension<RealType,Order>& cr, size_t z1, size_t isum1) const;
    dimension<RealType,Order> epsilon_multiply(size_t z0, size_t isum0, const root_type& ca) const;
    dimension<RealType,Order> inverse_apply() const;
    dimension<RealType,Order>& multiply_assign_by_root_type(bool is_root, const root_type&);

    template<typename RealType2,size_t Orders2>
    friend class dimension;
    template<typename RealType2,size_t Order2>
    friend std::ostream& operator<<(std::ostream&, const dimension<RealType2,Order2>&);

// C++11 Compatibility
#ifdef BOOST_NO_CXX17_IF_CONSTEXPR
    template<typename... Orders>
    typename type_at<RealType, sizeof...(Orders)>::type at_cpp11(std::true_type, size_t order, Orders... orders) const;
    template<typename... Orders>
    typename type_at<RealType, sizeof...(Orders)>::type at_cpp11(std::false_type, size_t order, Orders... orders) const;
    template<typename SizeType>
    dimension<RealType,Order> epsilon_multiply_cpp11(std::true_type, SizeType z0, size_t isum0,
        const dimension<RealType,Order>& cr, size_t z1, size_t isum1) const;
    template<typename SizeType>
    dimension<RealType,Order> epsilon_multiply_cpp11(std::false_type, SizeType z0, size_t isum0,
        const dimension<RealType,Order>& cr, size_t z1, size_t isum1) const;
    template<typename SizeType>
    dimension<RealType,Order> epsilon_multiply_cpp11(std::true_type, SizeType z0, size_t isum0,
        const root_type& ca) const;
    template<typename SizeType>
    dimension<RealType,Order> epsilon_multiply_cpp11(std::false_type, SizeType z0, size_t isum0,
        const root_type& ca) const;
    template<typename RootType>
    dimension<RealType,Order>& multiply_assign_by_root_type_cpp11(std::true_type, bool is_root, const RootType& ca);
    template<typename RootType>
    dimension<RealType,Order>& multiply_assign_by_root_type_cpp11(std::false_type, bool is_root, const RootType& ca);
    template<typename RootType>
    dimension<RealType,Order>& set_root_cpp11(std::true_type, const RootType& root);
    template<typename RootType>
    dimension<RealType,Order>& set_root_cpp11(std::false_type, const RootType& root);
#endif
};

// Standard Library Support Requirements
//
// fabs(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> fabs(const dimension<RealType,Order>&);
// abs(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> abs(const dimension<RealType,Order>&);
// ceil(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> ceil(const dimension<RealType,Order>&);
// floor(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> floor(const dimension<RealType,Order>&);
// exp(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> exp(const dimension<RealType,Order>&);
// pow(cr, ca) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> pow(const dimension<RealType,Order>&,const typename dimension<RealType,Order>::root_type&);
// pow(ca, cr) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> pow(const typename dimension<RealType,Order>::root_type&,const dimension<RealType,Order>&);
// pow(cr1, cr2) | RealType
template<typename RealType1,size_t Order1,typename RealType2,size_t Order2>
promote<dimension<RealType1,Order1>,dimension<RealType2,Order2>>
    pow(const dimension<RealType1,Order1>&, const dimension<RealType2,Order2>&);
// sqrt(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> sqrt(const dimension<RealType,Order>&);
// log(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> log(const dimension<RealType,Order>&);
// frexp(cr1, &i) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> frexp(const dimension<RealType,Order>&, int*);
// ldexp(cr1, i) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> ldexp(const dimension<RealType,Order>&, int);
// cos(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> cos(const dimension<RealType,Order>&);
// sin(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> sin(const dimension<RealType,Order>&);
// asin(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> asin(const dimension<RealType,Order>&);
// tan(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> tan(const dimension<RealType,Order>&);
// atan(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> atan(const dimension<RealType,Order>&);
// fmod(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> fmod(const dimension<RealType,Order>&, const typename dimension<RealType,Order>::root_type&);
// round(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> round(const dimension<RealType,Order>&);
// lround(cr1) | long
template<typename RealType,size_t Order>
long lround(const dimension<RealType,Order>&);
// llround(cr1) | long long
template<typename RealType,size_t Order>
long long llround(const dimension<RealType,Order>&);
// trunc(cr1) | RealType
template<typename RealType,size_t Order>
dimension<RealType,Order> trunc(const dimension<RealType,Order>&);
// truncl(cr1) | long double
template<typename RealType,size_t Order>
long double truncl(const dimension<RealType,Order>&);

// Additional functions
template<typename RealType,size_t Order>
dimension<RealType,Order> acos(const dimension<RealType,Order>&);
template<typename RealType,size_t Order>
dimension<RealType,Order> erfc(const dimension<RealType,Order>&);

template<typename RealType,size_t Order>
struct nested_dimensions<RealType,Order> { using type = dimension<RealType,Order>; };

template<typename RealType,size_t Order>
struct root_type_finder<dimension<RealType,Order>> { using type = typename root_type_finder<RealType>::type; };

// Specialization of type_at<> for dimension<>. Examples:
// * type_at<T,0>::type is T.
// * type_at<dimension<T,O1>,1>::type is T.
// * type_at<dimension<dimension<T,O2>,O1>,2>::type is T.
// * type_at<dimension<dimension<dimension<T,O3>,O2>,O1>,3>::type is T.
template<typename RealType,size_t Order,size_t Depth>
struct type_at<dimension<RealType,Order>,Depth>
{
    using type =
        typename std::conditional<Depth==0,dimension<RealType,Order>,typename type_at<RealType,Depth-1>::type>::type;
};

// Compile-time test for dimension<> type.
template<typename>
struct is_dimension : std::false_type {};
template<typename RealType,size_t Order>
struct is_dimension<dimension<RealType,Order>> : std::true_type {};

// C++11 compatibility
#ifdef BOOST_NO_CXX17_IF_CONSTEXPR
#  define BOOST_AUTODIFF_IF_CONSTEXPR
#else
#  define BOOST_AUTODIFF_IF_CONSTEXPR constexpr
#endif

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
dimension<RealType,Order>::dimension(const dimension<RealType2,Order2>& cr)
{
    if BOOST_AUTODIFF_IF_CONSTEXPR (is_dimension<RealType2>::value)
        for (size_t i=0 ; i<=std::min(Order,Order2) ; ++i)
            v[i] = RealType(cr.v[i]);
    else
        for (size_t i=0 ; i<=std::min(Order,Order2) ; ++i)
            v[i] = cr.v[i];
    if BOOST_AUTODIFF_IF_CONSTEXPR (Order2 < Order)
        std::fill(v.begin()+(Order2+1), v.end(), RealType{0});
}

// Note difference between arithmetic constructors and arithmetic assignment:
//  * non-initializer_list arithmetic constructor creates a variable dimension (epsilon coefficient = 1).
//  * initializer_list arithmetic constructor creates a constant dimension (epsilon coefficient = 0).
//  * arithmetic assignment creates a constant (epsilon coefficients = 0).
template<typename RealType,size_t Order>
dimension<RealType,Order>::dimension(const root_type& ca)
:    v{{static_cast<RealType>(ca)}}
{
    if BOOST_AUTODIFF_IF_CONSTEXPR (depth() == 1 && 0 < Order)
        v[1] = static_cast<root_type>(1); // Set epsilon coefficient = 1.
}

template<typename RealType,size_t Order>
dimension<RealType,Order>::dimension(const std::initializer_list<root_type>& list)
:    v{}
{
    for (size_t i=0 ; i<std::min(Order+1,list.size()) ; ++i)
        v[i] = *(list.begin()+i);
}

template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::operator=(const root_type& ca)
{
    v.front() = RealType{ca};
    if BOOST_AUTODIFF_IF_CONSTEXPR (0 < Order)
        std::fill(v.begin()+1, v.end(), RealType{0});
    return *this;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
dimension<RealType,Order>& dimension<RealType,Order>::operator+=(const dimension<RealType2,Order2>& cr)
{
    for (size_t i=0 ; i<=std::min(Order,Order2) ; ++i)
        v[i] += cr.v[i];
    return *this;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::operator+=(const root_type& ca)
{
    v.front() += ca;
    return *this;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
dimension<RealType,Order>& dimension<RealType,Order>::operator-=(const dimension<RealType2,Order2>& cr)
{
    for (size_t i=0 ; i<=Order ; ++i)
        v[i] -= cr.v[i];
    return *this;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::operator-=(const root_type& ca)
{
    v.front() -= ca;
    return *this;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
dimension<RealType,Order>& dimension<RealType,Order>::operator*=(const dimension<RealType2,Order2>& cr)
{
    const promote<RealType,RealType2> zero{0};
    if BOOST_AUTODIFF_IF_CONSTEXPR (Order <= Order2)
        for (size_t i=0, j=Order ; i<=Order ; ++i, --j)
            v[j] = std::inner_product(v.cbegin(), v.cend()-i, cr.v.crbegin()+i, zero);
    else
    {
        for (size_t i=0, j=Order ; i<=Order-Order2 ; ++i, --j)
            v[j] = std::inner_product(cr.v.cbegin(), cr.v.cend(), v.crbegin()+i, zero);
        for (size_t i=Order-Order2+1, j=Order2-1 ; i<=Order ; ++i, --j)
            v[j] = std::inner_product(cr.v.cbegin(), cr.v.cbegin()+(j+1), v.crbegin()+i, zero);
    }
    return *this;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::operator*=(const root_type& ca)
{
    return multiply_assign_by_root_type(true, ca);
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
dimension<RealType,Order>& dimension<RealType,Order>::operator/=(const dimension<RealType2,Order2>& cr)
{
    const RealType zero{0};
    v.front() /= cr.v.front();
    if BOOST_AUTODIFF_IF_CONSTEXPR (Order < Order2)
        for (size_t i=1, j=Order2-1, k=Order ; i<=Order ; ++i, --j, --k)
            (v[i] -= std::inner_product(cr.v.cbegin()+1, cr.v.cend()-j, v.crbegin()+k, zero)) /= cr.v.front();
    else if BOOST_AUTODIFF_IF_CONSTEXPR (0 < Order2)
        for (size_t i=1, j=Order2-1, k=Order ; i<=Order ; ++i, j&&--j, --k)
            (v[i] -= std::inner_product(cr.v.cbegin()+1, cr.v.cend()-j, v.crbegin()+k, zero)) /= cr.v.front();
    else
        for (size_t i=1 ; i<=Order ; ++i)
            v[i] /= cr.v.front();
    return *this;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::operator/=(const root_type& ca)
{
    std::for_each(v.begin(), v.end(), [&ca](RealType& x) { x /= ca; });
    return *this;
}

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::operator-() const
{
    dimension<RealType,Order> retval;
    for (size_t i=0 ; i<=Order ; ++i)
        retval.v[i] = -v[i];
    return retval;
}

template<typename RealType,size_t Order>
const dimension<RealType,Order>& dimension<RealType,Order>::operator+() const
{
    return *this;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
promote<dimension<RealType,Order>,dimension<RealType2,Order2>>
    dimension<RealType,Order>::operator+(const dimension<RealType2,Order2>& cr) const
{
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> retval;
    for (size_t i=0 ; i<=std::min(Order,Order2) ; ++i)
        retval.v[i] = v[i] + cr.v[i];
    if BOOST_AUTODIFF_IF_CONSTEXPR (Order < Order2)
        for (size_t i=Order+1 ; i<=Order2 ; ++i)
            retval.v[i] = cr.v[i];
    else if BOOST_AUTODIFF_IF_CONSTEXPR (Order2 < Order)
        for (size_t i=Order2+1 ; i<=Order ; ++i)
            retval.v[i] = v[i];
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::operator+(const root_type& ca) const
{
    dimension<RealType,Order> retval(*this);
    retval.v.front() += ca;
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>
    operator+(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return cr + ca;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
promote<dimension<RealType,Order>,dimension<RealType2,Order2>>
    dimension<RealType,Order>::operator-(const dimension<RealType2,Order2>& cr) const
{
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> retval;
    for (size_t i=0 ; i<=std::min(Order,Order2) ; ++i)
        retval.v[i] = v[i] - cr.v[i];
    if BOOST_AUTODIFF_IF_CONSTEXPR (Order < Order2)
        for (size_t i=Order+1 ; i<=Order2 ; ++i)
            retval.v[i] = -cr.v[i];
    else if BOOST_AUTODIFF_IF_CONSTEXPR (Order2 < Order)
        for (size_t i=Order2+1 ; i<=Order ; ++i)
            retval.v[i] = v[i];
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::operator-(const root_type& ca) const
{
    dimension<RealType,Order> retval(*this);
    retval.v.front() -= ca;
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>
    operator-(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return -cr += ca;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
promote<dimension<RealType,Order>,dimension<RealType2,Order2>>
    dimension<RealType,Order>::operator*(const dimension<RealType2,Order2>& cr) const
{
    const promote<RealType,RealType2> zero{0};
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> retval;
    if BOOST_AUTODIFF_IF_CONSTEXPR (Order < Order2)
        for (size_t i=0, j=Order, k=Order2 ; i<=Order2 ; ++i, j&&--j, --k)
            retval.v[i] = std::inner_product(v.cbegin(), v.cend()-j, cr.v.crbegin()+k, zero);
    else
        for (size_t i=0, j=Order2, k=Order ; i<=Order ; ++i, j&&--j, --k)
            retval.v[i] = std::inner_product(cr.v.cbegin(), cr.v.cend()-j, v.crbegin()+k, zero);
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::operator*(const root_type& ca) const
{
    return dimension<RealType,Order>(*this) *= ca;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>
    operator*(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return cr * ca;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
promote<dimension<RealType,Order>,dimension<RealType2,Order2>>
    dimension<RealType,Order>::operator/(const dimension<RealType2,Order2>& cr) const
{
    const promote<RealType,RealType2> zero{0};
    promote<dimension<RealType,Order>,dimension<RealType2,Order2>> retval;
    retval.v.front() = v.front() / cr.v.front();
    if BOOST_AUTODIFF_IF_CONSTEXPR (Order < Order2)
    {
        for (size_t i=1, j=Order2-1 ; i<=Order ; ++i, --j)
            retval.v[i] = (v[i] -
                std::inner_product(cr.v.cbegin()+1, cr.v.cend()-j, retval.v.crbegin()+(j+1), zero)) / cr.v.front();
        for (size_t i=Order+1, j=Order2-Order-1 ; i<=Order2 ; ++i, --j)
            retval.v[i] =
                -std::inner_product(cr.v.cbegin()+1, cr.v.cend()-j, retval.v.crbegin()+(j+1), zero) / cr.v.front();
    }
    else if BOOST_AUTODIFF_IF_CONSTEXPR (0 < Order2)
        for (size_t i=1, j=Order2-1, k=Order ; i<=Order ; ++i, j&&--j, --k)
            retval.v[i] =
                (v[i] - std::inner_product(cr.v.cbegin()+1, cr.v.cend()-j, retval.v.crbegin()+k, zero)) / cr.v.front();
    else
        for (size_t i=1 ; i<=Order ; ++i)
            retval.v[i] = v[i] / cr.v.front();
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::operator/(const root_type& ca) const
{
    return dimension<RealType,Order>(*this) /= ca;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>
    operator/(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    dimension<RealType,Order> retval;
    retval.v.front() = ca / cr.v.front();
    if BOOST_AUTODIFF_IF_CONSTEXPR (0 < Order)
    {
        const RealType zero{0};
        for (size_t i=1, j=Order-1 ; i<=Order ; ++i, --j)
            retval.v[i] = -std::inner_product(cr.v.cbegin()+1, cr.v.cend()-j, retval.v.crbegin()+(j+1), zero)
                / cr.v.front();
    }
    return retval;
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
bool dimension<RealType,Order>::operator==(const dimension<RealType2,Order2>& cr) const
{
    return v.front() == cr.v.front();
}

template<typename RealType,size_t Order>
bool dimension<RealType,Order>::operator==(const root_type& ca) const
{
    return v.front() == ca;
}

template<typename RealType,size_t Order>
bool operator==(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return ca == cr.v.front();
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
bool dimension<RealType,Order>::operator!=(const dimension<RealType2,Order2>& cr) const
{
    return v.front() != cr.v.front();
}

template<typename RealType,size_t Order>
bool dimension<RealType,Order>::operator!=(const root_type& ca) const
{
    return v.front() != ca;
}

template<typename RealType,size_t Order>
bool operator!=(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return ca != cr.v.front();
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
bool dimension<RealType,Order>::operator<=(const dimension<RealType2,Order2>& cr) const
{
    return v.front() <= cr.v.front();
}

template<typename RealType,size_t Order>
bool dimension<RealType,Order>::operator<=(const root_type& ca) const
{
    return v.front() <= ca;
}

template<typename RealType,size_t Order>
bool operator<=(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return ca <= cr.v.front();
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
bool dimension<RealType,Order>::operator>=(const dimension<RealType2,Order2>& cr) const
{
    return v.front() >= cr.v.front();
}

template<typename RealType,size_t Order>
bool dimension<RealType,Order>::operator>=(const root_type& ca) const
{
    return v.front() >= ca;
}

template<typename RealType,size_t Order>
bool operator>=(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return ca >= cr.v.front();
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
bool dimension<RealType,Order>::operator<(const dimension<RealType2,Order2>& cr) const
{
    return v.front() < cr.v.front();
}

template<typename RealType,size_t Order>
bool dimension<RealType,Order>::operator<(const root_type& ca) const
{
    return v.front() < ca;
}

template<typename RealType,size_t Order>
bool operator<(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return ca < cr.v.front();
}

template<typename RealType,size_t Order>
template<typename RealType2,size_t Order2>
bool dimension<RealType,Order>::operator>(const dimension<RealType2,Order2>& cr) const
{
    return v.front() > cr.v.front();
}

template<typename RealType,size_t Order>
bool dimension<RealType,Order>::operator>(const root_type& ca) const
{
    return v.front() > ca;
}

template<typename RealType,size_t Order>
bool operator>(const typename dimension<RealType,Order>::root_type& ca, const dimension<RealType,Order>& cr)
{
    return ca > cr.v.front();
}

/*** Other methods and functions ***/

// f : order -> derivative(order)
template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::apply(const std::function<root_type(size_t)>& f) const
{
    const dimension<RealType,Order> epsilon = dimension<RealType,Order>(*this).set_root(0);
    dimension<RealType,Order> epsilon_i = dimension<RealType,Order>{1}; // epsilon to the power of i
    dimension<RealType,Order> accumulator = dimension<RealType,Order>{f(0)};
    for (size_t i=1 ; i<=order_sum() ; ++i)
    {    // accumulator += (epsilon_i *= epsilon) * (f(i) / boost::math::factorial<root_type>(i));
        epsilon_i = epsilon_i.epsilon_multiply(i-1, 0, epsilon, 1, 0);
        accumulator += epsilon_i.epsilon_multiply(i, 0, f(i) / boost::math::factorial<root_type>(i));
    }
    return accumulator;
}

// f : order -> derivative(order)/factorial(order)
// Use this when the computation of the derivatives already includes the factorial terms. E.g. See atan().
template<typename RealType,size_t Order>
dimension<RealType,Order>
    dimension<RealType,Order>::apply_with_factorials(const std::function<root_type(size_t)>& f) const
{
    const dimension<RealType,Order> epsilon = dimension<RealType,Order>(*this).set_root(0);
    dimension<RealType,Order> epsilon_i = dimension<RealType,Order>{1}; // epsilon to the power of i
    dimension<RealType,Order> accumulator = dimension<RealType,Order>{f(0)};
    for (size_t i=1 ; i<=order_sum() ; ++i)
    {    // accumulator += (epsilon_i *= epsilon) * f(i);
        epsilon_i = epsilon_i.epsilon_multiply(i-1, 0, epsilon, 1, 0);
        accumulator += epsilon_i.epsilon_multiply(i, 0, f(i));
    }
    return accumulator;
}

// f : order -> derivative(order)
template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::apply_with_horner(const std::function<root_type(size_t)>& f) const
{
    const dimension<RealType,Order> epsilon = dimension<RealType,Order>(*this).set_root(0);
    auto accumulator = dimension<RealType,Order>{f(order_sum())/boost::math::factorial<root_type>(order_sum())};
    for (size_t i=order_sum() ; i-- ;)
        (accumulator *= epsilon) += f(i) / boost::math::factorial<root_type>(i);
    return accumulator;
}

// f : order -> derivative(order)/factorial(order)
// Use this when the computation of the derivatives already includes the factorial terms. E.g. See atan().
template<typename RealType,size_t Order>
dimension<RealType,Order>
    dimension<RealType,Order>::apply_with_horner_factorials(const std::function<root_type(size_t)>& f) const
{
    const dimension<RealType,Order> epsilon = dimension<RealType,Order>(*this).set_root(0);
    auto accumulator = dimension<RealType,Order>{f(order_sum())};
    for (size_t i=order_sum() ; i-- ;)
        (accumulator *= epsilon) += f(i);
    return accumulator;
}

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
// Can throw "std::out_of_range: array::at: __n (which is 7) >= _Nm (which is 7)"
template<typename RealType,size_t Order>
template<typename... Orders>
typename type_at<RealType,sizeof...(Orders)>::type dimension<RealType,Order>::at(size_t order, Orders... orders) const
{
    if constexpr (0 < sizeof...(orders))
        return v.at(order).at(orders...);
    else
        return v.at(order);
}
#endif

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
template<typename RealType,size_t Order>
constexpr size_t dimension<RealType,Order>::depth()
{
    if constexpr (is_dimension<RealType>::value)
        return 1 + RealType::depth();
    else
        return 1;
}
#endif

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
template<typename RealType,size_t Order>
constexpr size_t dimension<RealType,Order>::order_sum()
{
    if constexpr (is_dimension<RealType>::value)
        return Order + RealType::order_sum();
    else
        return Order;
}
#endif

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
// Can throw "std::out_of_range: array::at: __n (which is 7) >= _Nm (which is 7)"
template<typename RealType,size_t Order>
template<typename... Orders>
typename type_at<RealType,sizeof...(Orders)-1>::type dimension<RealType,Order>::derivative(Orders... orders) const
{
    static_assert(sizeof...(orders) <= depth(),
        "Number of parameters to derivative(...) cannot exceed the number of dimensions in the dimension<...>.");
    return at(orders...) * (... * boost::math::factorial<root_type>(orders));
}
#endif

template<typename RealType,size_t Order>
RealType dimension<RealType,Order>::epsilon_inner_product(size_t z0, size_t isum0, size_t m0,
    const dimension<RealType,Order>& cr, size_t z1, size_t isum1, size_t m1, size_t j) const
{
    static_assert(is_dimension<RealType>::value, "epsilon_inner_product() must have 1 < depth().");
    RealType accumulator = RealType();
    const size_t i0_max = m1 < j ? j-m1 : 0;
    for (size_t i0=m0, i1=j-m0 ; i0<=i0_max ; ++i0, --i1)
        accumulator += v.at(i0).epsilon_multiply(z0, isum0+i0, cr.v.at(i1), z1, isum1+i1);
    return accumulator;
}

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply(size_t z0, size_t isum0,
    const dimension<RealType,Order>& cr, size_t z1, size_t isum1) const
{
    const RealType zero{0};
    const size_t m0 = order_sum() + isum0 < Order + z0 ? Order + z0 - (order_sum() + isum0) : 0;
    const size_t m1 = order_sum() + isum1 < Order + z1 ? Order + z1 - (order_sum() + isum1) : 0;
    const size_t i_max = m0 + m1 < Order ? Order - (m0 + m1) : 0;
    dimension<RealType,Order> retval = dimension<RealType,Order>();
    if constexpr (is_dimension<RealType>::value)
        for (size_t i=0, j=Order ; i<=i_max ; ++i, --j)
            retval.v[j] = epsilon_inner_product(z0, isum0, m0, cr, z1, isum1, m1, j);
    else
        for (size_t i=0, j=Order ; i<=i_max ; ++i, --j)
            retval.v[j] = std::inner_product(v.cbegin()+m0, v.cend()-(i+m1), cr.v.crbegin()+(i+m0), zero);
    return retval;
}
#endif

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
// When called from outside this method, z0 should be non-zero. Otherwise if z0=0 then it will give an
// incorrect result of 0 when the root value is 0 and ca=inf, when instead the correct product is nan.
// If z0=0 then use the regular multiply operator*() instead.
template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply(size_t z0, size_t isum0,
    const dimension<RealType,Order>::root_type& ca) const
{
    dimension<RealType,Order> retval(*this);
    const size_t m0 = order_sum() + isum0 < Order + z0 ? Order + z0 - (order_sum() + isum0) : 0;
    if constexpr (is_dimension<RealType>::value)
        for (size_t i=m0 ; i<=Order ; ++i)
            retval.v[i] = retval.v[i].epsilon_multiply(z0, isum0+i, ca);
    else
        for (size_t i=m0 ; i<=Order ; ++i)
            if (retval.v[i] != static_cast<RealType>(0))
                retval.v[i] *= ca;
    return retval;
}
#endif

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::inverse() const
{
    return operator root_type() == 0 ? inverse_apply() : 1 / *this;
}

// This gives autodiff::log(0.0) = depth(1)(-inf,inf,-inf,inf,-inf,inf)
// 1 / *this: autodiff::log(0.0) = depth(1)(-inf,inf,-inf,-nan,-nan,-nan)
template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::inverse_apply() const
{
    std::array<root_type,order_sum()+1> derivatives; // derivatives of 1/x
    const root_type x0 = static_cast<root_type>(*this);
    derivatives[0] = 1 / x0;
    for (size_t i=1 ; i<=order_sum() ; ++i)
        derivatives[i] = -derivatives[i-1] * i / x0;
    return apply([&derivatives](size_t j) { return derivatives[j]; });
}

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::multiply_assign_by_root_type(bool is_root, const root_type& ca)
{
    auto itr = v.begin();
    if constexpr (is_dimension<RealType>::value)
    {
        itr->multiply_assign_by_root_type(is_root, ca);
        for (++itr ; itr!=v.end() ; ++itr)
            itr->multiply_assign_by_root_type(false, ca);
    }
    else
    {
        if (is_root || *itr != 0)
            *itr *= ca; // Skip multiplication of 0 by ca=inf to avoid nan. Exception: root value is always multiplied.
        for (++itr ; itr!=v.end() ; ++itr)
            if (*itr != 0)
                *itr *= ca;
    }
    return *this;
}
#endif

template<typename RealType,size_t Order>
dimension<RealType,Order>::operator root_type() const
{
    return static_cast<root_type>(v.front());
}

#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::set_root(const root_type& root)
{
    if constexpr (is_dimension<RealType>::value)
        v.front().set_root(root);
    else
        v.front() = root;
    return *this;
}
#endif

// Standard Library Support Requirements

template<typename RealType,size_t Order>
dimension<RealType,Order> fabs(const dimension<RealType,Order>& cr)
{
    const typename dimension<RealType,Order>::root_type zero{0};
    return zero < cr ? cr
        : cr < zero ? -cr
        : cr == zero ? dimension<RealType,Order>() // Canonical fabs'(0) = 0.
        : cr; // Propagate NaN.
}

template<typename RealType,size_t Order>
dimension<RealType,Order> abs(const dimension<RealType,Order>& cr)
{
    return fabs(cr);
}

template<typename RealType,size_t Order>
dimension<RealType,Order> ceil(const dimension<RealType,Order>& cr)
{
    using std::ceil;
    return dimension<RealType,Order>{ceil(cr.at(0))}; // constant with all epsilon terms zero.
}

template<typename RealType,size_t Order>
dimension<RealType,Order> floor(const dimension<RealType,Order>& cr)
{
    using std::floor;
    return dimension<RealType,Order>{floor(cr.at(0))}; // constant with all epsilon terms zero.
}

template<typename RealType,size_t Order>
dimension<RealType,Order> exp(const dimension<RealType,Order>& cr)
{
    using std::exp;
    using root_type = typename dimension<RealType,Order>::root_type;
    const root_type d0 = exp(static_cast<root_type>(cr));
    return cr.apply_with_horner([&d0](size_t) { return d0; });
}

template<typename RealType,size_t Order>
dimension<RealType,Order> pow(const dimension<RealType,Order>& x,const typename dimension<RealType,Order>::root_type& y)
{
    using std::pow;
    using root_type = typename dimension<RealType,Order>::root_type;
    constexpr size_t order = dimension<RealType,Order>::order_sum();
    std::array<root_type,order+1> derivatives; // array of derivatives
    const root_type x0 = static_cast<root_type>(x);
    size_t i = 0;
    root_type coef = 1;
    for (; i<=order && coef!=0 ; ++i)
    {
        derivatives[i] = coef * pow(x0, y-i);
        coef *= y - i;
    }
    return x.apply([&derivatives,i](size_t j) { return j < i ? derivatives[j] : 0; });
}

template<typename RealType,size_t Order>
dimension<RealType,Order> pow(const typename dimension<RealType,Order>::root_type& x,const dimension<RealType,Order>& y)
{
    using std::log;
    return exp(y*log(x));
}

template<typename RealType1,size_t Order1,typename RealType2,size_t Order2>
promote<dimension<RealType1,Order1>,dimension<RealType2,Order2>>
    pow(const dimension<RealType1,Order1>& x, const dimension<RealType2,Order2>& y)
{
    return exp(y*log(x));
}

template<typename RealType,size_t Order>
dimension<RealType,Order> sqrt(const dimension<RealType,Order>& cr)
{
    return pow(cr,0.5);
}

// Natural logarithm. If cr==0 then derivative(i) may have nans due to nans from inverse().
template<typename RealType,size_t Order>
dimension<RealType,Order> log(const dimension<RealType,Order>& cr)
{
    using std::log;
    using root_type = typename dimension<RealType,Order>::root_type;
    constexpr size_t order = dimension<RealType,Order>::order_sum();
    const root_type d0 = log(static_cast<root_type>(cr));
    if BOOST_AUTODIFF_IF_CONSTEXPR (order == 0)
        return dimension<RealType,0>(d0);
    else
    {
        const auto d1 = dimension<root_type,order-1>(static_cast<root_type>(cr)).inverse(); // log'(x) = 1 / x
        return cr.apply_with_factorials([&d0,&d1](size_t i) { return i ? d1.at(i-1)/i : d0; });
    }
}

template<typename RealType,size_t Order>
dimension<RealType,Order> frexp(const dimension<RealType,Order>& cr, int* exp)
{
    using std::exp2;
    using std::frexp;
    using root_type = typename dimension<RealType,Order>::root_type;
    frexp(static_cast<root_type>(cr), exp);
    return cr * exp2(-*exp);
}

template<typename RealType,size_t Order>
dimension<RealType,Order> ldexp(const dimension<RealType,Order>& cr, int exp)
{
    using std::exp2;
    return cr * exp2(exp);
}

template<typename RealType,size_t Order>
dimension<RealType,Order> cos(const dimension<RealType,Order>& cr)
{
    using std::cos;
    using std::sin;
    using root_type = typename dimension<RealType,Order>::root_type;
    const root_type d0 = cos(static_cast<root_type>(cr));
    if BOOST_AUTODIFF_IF_CONSTEXPR (dimension<RealType,Order>::order_sum() == 0)
        return dimension<RealType,0>(d0);
    else
    {
        const root_type d1 = -sin(static_cast<root_type>(cr));
        const root_type derivatives[] { d0, d1, -d0, -d1 };
        return cr.apply_with_horner([derivatives](size_t i) { return derivatives[i&3]; });
    }
}

template<typename RealType,size_t Order>
dimension<RealType,Order> sin(const dimension<RealType,Order>& cr)
{
    using std::sin;
    using std::cos;
    using root_type = typename dimension<RealType,Order>::root_type;
    const root_type d0 = sin(static_cast<root_type>(cr));
    if BOOST_AUTODIFF_IF_CONSTEXPR (dimension<RealType,Order>::order_sum() == 0)
        return dimension<RealType,0>(d0);
    else
    {
        const root_type d1 = cos(static_cast<root_type>(cr));
        const root_type derivatives[] { d0, d1, -d0, -d1 };
        return cr.apply_with_horner([derivatives](size_t i) { return derivatives[i&3]; });
    }
}

template<typename RealType,size_t Order>
dimension<RealType,Order> asin(const dimension<RealType,Order>& cr)
{
    using std::asin;
    using root_type = typename dimension<RealType,Order>::root_type;
    constexpr size_t order = dimension<RealType,Order>::order_sum();
    const root_type d0 = asin(static_cast<root_type>(cr));
    if BOOST_AUTODIFF_IF_CONSTEXPR (order == 0)
        return dimension<RealType,0>(d0);
    else
    {
        auto d1 = dimension<root_type,order-1>(static_cast<root_type>(cr)); // asin'(x) = 1 / sqrt(1-x*x).
        d1 = sqrt(1-(d1*=d1)).inverse(); // asin(1): d1 = depth(1)(inf,inf,-nan,-nan,-nan)
        //d1 = sqrt((1-(d1*=d1)).inverse()); // asin(1): d1 = depth(1)(inf,-nan,-nan,-nan,-nan)
        return cr.apply_with_factorials([&d0,&d1](size_t i) { return i ? d1.at(i-1)/i : d0; });
    }
}

template<typename RealType,size_t Order>
dimension<RealType,Order> tan(const dimension<RealType,Order>& cr)
{
    return sin(cr) / cos(cr);
}

template<typename RealType,size_t Order>
dimension<RealType,Order> atan(const dimension<RealType,Order>& cr)
{
    using std::atan;
    using root_type = typename dimension<RealType,Order>::root_type;
    constexpr size_t order = dimension<RealType,Order>::order_sum();
    const root_type d0 = atan(static_cast<root_type>(cr));
    if BOOST_AUTODIFF_IF_CONSTEXPR (order == 0)
        return dimension<RealType,0>(d0);
    else
    {
        auto d1 = dimension<root_type,order-1>(static_cast<root_type>(cr));
        d1 = ((d1*=d1)+=1).inverse(); // atan'(x) = 1 / (x*x+1).
        return cr.apply_with_horner_factorials([&d0,&d1](size_t i) { return i ? d1.at(i-1)/i : d0; });
    }
}

template<typename RealType,size_t Order>
dimension<RealType,Order>
    fmod(const dimension<RealType,Order>& cr, const typename dimension<RealType,Order>::root_type& ca)
{
    using std::fmod;
    using root_type = typename dimension<RealType,Order>::root_type;
    return dimension<RealType,Order>(cr).set_root(0) += fmod(static_cast<root_type>(cr), ca);
}

template<typename RealType,size_t Order>
dimension<RealType,Order> round(const dimension<RealType,Order>& cr)
{
    using std::round;
    return dimension<RealType,Order>{round(cr.at(0))}; // constant with all epsilon terms zero.
}

template<typename RealType,size_t Order>
long lround(const dimension<RealType,Order>& cr)
{
    using std::lround;
    return lround(cr.at(0));
}

template<typename RealType,size_t Order>
long long llround(const dimension<RealType,Order>& cr)
{
    using std::llround;
    return llround(cr.at(0));
}

template<typename RealType,size_t Order>
dimension<RealType,Order> trunc(const dimension<RealType,Order>& cr)
{
    using std::trunc;
    return dimension<RealType,Order>{trunc(cr.at(0))}; // constant with all epsilon terms zero.
}

template<typename RealType,size_t Order>
long double truncl(const dimension<RealType,Order>& cr)
{
    using std::truncl;
    return truncl(cr.at(0));
}

template<typename RealType,size_t Order>
std::ostream& operator<<(std::ostream& out, const dimension<RealType,Order>& dim)
{
    const std::streamsize original_precision = out.precision();
    out.precision(std::numeric_limits<typename dimension<RealType,Order>::root_type>::digits10);
    out << "depth(" << dim.depth() << ')';
    for (size_t i=0 ; i<dim.v.size() ; ++i)
        out << (i?',':'(') << dim.v[i];
    out.precision(original_precision);
    return out << ')';
}

// Additional functions

template<typename RealType,size_t Order>
dimension<RealType,Order> acos(const dimension<RealType,Order>& cr)
{
    using std::acos;
    using root_type = typename dimension<RealType,Order>::root_type;
    constexpr size_t order = dimension<RealType,Order>::order_sum();
    const root_type d0 = acos(static_cast<root_type>(cr));
    if BOOST_AUTODIFF_IF_CONSTEXPR (order == 0)
        return dimension<RealType,0>(d0);
    else
    {
        auto d1 = dimension<root_type,order-1>(static_cast<root_type>(cr));
        d1 = -sqrt(1-(d1*=d1)).inverse(); // acos'(x) = -1 / sqrt(1-x*x).
        return cr.apply_with_horner_factorials([&d0,&d1](size_t i) { return i ? d1.at(i-1)/i : d0; });
    }
}

template<typename RealType,size_t Order>
dimension<RealType,Order> erfc(const dimension<RealType,Order>& cr)
{
    using std::erfc;
    using root_type = typename dimension<RealType,Order>::root_type;
    constexpr size_t order = dimension<RealType,Order>::order_sum();
    const root_type d0 = erfc(static_cast<root_type>(cr));
    if BOOST_AUTODIFF_IF_CONSTEXPR (order == 0)
        return dimension<RealType,0>(d0);
    else
    {
        auto d1 = dimension<root_type,order-1>(static_cast<root_type>(cr));
        d1 = -2*boost::math::constants::one_div_root_pi<root_type>()*exp(-(d1*=d1)); // erfc'(x)=-2/sqrt(pi)*exp(-x*x)
        return cr.apply_with_horner_factorials([&d0,&d1](size_t i) { return i ? d1.at(i-1)/i : d0; });
    }
}

} } } } // namespace boost::math::autodiff::v1

namespace std {
/// boost::math::tools::digits<RealType>() is handled by this std::numeric_limits<> specialization,
/// and similarly for max_value, min_value, log_max_value, log_min_value, and epsilon.
template <typename RealType,size_t Order>
class numeric_limits<boost::math::autodiff::dimension<RealType,Order>>
    : public numeric_limits<typename boost::math::autodiff::dimension<RealType,Order>::root_type>
{ };
} // namespace std

namespace boost { namespace math { namespace tools {

// See boost/math/tools/promotion.hpp
#ifndef BOOST_NO_CXX17_IF_CONSTEXPR
template <typename RealType0,size_t Order0,typename RealType1,size_t Order1>
struct promote_args_2<autodiff::dimension<RealType0,Order0>,autodiff::dimension<RealType1,Order1>>
{
    using type = autodiff::dimension<typename promote_args_2<RealType0,RealType1>::type,std::max(Order0,Order1)>;
};
#endif

template <typename RealType0,size_t Order0,typename RealType1>
struct promote_args_2<autodiff::dimension<RealType0,Order0>,RealType1>
{
    using type = autodiff::dimension<typename promote_args_2<RealType0,RealType1>::type,Order0>;
};

template <typename RealType0,typename RealType1,size_t Order1>
struct promote_args_2<RealType0,autodiff::dimension<RealType1,Order1>>
{
    using type = autodiff::dimension<typename promote_args_2<RealType0,RealType1>::type,Order1>;
};

} } } // namespace boost::math::tools

#ifdef BOOST_NO_CXX17_IF_CONSTEXPR
#include "autodiff_cpp11.hpp"
#endif

#endif // BOOST_MATH_AUTODIFF_HPP
