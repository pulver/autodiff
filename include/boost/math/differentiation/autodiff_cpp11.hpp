//               Copyright Matthew Pulver 2018.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)

// Contributors:
//  * Kedar R. Bhat - C++11 compatibility.

// Notes:
//  * Any changes to this file should always be downstream from autodiff.hpp.
//    C++17 is a higher-level language and is easier to maintain. For example, a number of functions which are
//    lucidly read in autodiff.cpp are forced to be split into multiple structs/functions in this file for C++11.
//  * Use of typename RootType and SizeType is a hack to prevent Visual Studio 2015 from compiling functions
//    that are never called, that would otherwise produce compiler errors.

#ifndef BOOST_MATH_AUTODIFF_HPP
#   error "Do not #include this file directly. This should only be #included by autodiff.hpp for C++11 compatibility."
#endif

namespace boost { namespace math { namespace differentiation { namespace autodiff { inline namespace v1 {

template<typename RealType,size_t Order>
template<typename... Orders>
typename type_at<RealType, sizeof...(Orders)>::type
    dimension<RealType,Order>::at_cpp11(std::true_type, size_t order, Orders... orders) const
{
    return v.at(order);
}

template<typename RealType,size_t Order>
template<typename... Orders>
typename type_at<RealType, sizeof...(Orders)>::type
    dimension<RealType,Order>::at_cpp11(std::false_type, size_t order, Orders... orders) const
{
    return v.at(order).at(orders...);
}

// Can throw "std::out_of_range: array::at: __n (which is 7) >= _Nm (which is 7)"
template<typename RealType,size_t Order>
template<typename... Orders>
typename type_at<RealType,sizeof...(Orders)>::type dimension<RealType,Order>::at(size_t order, Orders... orders) const
{
    return at_cpp11(std::integral_constant<bool,sizeof...(orders)==0>{}, order, orders...);
}

namespace detail
{
	template<typename T, typename = void>
	struct depth_t : std::integral_constant<std::size_t, 1> {};

	template<typename RealType, size_t Order>
	struct depth_t<dimension<RealType, Order>, typename std::enable_if<is_dimension<RealType>::value>::type> : std::integral_constant<std::size_t, depth_t<RealType>::value + 1> {};

	template<typename T> struct get_order_t;

	template<typename RealType, size_t Order> struct get_order_t <dimension<RealType, Order>> : std::integral_constant<size_t, Order> {};

	template<typename T, typename = void>
	struct order_sum_t : get_order_t<T> {};

	template<typename RealType, size_t Order>
	struct order_sum_t<dimension<RealType, Order>, typename std::enable_if<is_dimension<RealType>::value>::type> : std::integral_constant<std::size_t, order_sum_t<RealType>::value + Order> {};
} // namespace detail

template<typename RealType, size_t Order>
constexpr size_t dimension<RealType,Order>::depth()
{
	return detail::depth_t<dimension<RealType, Order>>::value;
}

template<typename RealType, size_t Order>
constexpr size_t dimension<RealType, Order>::order_sum()
{
	return detail::order_sum_t<dimension<RealType, Order>>::value;
}

template<typename T, typename... Ts>
constexpr T product(Ts... factors)
{
    return 1;
}

template<typename T, typename... Ts>
constexpr T product(T factor, Ts... factors)
{
    return factor * product<T>(factors...);
}

// Can throw "std::out_of_range: array::at: __n (which is 7) >= _Nm (which is 7)"
template<typename RealType,size_t Order>
template<typename... Orders>
typename type_at<RealType,sizeof...(Orders)-1>::type dimension<RealType,Order>::derivative(Orders... orders) const
{
    static_assert(sizeof...(Orders) <= depth(),
        "Number of parameters to derivative(...) cannot exceed the number of dimensions in the dimension<...>.");
    return at(orders...) * product(boost::math::factorial<root_type>(orders)...);
}

template<typename RealType,size_t Order>
template<typename SizeType>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply_cpp11(std::true_type,
    SizeType z0, size_t isum0, const dimension<RealType,Order>& cr, size_t z1, size_t isum1) const
{
    const RealType zero{0};
    const size_t m0 = order_sum() + isum0 < Order + z0 ? Order + z0 - (order_sum() + isum0) : 0;
    const size_t m1 = order_sum() + isum1 < Order + z1 ? Order + z1 - (order_sum() + isum1) : 0;
    const size_t i_max = m0 + m1 < Order ? Order - (m0 + m1) : 0;
    dimension<RealType,Order> retval = dimension<RealType,Order>();
    for (size_t i=0, j=Order ; i<=i_max ; ++i, --j)
        retval.v[j] = epsilon_inner_product(z0, isum0, m0, cr, z1, isum1, m1, j);
    return retval;
}

template<typename RealType,size_t Order>
template<typename SizeType>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply_cpp11(std::false_type,
    SizeType z0, size_t isum0, const dimension<RealType,Order>& cr, size_t z1, size_t isum1) const
{
    const RealType zero{0};
    const size_t m0 = order_sum() + isum0 < Order + z0 ? Order + z0 - (order_sum() + isum0) : 0;
    const size_t m1 = order_sum() + isum1 < Order + z1 ? Order + z1 - (order_sum() + isum1) : 0;
    const size_t i_max = m0 + m1 < Order ? Order - (m0 + m1) : 0;
    dimension<RealType,Order> retval = dimension<RealType,Order>();
    for (size_t i=0, j=Order ; i<=i_max ; ++i, --j)
        retval.v[j] = std::inner_product(v.cbegin()+m0, v.cend()-(i+m1), cr.v.crbegin()+(i+m0), zero);
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply(size_t z0, size_t isum0,
    const dimension<RealType,Order>& cr, size_t z1, size_t isum1) const
{
    return epsilon_multiply_cpp11(std::integral_constant<bool,is_dimension<RealType>::value>{},
        z0, isum0, cr, z1, isum1);
}

template<typename RealType,size_t Order>
template<typename SizeType>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply_cpp11(std::true_type,
    SizeType z0, size_t isum0, const root_type& ca) const
{
    dimension<RealType,Order> retval(*this);
    const size_t m0 = order_sum() + isum0 < Order + z0 ? Order + z0 - (order_sum() + isum0) : 0;
    for (size_t i=m0 ; i<=Order ; ++i)
        retval.v[i] = retval.v[i].epsilon_multiply(z0, isum0+i, ca);
    return retval;
}

template<typename RealType,size_t Order>
template<typename SizeType>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply_cpp11(std::false_type,
    SizeType z0, size_t isum0, const root_type& ca) const
{
    dimension<RealType,Order> retval(*this);
    const size_t m0 = order_sum() + isum0 < Order + z0 ? Order + z0 - (order_sum() + isum0) : 0;
    for (size_t i=m0 ; i<=Order ; ++i)
        if (retval.v[i] != static_cast<RealType>(0))
            retval.v[i] *= ca;
    return retval;
}

template<typename RealType,size_t Order>
dimension<RealType,Order> dimension<RealType,Order>::epsilon_multiply(size_t z0, size_t isum0,
    const root_type& ca) const
{
    return epsilon_multiply_cpp11(std::integral_constant<bool,is_dimension<RealType>::value>{}, z0, isum0, ca);
}

template<typename RealType,size_t Order>
template<typename RootType>
dimension<RealType,Order>& dimension<RealType,Order>::multiply_assign_by_root_type_cpp11(std::true_type,
    bool is_root, const RootType& ca)
{
    auto itr = v.begin();
    itr->multiply_assign_by_root_type(is_root, ca);
    for (++itr ; itr!=v.end() ; ++itr)
        itr->multiply_assign_by_root_type(false, ca);
    return *this;
}

template<typename RealType,size_t Order>
template<typename RootType>
dimension<RealType,Order>& dimension<RealType,Order>::multiply_assign_by_root_type_cpp11(std::false_type,
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

template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::multiply_assign_by_root_type(bool is_root, const root_type& ca)
{
    return multiply_assign_by_root_type_cpp11(std::integral_constant<bool,is_dimension<RealType>::value>{},
        is_root, ca);
}

template<typename RealType,size_t Order>
template<typename RootType>
dimension<RealType,Order>& dimension<RealType,Order>::set_root_cpp11(std::true_type, const RootType& root)
{
    v.front().set_root(root);
    return *this;
}

template<typename RealType,size_t Order>
template<typename RootType>
dimension<RealType,Order>& dimension<RealType,Order>::set_root_cpp11(std::false_type, const RootType& root)
{
    v.front() = root;
    return *this;
}

template<typename RealType,size_t Order>
dimension<RealType,Order>& dimension<RealType,Order>::set_root(const root_type& root)
{
    return set_root_cpp11(std::integral_constant<bool,is_dimension<RealType>::value>{}, root);
}

// This gives autodiff::log(0.0) = depth(1)(-inf,inf,-inf,inf,-inf,inf)
// 1 / *this: autodiff::log(0.0) = depth(1)(-inf,inf,-inf,-nan,-nan,-nan)
template<typename RealType, size_t Order>
dimension<RealType, Order> dimension<RealType, Order>::inverse_apply() const
{
	std::array<root_type, detail::order_sum_t<dimension<RealType,Order>>::value + 1> derivatives; // derivatives of 1/x
	const root_type x0 = static_cast<root_type>(*this);
	derivatives[0] = 1 / x0;
	for (size_t i = 1; i <= order_sum(); ++i)
		derivatives[i] = -derivatives[i - 1] * i / x0;
	return apply([&derivatives](size_t j) { return derivatives[j]; });
}

// Natural logarithm. If cr==0 then derivative(i) may have nans due to nans from inverse().
template<typename RealType, size_t Order>
dimension<RealType, Order> log(const dimension<RealType, Order>& cr)
{
	using std::log;
	using root_type = typename dimension<RealType, Order>::root_type;
	constexpr size_t order = detail::order_sum_t<dimension<RealType,Order>>::value;
	const root_type d0 = log(static_cast<root_type>(cr));
	if BOOST_AUTODIFF_IF_CONSTEXPR(order == 0)
		return dimension<RealType, 0>(d0);
	else
	{
		const auto d1 = dimension<root_type, order - 1>(static_cast<root_type>(cr)).inverse(); // log'(x) = 1 / x
		return cr.apply_with_factorials([&d0, &d1](size_t i) { return i ? d1.at(i - 1) / i : d0; });
	}
}


} } } } } // namespace boost::math::differentiation::autodiff::v1