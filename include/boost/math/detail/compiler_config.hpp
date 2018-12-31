//               Copyright Kedar R. Bhat 2018.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)


#ifndef AUTODIFF_COMPILER_CONFIG_HPP
#define AUTODIFF_COMPILER_CONFIG_HPP

#include <boost/config.hpp>
#ifdef BOOST_COMPILER_CONFIG
#include BOOST_COMPILER_CONFIG
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1700L && !defined(_MSVC_LANG)
#define BOOST_MATH_AUTODIFF_CPP_STD 201103L
#elif defined(_MSVC_LANG)
#define BOOST_MATH_AUTODIFF_CPP_STD _MSVC_LANG
#else
#define BOOST_MATH_AUTODIFF_CPP_STD __cplusplus
#endif

#endif //AUTODIFF_COMPILER_CONFIG_HPP
