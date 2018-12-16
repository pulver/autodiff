#define BOOST_TEST_MODULE test_autodiff
#include <boost/math/autodiff.hpp>

#include <boost/test/included/unit_test.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/fpclassify.hpp> // isnan

#include <iostream>

template<typename W,typename X,typename Y,typename Z>
auto mixed_partials_f(const W& w, const X& x, const Y& y, const Z& z)
{
    using namespace boost::math::autodiff;
    using namespace std;
    return exp(w*sin(x*log(y)/z) + sqrt(w*z/(x*y))) + w*w/tan(z);
}

BOOST_AUTO_TEST_SUITE(test_autodiff)

BOOST_AUTO_TEST_CASE(constructors)
{
	constexpr int m = 3;
	constexpr int n = 4;
	// Verify value-initialized instance has all 0 entries.
	const boost::math::autodiff::variable<double,m> empty1 = boost::math::autodiff::variable<double,m>();
	for (int i=0 ; i<=m ; ++i)
		BOOST_TEST(empty1.derivative(i) == 0.0);
	const auto empty2 = boost::math::autodiff::variable<double,m,n>();
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			BOOST_TEST(empty2.derivative(i,j) == 0.0);
	// Single variable
	constexpr double cx = 10.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	for (int i=0 ; i<=m ; ++i)
		if (i==0)
			BOOST_TEST(x.derivative(i) == cx);
		else if (i==1)
			BOOST_TEST(x.derivative(i) == 1.0);
		else
			BOOST_TEST(x.derivative(i) == 0.0);
	// Second independent variable
	constexpr double cy = 100.0;
	const auto y = boost::math::autodiff::variable<double,m,n>(cy);
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(y.derivative(i,j) == cy);
			else if (i==0 && j==1)
				BOOST_TEST(y.derivative(i,j) == 1.0);
			else
				BOOST_TEST(y.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(assignment)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 10.0;
	constexpr double cy = 10.0;
	boost::math::autodiff::variable<double,m,n> empty; // Uninitialized variable<> may have non-zero values.
	// Single variable
	auto x = boost::math::autodiff::variable<double,m>(cx);
	empty = x; // Test assignment operator of single-variable to double-variable.
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(empty.derivative(i,j) == cx);
			else if (i==1 && j==0)
				BOOST_TEST(empty.derivative(i,j) == 1.0);
			else
				BOOST_TEST(empty.derivative(i,j) == 0.0);
	auto y = boost::math::autodiff::variable<double,m,n>(cy);
	empty = y; // default assignment operator
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(empty.derivative(i,j) == cy);
			else if (i==0 && j==1)
				BOOST_TEST(empty.derivative(i,j) == 1.0);
			else
				BOOST_TEST(empty.derivative(i,j) == 0.0);
	empty = cx; // set a constant
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(empty.derivative(i,j) == cx);
			else
				BOOST_TEST(empty.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(addition_assignment)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 10.0;
	auto sum = boost::math::autodiff::variable<double,m,n>(); // zero-initialized
	// Single variable
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	sum += x;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(sum.derivative(i,j) == cx);
			else if (i==1 && j==0)
				BOOST_TEST(sum.derivative(i,j) == 1.0);
			else
				BOOST_TEST(sum.derivative(i,j) == 0.0);
	// Arithmetic constant
	constexpr double cy = 11.0;
	sum = 0;
	sum += cy;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(sum.derivative(i,j) == cy);
			else
				BOOST_TEST(sum.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(subtraction_assignment)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 10.0;
	auto sum = boost::math::autodiff::variable<double,m,n>(); // zero-initialized
	// Single variable
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	sum -= x;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(sum.derivative(i,j) == -cx);
			else if (i==1 && j==0)
				BOOST_TEST(sum.derivative(i,j) == -1.0);
			else
				BOOST_TEST(sum.derivative(i,j) == 0.0);
	// Arithmetic constant
	constexpr double cy = 11.0;
	sum = 0;
	sum -= cy;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(sum.derivative(i,j) == -cy);
			else
				BOOST_TEST(sum.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(multiplication_assignment)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 10.0;
	auto product = boost::math::autodiff::variable<double,m,n>{1}; // unit constant
	// Single variable
	auto x = boost::math::autodiff::variable<double,m>(cx);
	product *= x;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(product.derivative(i,j) == cx);
			else if (i==1 && j==0)
				BOOST_TEST(product.derivative(i,j) == 1.0);
			else
				BOOST_TEST(product.derivative(i,j) == 0.0);
	// Arithmetic constant
	constexpr double cy = 11.0;
	product = 1;
	product *= cy;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(product.derivative(i,j) == cy);
			else
				BOOST_TEST(product.derivative(i,j) == 0.0);
	// 0 * inf = nan
	x = boost::math::autodiff::variable<double,m>(0.0);
	x *= std::numeric_limits<double>::infinity();
	//std::cout << "x = " << x << std::endl;
	for (int i=0 ; i<=m ; ++i)
		if (i==0)
			BOOST_TEST(boost::math::isnan(static_cast<double>(x))); // Correct
			//BOOST_TEST(x.derivative(i) == 0.0); // Wrong. See multiply_assign_by_root_type().
		else if (i==1)
			BOOST_TEST(boost::math::isinf(x.derivative(i)));
		else
			BOOST_TEST(x.derivative(i) == 0.0);
}

BOOST_AUTO_TEST_CASE(division_assignment)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 16.0;
	auto quotient = boost::math::autodiff::variable<double,m,n>{1}; // unit constant
	// Single variable
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	quotient /= x;
	BOOST_TEST(quotient.derivative(0,0) == 1/cx);
	BOOST_TEST(quotient.derivative(1,0) == -1/std::pow(cx,2));
	BOOST_TEST(quotient.derivative(2,0) == 2/std::pow(cx,3));
	BOOST_TEST(quotient.derivative(3,0) == -6/std::pow(cx,4));
	for (int i=0 ; i<=m ; ++i)
		for (int j=1 ; j<=n ; ++j)
			BOOST_TEST(quotient.derivative(i,j) == 0.0);
	// Arithmetic constant
	constexpr double cy = 32.0;
	quotient = 1;
	quotient /= cy;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(quotient.derivative(i,j) == 1/cy);
			else
				BOOST_TEST(quotient.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(unary_signs)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 16.0;
	boost::math::autodiff::variable<double,m,n> lhs;
	// Single variable
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	lhs = -x;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(lhs.derivative(i,j) == -cx);
			else if (i==1 && j==0)
				BOOST_TEST(lhs.derivative(i,j) == -1.0);
			else
				BOOST_TEST(lhs.derivative(i,j) == 0.0);
	lhs = +x;
	for (int i=0 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			if (i==0 && j==0)
				BOOST_TEST(lhs.derivative(i,j) == cx);
			else if (i==1 && j==0)
				BOOST_TEST(lhs.derivative(i,j) == 1.0);
			else
				BOOST_TEST(lhs.derivative(i,j) == 0.0);
}

// TODO 3 tests for 3 operator+() definitions.

BOOST_AUTO_TEST_CASE(cast_double)
{
	constexpr double ca = 3.0;
	const auto x0 = boost::math::autodiff::variable<double,0>(ca);
	BOOST_TEST(static_cast<double>(x0) == ca);
	const auto x1 = boost::math::autodiff::variable<double,1>(ca);
	BOOST_TEST(static_cast<double>(x1) == ca);
	const auto x2 = boost::math::autodiff::variable<double,2>(ca);
	BOOST_TEST(static_cast<double>(x2) == ca);
}

BOOST_AUTO_TEST_CASE(scalar_addition)
{
	constexpr double ca = 3.0;
	constexpr double cb = 4.0;
	const auto sum0 = boost::math::autodiff::variable<double,0>(ca) + boost::math::autodiff::variable<double,0>(cb);
	BOOST_TEST(ca+cb == static_cast<double>(sum0));
	const auto sum1 = boost::math::autodiff::variable<double,0>(ca) + cb;
	BOOST_TEST(ca+cb == static_cast<double>(sum1));
	const auto sum2 = ca + boost::math::autodiff::variable<double,0>(cb);
	BOOST_TEST(ca+cb == static_cast<double>(sum2));
}

BOOST_AUTO_TEST_CASE(power8)
{
	constexpr int n = 8;
	constexpr double ca = 3.0;
	auto x = boost::math::autodiff::variable<double,n>(ca);
	// Test operator*=()
	x *= x;
	x *= x;
	x *= x;
	const double power_factorial = boost::math::factorial<double>(n);
	for (int i=0 ; i<=n ; ++i)
		BOOST_TEST(x.derivative(i) == power_factorial/boost::math::factorial<double>(n-i)*std::pow(ca,n-i));
	x = boost::math::autodiff::variable<double,n>(ca);
	// Test operator*()
	x = x*x*x*x * x*x*x*x;
	for (int i=0 ; i<=n ; ++i)
		BOOST_TEST(x.derivative(i) == power_factorial/boost::math::factorial<double>(n-i)*std::pow(ca,n-i));
}

BOOST_AUTO_TEST_CASE(dim1_multiplication)
{
	constexpr int m = 2;
	constexpr int n = 3;
	constexpr double cy = 4.0;
	auto y0 = boost::math::autodiff::variable<double,m>(cy);
	auto y  = boost::math::autodiff::variable<double,n>(cy);
	y *= y0;
	BOOST_TEST(y.derivative(0) == cy*cy);
	BOOST_TEST(y.derivative(1) == 2*cy);
	BOOST_TEST(y.derivative(2) == 2.0);
	BOOST_TEST(y.derivative(3) == 0.0);
	y = y * cy;
	BOOST_TEST(y.derivative(0) == cy*cy*cy);
	BOOST_TEST(y.derivative(1) == 2*cy*cy);
	BOOST_TEST(y.derivative(2) == 2.0*cy);
	BOOST_TEST(y.derivative(3) == 0.0);
}

BOOST_AUTO_TEST_CASE(dim1and2_multiplication)
{
	constexpr int m = 2;
	constexpr int n = 3;
	constexpr double cx = 3.0;
	constexpr double cy = 4.0;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::variable<double,m,n>(cy);
	y *= x;
	BOOST_TEST(y.derivative(0,0) == cx*cy);
	BOOST_TEST(y.derivative(0,1) == cx);
	BOOST_TEST(y.derivative(1,0) == cy);
	BOOST_TEST(y.derivative(1,1) == 1.0);
	for (int i=1 ; i<m ; ++i)
		for (int j=1 ; j<n ; ++j)
			if (i==1 && j==1)
				BOOST_TEST(y.derivative(i,j) == 1.0);
			else
				BOOST_TEST(y.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(dim2_addition)
{
	constexpr int m = 2;
	constexpr int n = 3;
	constexpr double cx = 3.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	BOOST_TEST(x.derivative(0) == cx);
	BOOST_TEST(x.derivative(1) == 1.0);
	BOOST_TEST(x.derivative(2) == 0.0);
	constexpr double cy = 4.0;
	const auto y = boost::math::autodiff::variable<double,m,n>(cy);
	BOOST_TEST(static_cast<double>(y.derivative(0)) == cy);
	BOOST_TEST(static_cast<double>(y.derivative(1)) == 0.0); // partial of y w.r.t. x.

	BOOST_TEST(y.derivative(0,0) == cy);
	BOOST_TEST(y.derivative(0,1) == 1.0);
	BOOST_TEST(y.derivative(1,0) == 0.0);
	BOOST_TEST(y.derivative(1,1) == 0.0);
	const auto z = x + y;
	BOOST_TEST(z.derivative(0,0) == cx + cy);
	BOOST_TEST(z.derivative(0,1) == 1.0);
	BOOST_TEST(z.derivative(1,0) == 1.0);
	BOOST_TEST(z.derivative(1,1) == 0.0);
	// The following 4 are unnecessarily more expensive than the previous 4.
	BOOST_TEST(z.derivative(0).derivative(0) == cx + cy);
	BOOST_TEST(z.derivative(0).derivative(1) == 1.0);
	BOOST_TEST(z.derivative(1).derivative(0) == 1.0);
	BOOST_TEST(z.derivative(1).derivative(1) == 0.0);
}

BOOST_AUTO_TEST_CASE(dim2_multiplication)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 6.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	constexpr double cy = 5.0;
	const auto y = boost::math::autodiff::variable<double,0,n>(cy);
	const auto z = x*x * y*y*y;
	BOOST_TEST(z.derivative(0,0) == cx*cx * cy*cy*cy); // x^2 * y^3
	BOOST_TEST(z.derivative(0,1) == cx*cx * 3*cy*cy); // x^2 * 3y^2
	BOOST_TEST(z.derivative(0,2) == cx*cx * 6*cy); // x^2 * 6y
	BOOST_TEST(z.derivative(0,3) == cx*cx * 6); // x^2 * 6
	BOOST_TEST(z.derivative(0,4) == 0.0); // x^2 * 0
	BOOST_TEST(z.derivative(1,0) == 2*cx * cy*cy*cy); // 2x * y^3
	BOOST_TEST(z.derivative(1,1) == 2*cx * 3*cy*cy); // 2x * 3y^2
	BOOST_TEST(z.derivative(1,2) == 2*cx * 6*cy); // 2x * 6y
	BOOST_TEST(z.derivative(1,3) == 2*cx * 6); // 2x * 6
	BOOST_TEST(z.derivative(1,4) == 0.0); // 2x * 0
	BOOST_TEST(z.derivative(2,0) == 2 * cy*cy*cy); // 2 * y^3
	BOOST_TEST(z.derivative(2,1) == 2 * 3*cy*cy); // 2 * 3y^2
	BOOST_TEST(z.derivative(2,2) == 2 * 6*cy); // 2 * 6y
	BOOST_TEST(z.derivative(2,3) == 2 * 6); // 2 * 6
	BOOST_TEST(z.derivative(2,4) == 0.0); // 2 * 0
	BOOST_TEST(z.derivative(3,0) == 0.0); // 0 * y^3
	BOOST_TEST(z.derivative(3,1) == 0.0); // 0 * 3y^2
	BOOST_TEST(z.derivative(3,2) == 0.0); // 0 * 6y
	BOOST_TEST(z.derivative(3,3) == 0.0); // 0 * 6
	BOOST_TEST(z.derivative(3,4) == 0.0); // 0 * 0
}

BOOST_AUTO_TEST_CASE(dim2_multiplication_and_subtraction)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 6.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	constexpr double cy = 5.0;
	const auto y = boost::math::autodiff::variable<double,0,n>(cy);
	const auto z = x*x - y*y;
	BOOST_TEST(z.derivative(0,0) == cx*cx - cy*cy);
	BOOST_TEST(z.derivative(0,1) == -2*cy);
	BOOST_TEST(z.derivative(0,2) == -2.0);
	BOOST_TEST(z.derivative(0,3) == 0.0);
	BOOST_TEST(z.derivative(0,4) == 0.0);
	BOOST_TEST(z.derivative(1,0) == 2*cx);
	for (int i=1 ; i<=m ; ++i)
		for (int j=1 ; j<=n ; ++j)
			if (i==2 && j==0)
				BOOST_TEST(z.derivative(i,j) == 2.0);
			else
				BOOST_TEST(z.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(inverse)
{
	constexpr int m = 3;
	constexpr double cx = 4.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	const auto xinv = x.inverse();
	BOOST_TEST(xinv.derivative(0) == 1/cx);
	BOOST_TEST(xinv.derivative(1) == -1/std::pow(cx,2));
	BOOST_TEST(xinv.derivative(2) == 2/std::pow(cx,3));
	BOOST_TEST(xinv.derivative(3) == -6/std::pow(cx,4));
}

BOOST_AUTO_TEST_CASE(division)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 5.0;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	constexpr double cy = 4.0;
	auto y = boost::math::autodiff::variable<double,0,n>(cy);
	auto z = x*x / (y*y);
	BOOST_TEST(z.derivative(0,0) == cx*cx / (cy*cy)); // x^2 * y^-2
	BOOST_TEST(z.derivative(0,1) == cx*cx * (-2)*std::pow(cy,-3));
	BOOST_TEST(z.derivative(0,2) == cx*cx * (6)*std::pow(cy,-4));
	BOOST_TEST(z.derivative(0,3) == cx*cx * (-24)*std::pow(cy,-5));
	BOOST_TEST(z.derivative(0,4) == cx*cx * (120)*std::pow(cy,-6));
	BOOST_TEST(z.derivative(1,0) == 2*cx / (cy*cy));
	BOOST_TEST(z.derivative(1,1) == 2*cx * (-2)*std::pow(cy,-3));
	BOOST_TEST(z.derivative(1,2) == 2*cx * (6)*std::pow(cy,-4));
	BOOST_TEST(z.derivative(1,3) == 2*cx * (-24)*std::pow(cy,-5));
	BOOST_TEST(z.derivative(1,4) == 2*cx * (120)*std::pow(cy,-6));
	BOOST_TEST(z.derivative(2,0) == 2 / (cy*cy));
	BOOST_TEST(z.derivative(2,1) == 2 * (-2)*std::pow(cy,-3));
	BOOST_TEST(z.derivative(2,2) == 2 * (6)*std::pow(cy,-4));
	BOOST_TEST(z.derivative(2,3) == 2 * (-24)*std::pow(cy,-5));
	BOOST_TEST(z.derivative(2,4) == 2 * (120)*std::pow(cy,-6));
	for (int j=0 ; j<=n ; ++j)
		BOOST_TEST(z.derivative(3,j) == 0.0);

	auto x1 = boost::math::autodiff::variable<double,m>(cx);
	auto z1 = x1/cy;
	BOOST_TEST(z1.derivative(0) == cx/cy);
	BOOST_TEST(z1.derivative(1) == 1/cy);
	BOOST_TEST(z1.derivative(2) == 0.0);
	BOOST_TEST(z1.derivative(3) == 0.0);
	auto y2 = boost::math::autodiff::variable<double,m,n>(cy);
	auto z2 = cx/y2;
	BOOST_TEST(z2.derivative(0,0) == cx/cy);
	BOOST_TEST(z2.derivative(0,1) == -cx/std::pow(cy,2));
	BOOST_TEST(z2.derivative(0,2) == 2*cx/std::pow(cy,3));
	BOOST_TEST(z2.derivative(0,3) == -6*cx/std::pow(cy,4));
	BOOST_TEST(z2.derivative(0,4) == 24*cx/std::pow(cy,5));
	for (int i=1 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			BOOST_TEST(z2.derivative(i,j) == 0.0);
}

BOOST_AUTO_TEST_CASE(equality)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 10.0;
	constexpr double cy = 10.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	const auto y = boost::math::autodiff::variable<double,0,n>(cy);
	BOOST_TEST((x == y));
	BOOST_TEST((x == cy));
	BOOST_TEST((cx == y));
	BOOST_TEST((cy == x));
	BOOST_TEST((y == cx));
}

BOOST_AUTO_TEST_CASE(inequality)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 10.0;
	constexpr double cy = 11.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	const auto y = boost::math::autodiff::variable<double,0,n>(cy);
	BOOST_TEST((x != y));
	BOOST_TEST((x != cy));
	BOOST_TEST((cx != y));
	BOOST_TEST((cy != x));
	BOOST_TEST((y != cx));
}

BOOST_AUTO_TEST_CASE(less_than_or_equal_to)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 10.0;
	constexpr double cy = 11.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	const auto y = boost::math::autodiff::variable<double,0,n>(cy);
	BOOST_TEST((x <= y));
	BOOST_TEST((x <= y-1));
	BOOST_TEST((x < y));
	BOOST_TEST((x <= cy));
	BOOST_TEST((x <= cy-1));
	BOOST_TEST((x < cy));
	BOOST_TEST((cx <= y));
	BOOST_TEST((cx <= y-1));
	BOOST_TEST((cx < y));
}

BOOST_AUTO_TEST_CASE(greater_than_or_equal_to)
{
	constexpr int m = 3;
	constexpr int n = 4;
	constexpr double cx = 11.0;
	constexpr double cy = 10.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	const auto y = boost::math::autodiff::variable<double,0,n>(cy);
	BOOST_TEST((x >= y));
	BOOST_TEST((x >= y+1));
	BOOST_TEST((x > y));
	BOOST_TEST((x >= cy));
	BOOST_TEST((x >= cy+1));
	BOOST_TEST((x > cy));
	BOOST_TEST((cx >= y));
	BOOST_TEST((cx >= y+1));
	BOOST_TEST((cx > y));
}

BOOST_AUTO_TEST_CASE(abs)
{
	constexpr int m = 3;
	constexpr double cx = 11.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	auto abs = boost::math::autodiff::abs(x);
	BOOST_TEST(abs.derivative(0) == std::abs(cx));
	BOOST_TEST(abs.derivative(1) == 1.0);
	BOOST_TEST(abs.derivative(2) == 0.0);
	BOOST_TEST(abs.derivative(3) == 0.0);
	abs = boost::math::autodiff::abs(-x);
	BOOST_TEST(abs.derivative(0) == std::abs(cx));
	BOOST_TEST(abs.derivative(1) == 1.0); // abs(-x) = abs(x)
	BOOST_TEST(abs.derivative(2) == 0.0);
	BOOST_TEST(abs.derivative(3) == 0.0);
	const auto xneg = boost::math::autodiff::variable<double,m>(-cx);
	abs = boost::math::autodiff::abs(xneg);
	BOOST_TEST(abs.derivative(0) == std::abs(cx));
	BOOST_TEST(abs.derivative(1) == -1.0);
	BOOST_TEST(abs.derivative(2) == 0.0);
	BOOST_TEST(abs.derivative(3) == 0.0);
	const auto zero = boost::math::autodiff::variable<double,m>(0);
	abs = boost::math::autodiff::abs(zero);
	for (int i=0 ; i<=m ; ++i)
		BOOST_TEST(abs.derivative(i) == 0.0);
}

BOOST_AUTO_TEST_CASE(ceil_and_floor)
{
	constexpr int m = 3;
	double tests[] { -1.5, 0.0, 1.5 };
	for (unsigned t=0 ; t<sizeof(tests)/sizeof(*tests) ; ++t)
	{
		const auto x = boost::math::autodiff::variable<double,m>(tests[t]);
		auto ceil = boost::math::autodiff::ceil(x);
		auto floor = boost::math::autodiff::floor(x);
		BOOST_TEST(ceil.derivative(0) == std::ceil(tests[t]));
		BOOST_TEST(floor.derivative(0) == std::floor(tests[t]));
		for (int i=1 ; i<=m ; ++i)
		{
			BOOST_TEST(ceil.derivative(i) == 0.0);
			BOOST_TEST(floor.derivative(i) == 0.0);
		}
	}
}

BOOST_AUTO_TEST_CASE(one_over_one_plus_x_squared)
{
	constexpr int m = 4;
	constexpr double cx = 1.0;
	auto f = boost::math::autodiff::variable<double,m>(cx);
	f = ((f *= f) += 1).inverse();
	BOOST_TEST(f.derivative(0) == 0.5);
	BOOST_TEST(f.derivative(1) == -0.5);
	BOOST_TEST(f.derivative(2) == 0.5);
	BOOST_TEST(f.derivative(3) == 0.0);
	BOOST_TEST(f.derivative(4) == -3.0);
}

BOOST_AUTO_TEST_CASE(exp)
{
	constexpr int m = 4;
	constexpr double cx = 2.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::exp(x);
	for (int i=0 ; i<=m ; ++i)
		BOOST_TEST(y.derivative(i) == std::exp(cx));
}

BOOST_AUTO_TEST_CASE(pow, * boost::unit_test::tolerance(1e-15))
{
	constexpr int m = 5;
	constexpr int n = 4;
	constexpr double cx = 2.0;
	constexpr double cy = 3.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	const auto y = boost::math::autodiff::variable<double,m,n>(cy);
	auto z0 = boost::math::autodiff::pow(x,cy);
	BOOST_TEST(z0.derivative(0) == std::pow(cx,cy));
	BOOST_TEST(z0.derivative(1) == cy*std::pow(cx,cy-1));
	BOOST_TEST(z0.derivative(2) == cy*(cy-1)*std::pow(cx,cy-2));
	BOOST_TEST(z0.derivative(3) == cy*(cy-1)*(cy-2)*std::pow(cx,cy-3));
	BOOST_TEST(z0.derivative(4) == 0.0);
	BOOST_TEST(z0.derivative(5) == 0.0);
	auto z1 = boost::math::autodiff::pow(cx,y);
	BOOST_TEST(z1.derivative(0,0) == std::pow(cx,cy));
	for (int j=1 ; j<=n ; ++j)
		BOOST_TEST(z1.derivative(0,j) == std::pow(std::log(cx),j)*std::exp(cy*std::log(cx)));
	for (int i=1 ; i<=m ; ++i)
		for (int j=0 ; j<=n ; ++j)
			BOOST_TEST(z1.derivative(i,j) == 0.0);
	auto z2 = boost::math::autodiff::pow(x,y);
	for (int j=0 ; j<=n ; ++j)
		BOOST_TEST(z2.derivative(0,j) == std::pow(cx,cy)*std::pow(std::log(cx),j));
	for (int j=0 ; j<=n ; ++j)
		BOOST_TEST(z2.derivative(1,j) == std::pow(cx,cy-1)*std::pow(std::log(cx),j-1)*(cy*std::log(cx)+j));
	BOOST_TEST(z2.derivative(2,0) == std::pow(cx,cy-2)*cy*(cy-1));
	BOOST_TEST(z2.derivative(2,1) == std::pow(cx,cy-2)*(cy*(cy-1)*std::log(cx)+2*cy-1));
	for (int j=2 ; j<=n ; ++j)
		BOOST_TEST(z2.derivative(2,j) == std::pow(cx,cy-2)*std::pow(std::log(cx),j-2)*(j*(2*cy-1)*std::log(cx)+(j-1)*j+(cy-1)*cy*std::pow(std::log(cx),2)));
	BOOST_TEST(z2.derivative(2,4) == std::pow(cx,cy-2)*std::pow(std::log(cx),2)*(4*(2*cy-1)*std::log(cx)+(4-1)*4+(cy-1)*cy*std::pow(std::log(cx),2)));
}

BOOST_AUTO_TEST_CASE(sqrt)
{
	constexpr int m = 5;
	constexpr double cx = 4.0;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::sqrt(x);
	BOOST_TEST(y.derivative(0) == std::sqrt(cx));
	BOOST_TEST(y.derivative(1) == 0.5*std::pow(cx,-0.5));
	BOOST_TEST(y.derivative(2) == -0.5*0.5*std::pow(cx,-1.5));
	BOOST_TEST(y.derivative(3) == 0.5*0.5*1.5*std::pow(cx,-2.5));
	BOOST_TEST(y.derivative(4) == -0.5*0.5*1.5*2.5*std::pow(cx,-3.5));
	BOOST_TEST(y.derivative(5) == 0.5*0.5*1.5*2.5*3.5*std::pow(cx,-4.5));
	x = boost::math::autodiff::variable<double,m>(0);
	y = boost::math::autodiff::sqrt(x);
	//std::cout << "boost::math::autodiff::sqrt(0) = " << y << std::endl; // (0,inf,-inf,inf,-inf,inf)
	BOOST_TEST(y.derivative(0) == 0.0);
	for (int i=1; i<=m ; ++i)
		BOOST_TEST(y.derivative(i) == (i&1?1:-1)*std::numeric_limits<double>::infinity());
}

BOOST_AUTO_TEST_CASE(log)
{
	constexpr int m = 5;
	constexpr double cx = 2.0;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::log(x);
	BOOST_TEST(y.derivative(0) == std::log(cx));
	BOOST_TEST(y.derivative(1) == 1/cx);
	BOOST_TEST(y.derivative(2) == -1/std::pow(cx,2));
	BOOST_TEST(y.derivative(3) == 2/std::pow(cx,3));
	BOOST_TEST(y.derivative(4) == -6/std::pow(cx,4));
	BOOST_TEST(y.derivative(5) == 24/std::pow(cx,5));
	x = boost::math::autodiff::variable<double,m>(0);
	y = boost::math::autodiff::log(x);
	//std::cout << "boost::math::autodiff::log(0) = " << y << std::endl; // boost::math::autodiff::log(0) = depth(1)(-inf,inf,-inf,inf,-inf,inf)
	for (int i=0; i<=m ; ++i)
		BOOST_TEST(y.derivative(i) == (i&1?1:-1)*std::numeric_limits<double>::infinity());
}

BOOST_AUTO_TEST_CASE(ylogx, * boost::unit_test::tolerance(1e-15))
{
	constexpr int m = 5;
	constexpr int n = 4;
	constexpr double cx = 2.0;
	constexpr double cy = 3.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	const auto y = boost::math::autodiff::variable<double,m,n>(cy);
	auto z = y*boost::math::autodiff::log(x);
	BOOST_TEST(z.derivative(0,0) == cy*std::log(cx));
	BOOST_TEST(z.derivative(0,1) == std::log(cx));
	BOOST_TEST(z.derivative(0,2) == 0.0);
	BOOST_TEST(z.derivative(0,3) == 0.0);
	BOOST_TEST(z.derivative(0,4) == 0.0);
	for (size_t i=1 ; i<=m ; ++i)
		BOOST_TEST(z.derivative(i,0) == std::pow(-1,i-1)*boost::math::factorial<double>(i-1)*cy/std::pow(cx,i));
	for (size_t i=1 ; i<=m ; ++i)
		BOOST_TEST(z.derivative(i,1) == std::pow(-1,i-1)*boost::math::factorial<double>(i-1)/std::pow(cx,i));
	for (size_t i=1 ; i<=m ; ++i)
		for (size_t j=2 ; j<=n ; ++j)
			BOOST_TEST(z.derivative(i,j) == 0.0);
	auto z1 = boost::math::autodiff::exp(z);
	BOOST_TEST(z1.derivative(2,4) == std::pow(cx,cy-2)*std::pow(std::log(cx),2)*(4*(2*cy-1)*std::log(cx)+(4-1)*4+(cy-1)*cy*std::pow(std::log(cx),2))); // RHS is confirmed by https://www.wolframalpha.com/input/?i=D%5Bx%5Ey,%7Bx,2%7D,%7By,4%7D%5D+%2F.+%7Bx-%3E2.0,+y-%3E3.0%7D
}

BOOST_AUTO_TEST_CASE(frexp)
{
	constexpr int m = 3;
	constexpr double cx = 3.5;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	int exp, testexp;
	auto y = boost::math::autodiff::frexp(x,&exp);
	BOOST_TEST(y.derivative(0) == std::frexp(cx,&testexp));
	BOOST_TEST(exp == testexp);
	BOOST_TEST(y.derivative(1) == std::exp2(-exp));
	BOOST_TEST(y.derivative(2) == 0.0);
	BOOST_TEST(y.derivative(3) == 0.0);
}

BOOST_AUTO_TEST_CASE(ldexp)
{
	constexpr int m = 3;
	constexpr double cx = 3.5;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	constexpr int exp = 3;
	auto y = boost::math::autodiff::ldexp(x,exp);
	BOOST_TEST(y.derivative(0) == std::ldexp(cx,exp));
	BOOST_TEST(y.derivative(1) == std::exp2(exp));
	BOOST_TEST(y.derivative(2) == 0.0);
	BOOST_TEST(y.derivative(3) == 0.0);
}

BOOST_AUTO_TEST_CASE(cos_and_sin)
{
	constexpr int m = 5;
	constexpr double cx = M_PI/3;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	auto cos = boost::math::autodiff::cos(x);
	BOOST_TEST(cos.derivative(0) == std::cos(cx));
	BOOST_TEST(cos.derivative(1) == -std::sin(cx));
	BOOST_TEST(cos.derivative(2) == -std::cos(cx));
	BOOST_TEST(cos.derivative(3) == std::sin(cx));
	BOOST_TEST(cos.derivative(4) == std::cos(cx));
	BOOST_TEST(cos.derivative(5) == -std::sin(cx));
	auto sin = boost::math::autodiff::sin(x);
	BOOST_TEST(sin.derivative(0) == std::sin(cx));
	BOOST_TEST(sin.derivative(1) == std::cos(cx));
	BOOST_TEST(sin.derivative(2) == -std::sin(cx));
	BOOST_TEST(sin.derivative(3) == -std::cos(cx));
	BOOST_TEST(sin.derivative(4) == std::sin(cx));
	BOOST_TEST(sin.derivative(5) == std::cos(cx));
}

BOOST_AUTO_TEST_CASE(asin, * boost::unit_test::tolerance(1e-15))
{
	constexpr int m = 5;
	constexpr double cx = 0.5;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::asin(x);
	BOOST_TEST(y.derivative(0) == std::asin(cx));
	BOOST_TEST(y.derivative(1) == 1/std::sqrt(1-cx*cx));
	BOOST_TEST(y.derivative(2) == cx/std::pow(1-cx*cx,1.5));
	BOOST_TEST(y.derivative(3) == (2*cx*cx+1)/std::pow(1-cx*cx,2.5));
	BOOST_TEST(y.derivative(4) == 3*cx*(2*cx*cx+3)/std::pow(1-cx*cx,3.5));
	BOOST_TEST(y.derivative(5) == (24*(cx*cx+3)*cx*cx+9)/std::pow(1-cx*cx,4.5));
}

BOOST_AUTO_TEST_CASE(asin_infinity)
{
	constexpr int m = 5;
	auto x = boost::math::autodiff::variable<double,m>(1);
	auto y = boost::math::autodiff::asin(x);
	//std::cout << "boost::math::autodiff::asin(1) = " << y << std::endl; // depth(1)(1.5707963267949,inf,inf,-nan,-nan,-nan)
	BOOST_TEST(y.derivative(0) == M_PI_2);
	BOOST_TEST(y.derivative(1) == std::numeric_limits<double>::infinity());
}

BOOST_AUTO_TEST_CASE(asin_derivative, * boost::unit_test::tolerance(1e-15))
{
	constexpr int m = 4;
	constexpr double cx = 0.5;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = 1-x*x;
	BOOST_TEST(y.derivative(0) == 1-cx*cx);
	BOOST_TEST(y.derivative(1) == -2*cx);
	BOOST_TEST(y.derivative(2) == -2);
	BOOST_TEST(y.derivative(3) == 0);
	BOOST_TEST(y.derivative(4) == 0);
	y = boost::math::autodiff::sqrt(y);
	BOOST_TEST(y.derivative(0) == std::sqrt(1-cx*cx));
	BOOST_TEST(y.derivative(1) == -cx/std::sqrt(1-cx*cx));
	BOOST_TEST(y.derivative(2) == -1/std::pow(1-cx*cx,1.5));
	BOOST_TEST(y.derivative(3) == -3*cx/std::pow(1-cx*cx,2.5));
	BOOST_TEST(y.derivative(4) == -(12*cx*cx+3)/std::pow(1-cx*cx,3.5));
	y = y.inverse(); // asin'(x) = 1 / sqrt(1-x*x).
	BOOST_TEST(y.derivative(0) == 1/std::sqrt(1-cx*cx));
	BOOST_TEST(y.derivative(1) == cx/std::pow(1-cx*cx,1.5));
	BOOST_TEST(y.derivative(2) == (2*cx*cx+1)/std::pow(1-cx*cx,2.5));
	BOOST_TEST(y.derivative(3) == 3*cx*(2*cx*cx+3)/std::pow(1-cx*cx,3.5));
	BOOST_TEST(y.derivative(4) == (24*(cx*cx+3)*cx*cx+9)/std::pow(1-cx*cx,4.5));
}

BOOST_AUTO_TEST_CASE(tan, * boost::unit_test::tolerance(2e-15))
{
	constexpr int m = 5;
	constexpr double cx = M_PI/3;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::tan(x);
	BOOST_TEST(y.derivative(0) == std::sqrt(3));
	BOOST_TEST(y.derivative(1) == 4.0);
	BOOST_TEST(y.derivative(2) == 8*std::sqrt(3));
	BOOST_TEST(y.derivative(3) == 80.0);
	BOOST_TEST(y.derivative(4) == 352*std::sqrt(3));
	BOOST_TEST(y.derivative(5) == 5824.0);
}

BOOST_AUTO_TEST_CASE(atan)
{
	constexpr int m = 5;
	constexpr double cx = 1.0;
	const auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::atan(x);
	BOOST_TEST(y.derivative(0) == M_PI_4);
	BOOST_TEST(y.derivative(1) == 0.5);
	BOOST_TEST(y.derivative(2) == -0.5);
	BOOST_TEST(y.derivative(3) == 0.5);
	BOOST_TEST(y.derivative(4) == 0.0);
	BOOST_TEST(y.derivative(5) == -3.0);
}

BOOST_AUTO_TEST_CASE(fmod)
{
	constexpr int m = 3;
	constexpr double cx = 3.25;
	constexpr double cy = 0.5;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::fmod(x,cy);
	BOOST_TEST(y.derivative(0) == 0.25);
	BOOST_TEST(y.derivative(1) == 1.0);
	BOOST_TEST(y.derivative(2) == 0.0);
	BOOST_TEST(y.derivative(3) == 0.0);
}

BOOST_AUTO_TEST_CASE(round_and_trunc)
{
	constexpr int m = 3;
	constexpr double cx = 3.25;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	auto y = boost::math::autodiff::round(x);
	BOOST_TEST(y.derivative(0) == std::round(cx));
	BOOST_TEST(y.derivative(1) == 0.0);
	BOOST_TEST(y.derivative(2) == 0.0);
	BOOST_TEST(y.derivative(3) == 0.0);
	y = boost::math::autodiff::trunc(x);
	BOOST_TEST(y.derivative(0) == std::trunc(cx));
	BOOST_TEST(y.derivative(1) == 0.0);
	BOOST_TEST(y.derivative(2) == 0.0);
	BOOST_TEST(y.derivative(3) == 0.0);
}

BOOST_AUTO_TEST_CASE(lround_llround_truncl)
{
	constexpr int m = 3;
	constexpr double cx = 3.25;
	auto x = boost::math::autodiff::variable<double,m>(cx);
	long yl = boost::math::autodiff::lround(x);
	BOOST_TEST(yl == std::lround(cx));
	long long yll = boost::math::autodiff::llround(x);
	BOOST_TEST(yll == std::llround(cx));
	long double yld = boost::math::autodiff::truncl(x);
	BOOST_TEST(yld == std::truncl(cx));
}

BOOST_AUTO_TEST_CASE(mixed_partials, * boost::unit_test::tolerance(1e-12))
{
    // Derivatives calculated from symbolic differentiation by Mathematica for comparison.
    const double answers[] = {19878.406289804349223,20731.748382749395173,14667.607676239390148,1840.5599364498131187,-9219.3180052370721296,-7272.3006340128117838,-2135.2963700622839242,3095.0810272518467995,4249.0267629086156274,2063.9890610627344166,-885.52841148764960841,-1962.1334204417431580,-1846.8998307870845186,-160.95901276032957552,1091.0394123416339941,452.43955743452299467,666.40139227277049900,-415.64641143336291078,-625.14641790399863613,369.94916697726171101,-24330.896138493893431,-18810.416051756267521,-4890.4061227023590999,8833.0050547689764171,8484.3507396816137478,3097.2041512403988935,-3255.0451367834406121,-4342.7785533321930979,-2407.9872379065234860,861.11739164703000843,2436.7437257633086191,-19.246496107338277838,187.78551488705117144,-1259.4660633352121952,-709.68605239721582613,1423.0005586086045369,484.92081333892339591,763.97468850744531805,-327.41629182280555682,-1122.3377072484945211,23973.060071923469893,8840.5431517787968699,-9082.5710332215493783,-12270.273782892587177,-4320.4340714205998547,3281.3519677072808985,5880.3362630834187672,-1288.4827852197065498,-803.97135376265805266,-2986.3872453316983903,-586.73168598226583063,3929.0731892807393562,1453.7282809838266301,1037.8780716859538297,-1482.7458052774013366,-1877.1347929338288106,-931.71387103692982071,254.65655904203226329,1391.2480647456116638,-431.48205631541379551,16975.340053651795550,19662.603563033417098,15765.851307040200043,3972.1550361959370138,-8681.7485397897205125,-7703.1830424603876567,-3049.7086965695187740,2971.4696859922708762,4370.1964998575500257,2524.6324733574356708,-656.60800002366790717,-2423.4529173252581326,-2074.9876642042632042,-381.22537949881329845,1219.5072457919973510,805.38022398408368773,838.40041900589123805,-390.61251971089838316,-828.20854892982357583,293.89998544549947901,-22965.859858439519778,-20026.691015299296217,-7316.0927450633559965,8632.4661339726146593,8987.0468828704522662,4199.9253995361375411,-2958.4298508960628932,-5665.5638912186240622,-2945.4045522503416159,555.65662724782625247,2936.7964035500791392,651.51916507471100081,444.76294274861551486,-1390.9896717990958013,-1142.8614689467638609,1541.9787231173408435,455.71460632938144702,998.79435039403570373,-204.84855819811212954,-1560.3541154604787861,25278.294506052472235,11873.223371790464699,-8242.1873033688781033,-15939.980564174657519,-5648.8335396980314868,2751.5139261227171185,7349.4320024790771292,194.99725459803711274,-402.81568576826882656,-3518.8719086830633712,-1494.3047934746826191,4640.9275094260800875,1585.7577052032271420,1565.1699924044071379,-1513.2598097335400189,-2974.4378726746800928,-1203.2362926538234416,72.524259498791533840,1871.6252742534199495,-2.4899843373796816664,14462.744235186331026,18367.747409164327117,16565.763244996739614,6054.3152526511029520,-8084.9812719820301461,-7988.3143591282012972,-3989.3193469414926985,2616.7211865346490167,4420.8592709704865621,2973.0335197645479091,-324.14530169827137080,-2843.2420399589692219,-2281.4618061432895177,-642.93532295820559249,1299.2872741769553585,1238.5970833720697622,1021.3340427708481651,-329.05293450692710796,-1046.2543015440520751,134.73430395544806552,-21431.416435076611924,-20856.882814790157847,-9829.2619705919309076,7806.8586470778118280,9319.7000856495681801,5319.8987680257582564,-2387.9548264668417364,-6958.2985251653597607,-3468.5391063919725607,130.41672533427094017,3371.1399302351759874,1569.2326780049081053,750.09121011790652458,-1462.2572096265974522,-1661.5778096302406157,1509.6285286038691333,383.89509025808162595,1248.0510963436380133,17.185695642652602749,-2038.0245980026048531,26118.981320178235148,14943.619434822279033,-6650.6862622761310724,-19519.815295474040679,-6983.1902365008486475,1899.2975028736889830,8715.0036526429634882,2368.1506906818643019,136.89207930934828319,-3954.7327061634171420,-2673.5564402311867864,5078.4839352490435947,1643.4591437212048172,2182.2169795063802937,-1345.8388309636205015,-4309.2853506291084135,-1488.0508699224178177,-228.05849430703437209,2373.3989404257091779,773.84813281039280582,12294.403877378555486,16977.349665718583019,17057.174756225031750,8121.1897585118309359,-7458.4435414062843899,-8134.1311608827380587,-4912.8811586137844196,2030.6531360989337179,4407.4905277094127309,3392.4345688258927524,104.03723558415061987,-3180.8176204844632144,-2460.5239870750694373,-938.22093140691334328,1315.2469055718764567,1735.8623924059921882,1209.7596572231669549,-227.33200545666422971,-1266.1262099919292594,-123.07945723381491568,-19806.907943338346855,-21314.816354405752293,-12317.583844301308050,6349.4186598882814744,9489.8196876965277351,6409.5389484563099944,-1550.2817990131252676,-8109.7111997852175121,-3957.8403302968748777,-404.07965558366678588,3693.6143513011819801,2716.1466583227900648,1094.5910866413989005,-1456.2696455499464209,-2244.3806087356369623,1268.5938915562618711,265.22067303277493466,1496.0915787786394884,354.61373510477227819,-2508.4771100486841292,26517.861408751573247,17922.983877419151441,-4328.2591421276680409,-22704.702459400809491,-8268.6137471737389714,740.40560743926114647,9848.9001828360350810,5213.5983414762103377,801.24629237235082333,-4241.8701339207678459,-4092.2413558685505706,5074.4359092060839438,1607.7653292548209160,2861.1556511165675262,-918.93105463172960902,-5803.2113236460920193,-1767.5418979944773144,-663.06462075200757263,2837.9031946139384145,1976.3196007477977178};
    constexpr int Nw=3;
    constexpr int Nx=2;
    constexpr int Ny=4;
    constexpr int Nz=3;
    const boost::math::autodiff::variable<double,Nw> w(11);
    const boost::math::autodiff::variable<double,0,Nx> x(12);
    const boost::math::autodiff::variable<double,0,0,Ny> y(13);
    const boost::math::autodiff::variable<double,0,0,0,Nz> z(14);
    const auto v = mixed_partials_f(w,x,y,z); // auto = boost::math::autodiff::variable<double,Nw,Nx,Ny,Nz>
    int ia=0;
    for (int iw=0 ; iw<=Nw ; ++iw)
        for (int ix=0 ; ix<=Nx ; ++ix)
            for (int iy=0 ; iy<=Ny ; ++iy)
                for (int iz=0 ; iz<=Nz ; ++iz)
                    BOOST_TEST(v.derivative(iw,ix,iy,iz) == answers[ia++]);
}

BOOST_AUTO_TEST_SUITE_END()
