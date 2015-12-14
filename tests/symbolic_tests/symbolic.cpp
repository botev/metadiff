//
// Created by AK on 13/10/15.
//

#include "gtest/gtest.h"
#include "symbolic.h"
#include "gmock/gmock.h"

template <typename T>
class SymbolicTest : public testing::Test {};

const size_t N = 5;
typedef testing::Types<unsigned short, unsigned int, unsigned long> Integers;

TYPED_TEST_CASE(SymbolicTest, Integers);

TYPED_TEST(SymbolicTest, MonomialConstructor){
    typedef symbolic::SymbolicMonomial<N, TypeParam> Monomial;
    // Constant monomial 1
    auto one = Monomial();
    EXPECT_EQ(one.coefficient, 1);
    EXPECT_TRUE(one.is_constant());
    EXPECT_THAT(one.powers, testing::ElementsAre(0, 0, 0, 0, 0));

    // Monomial with 1 variable
    auto x = Monomial::new_variable(0);
    EXPECT_EQ(x.coefficient, 1);
    EXPECT_FALSE(x.is_constant());
    EXPECT_THAT(x.powers, testing::ElementsAre(1, 0, 0, 0, 0));

    // From constant
    auto two = Monomial(2);
    EXPECT_EQ(two.coefficient, 2);
    EXPECT_TRUE(two.is_constant());
    EXPECT_THAT(two.powers, testing::ElementsAre(0, 0, 0, 0, 0));

    two.powers[0] = 2;
    // From another
    auto two_x2 = Monomial(two);
    EXPECT_EQ(two_x2.coefficient, 2);
    EXPECT_FALSE(two_x2.is_constant());
    EXPECT_THAT(two_x2.powers, testing::ElementsAre(2, 0, 0, 0, 0));

    // Test for exception
    EXPECT_THROW(Monomial::new_variable(N+0), symbolic::UnrecognisedSymbolicVariable);
}

TYPED_TEST(SymbolicTest, MonomialEquality) {
    typedef symbolic::SymbolicMonomial<N, TypeParam> Monomial;
    // Equality with integers
    auto two = Monomial(2);
    EXPECT_EQ(two, 2);
    EXPECT_EQ(2, two);
    EXPECT_NE(two, 1);
    EXPECT_NE(1, two);

    // Not equality between 'x' and a constant
    auto two_2 = Monomial(2);
    auto x = Monomial::new_variable(0);
    EXPECT_EQ(two, two_2);
    EXPECT_EQ(two_2, two);
    EXPECT_NE(two, x);
    EXPECT_NE(x, two);
    EXPECT_NE(two_2, x);
    EXPECT_NE(x, two_2);

    // Up to coefficient equality for constants
    EXPECT_TRUE(symbolic::up_to_coefficient(0, two));
    EXPECT_TRUE(symbolic::up_to_coefficient(0, two_2));
    EXPECT_FALSE(symbolic::up_to_coefficient(0, x));

    // Up to coefficient equality for 'x' and '10x'
    auto ten_x = Monomial::new_variable(0) * 10;
    EXPECT_TRUE(symbolic::up_to_coefficient(x, ten_x));
    EXPECT_TRUE(symbolic::up_to_coefficient(ten_x, x));
}

TYPED_TEST(SymbolicTest, MonomialProductAndDivision) {
    typedef symbolic::SymbolicMonomial<N, TypeParam> Monomial;
    auto x = Monomial::new_variable(0);
    auto y = Monomial::new_variable(1);
    auto z = Monomial::new_variable(2);
    auto composite = 2 * x * y * z;

    // Verify inner structure
    EXPECT_EQ(composite.coefficient, 2);
    EXPECT_FALSE(composite.is_constant());
    EXPECT_THAT(composite.powers, testing::ElementsAre(1, 1, 1, 0, 0));

    // Check equality between different forms
    EXPECT_EQ(composite / 2, x * y * z);
    EXPECT_EQ(composite / x, 2 * y * z);
    EXPECT_EQ(composite / y, 2 * x * z);
    EXPECT_EQ(composite / z, 2 * x * y);

    // Check for errors on non integer division
    EXPECT_THROW(composite / 4, symbolic::NonIntegerDivision);
    EXPECT_THROW(composite / (x * x), symbolic::NonIntegerDivision);
    EXPECT_THROW(composite / (y * y), symbolic::NonIntegerDivision);
    EXPECT_THROW(composite / (z * z), symbolic::NonIntegerDivision);
}

TYPED_TEST(SymbolicTest, PolynomialConstructor) {
    typedef symbolic::SymbolicMonomial<N, TypeParam> Monomial;
    typedef symbolic::SymbolicPolynomial<N, TypeParam> Polynomial;
    // Constant polynomial 0
    auto zero = Polynomial();
    EXPECT_EQ(zero.monomials.size(), 0);
    EXPECT_TRUE(zero.is_constant());

    // Polynomial with 1 variable
    auto x = Polynomial::new_variable(0);
    EXPECT_EQ(x.monomials.size(), 1);
    EXPECT_EQ(x.monomials[0].coefficient, 1);
    EXPECT_FALSE(x.is_constant());
    EXPECT_THAT(x.monomials[0].powers, testing::ElementsAre(1, 0, 0, 0, 0));

    // From constant
    auto two = Polynomial(2);
    EXPECT_EQ(two.monomials.size(), 1);
    EXPECT_EQ(two.monomials[0].coefficient, 2);
    EXPECT_TRUE(two.is_constant());
    EXPECT_THAT(two.monomials[0].powers, testing::ElementsAre(0, 0, 0, 0, 0));

    // From monomial
    auto x_monomial = Monomial::new_variable(0);
    x = Polynomial(x_monomial);
    EXPECT_EQ(x.monomials.size(), 1);
    EXPECT_EQ(x.monomials[0].coefficient, 1);
    EXPECT_FALSE(x.is_constant());
    EXPECT_THAT(x.monomials[0].powers, testing::ElementsAre(1, 0, 0, 0, 0));

    x.monomials[0].coefficient = 2;
    // From another
    auto two_x = Polynomial(x);
    EXPECT_EQ(two_x.monomials.size(), 1);
    EXPECT_EQ(two_x.monomials[0].coefficient, 2);
    EXPECT_FALSE(two_x.is_constant());
    EXPECT_THAT(two_x.monomials[0].powers, testing::ElementsAre(1, 0, 0, 0, 0));

    // Test for exception
    EXPECT_THROW(Polynomial::new_variable(N+0), symbolic::UnrecognisedSymbolicVariable);
}

TYPED_TEST(SymbolicTest, PolynomialEquality) {
    typedef symbolic::SymbolicMonomial<N, TypeParam> Monomial;
    typedef symbolic::SymbolicPolynomial<N, TypeParam> Polynomial;
    // Equality with integers
    auto two = Polynomial(2);
    EXPECT_EQ(two, 2);
    EXPECT_EQ(2, two);
    EXPECT_NE(two, 1);
    EXPECT_NE(1, two);

    // Not equality between 'x' and a constant
    auto two_2 = Polynomial(2);
    auto x = Polynomial::new_variable(0);
    EXPECT_EQ(two, two_2);
    EXPECT_EQ(two_2, two);
    EXPECT_NE(two, x);
    EXPECT_NE(x, two);
    EXPECT_NE(two_2, x);
    EXPECT_NE(x, two_2);

    // Equality with 'x' as monomial
    auto x_monomial = Monomial::new_variable(0);
    EXPECT_EQ(x, x_monomial);
    EXPECT_EQ(x_monomial, x);

    // Non equality with 'y' as monomial
    auto y_monomial = Monomial::new_variable(1);
    EXPECT_NE(x, y_monomial);
    EXPECT_NE(y_monomial, x);

    // Equality and non equality between polynomials
    auto x_again = Polynomial::new_variable(0);
    auto y = Polynomial::new_variable(1);
    EXPECT_EQ(x, x_again);
    EXPECT_EQ(x_again, x);
    EXPECT_NE(x, y);
    EXPECT_NE(y, x);
}

TYPED_TEST(SymbolicTest, PolynomialAddition) {
    typedef symbolic::SymbolicMonomial<N, TypeParam> Monomial;
    typedef symbolic::SymbolicPolynomial<N, TypeParam> Polynomial;

    // Compare x + y + 2
    auto x_monomial = Monomial::new_variable(0);
    auto x = Polynomial::new_variable(0);
    auto y = Polynomial::new_variable(1);
    auto x_plus_y = x + y + 1;
    auto x_plus_y_2 = x_monomial + y + 1;
    EXPECT_FALSE(x_plus_y.is_constant());
    EXPECT_FALSE(x_plus_y_2.is_constant());
    EXPECT_EQ(x_plus_y, x_plus_y_2);
    EXPECT_EQ(x_plus_y.monomials.size(), 3);
    EXPECT_EQ(x_plus_y_2.monomials.size(), 3);
    for(int i=0;i<3;i++) {
        EXPECT_EQ(x_plus_y.monomials[i], x_plus_y_2.monomials[i]);
    }

    auto two_x_plus_y = x_plus_y + x_plus_y_2;
    EXPECT_EQ(two_x_plus_y.monomials.size(),3);
    EXPECT_FALSE(two_x_plus_y.is_constant());
    for(int i=0;i<3;i++) {
        EXPECT_EQ(two_x_plus_y.monomials[i].coefficient, 2);
        EXPECT_TRUE(symbolic::up_to_coefficient(two_x_plus_y.monomials[i], x_plus_y.monomials[i]));
    }

    // Check subtraction
    auto two = two_x_plus_y - 2 * Polynomial::new_variable(0) - 2 * Polynomial::new_variable(1);
    EXPECT_EQ(two, 2);
    EXPECT_EQ(2, two);
    EXPECT_TRUE(two.is_constant());
}

TYPED_TEST(SymbolicTest, PolynomialProductAndDivision) {
    typedef symbolic::SymbolicMonomial<N, TypeParam> Monomial;
    typedef symbolic::SymbolicPolynomial<N, TypeParam> Polynomial;
    auto x = Polynomial::new_variable(0);
    auto y = Polynomial::new_variable(1);
    auto xy_plus_x_square_plus_one = x * y + x * x + 1;
    auto xy_plus_y_square_plus_two = x * y + y * y + 2;
    // x^3y  + 2x^2y^2 + 2x^2 + xy^3 + 3xy + y^2 + 2
    auto product = xy_plus_x_square_plus_one * xy_plus_y_square_plus_two;
    EXPECT_EQ(product.monomials.size(), 7);
    EXPECT_EQ(product.monomials[0].coefficient, 1);
    EXPECT_THAT(product.monomials[0].powers, testing::ElementsAre(3, 1, 0, 0, 0));
    EXPECT_EQ(product.monomials[1].coefficient, 2);
    EXPECT_THAT(product.monomials[1].powers, testing::ElementsAre(2, 2, 0, 0, 0));
    EXPECT_EQ(product.monomials[2].coefficient, 2);
    EXPECT_THAT(product.monomials[2].powers, testing::ElementsAre(2, 0, 0, 0, 0));
    EXPECT_EQ(product.monomials[3].coefficient, 1);
    EXPECT_THAT(product.monomials[3].powers, testing::ElementsAre(1, 3, 0, 0, 0));
    EXPECT_EQ(product.monomials[4].coefficient, 3);
    EXPECT_THAT(product.monomials[4].powers, testing::ElementsAre(1, 1, 0, 0, 0));
    EXPECT_EQ(product.monomials[5].coefficient, 1);
    EXPECT_THAT(product.monomials[5].powers, testing::ElementsAre(0, 2, 0, 0, 0));
    EXPECT_EQ(product.monomials[6].coefficient, 2);
    EXPECT_THAT(product.monomials[6].powers, testing::ElementsAre(0, 0, 0, 0, 0));

    // Check division
    EXPECT_EQ(product / xy_plus_x_square_plus_one, xy_plus_y_square_plus_two);
    EXPECT_EQ(product / xy_plus_y_square_plus_two, xy_plus_x_square_plus_one);

    // Test for exception
    EXPECT_THROW(product / x*x, symbolic::NonIntegerDivision);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}