//
// Created by AK on 13/10/15.
//

#include "gtest/gtest.h"
#include "symbolic.h"


TEST(monomials, test_init) {
    EXPECT_EQ(1, 1);
}

TEST(basic_check2, test_neq) {
    EXPECT_NE(1, 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}