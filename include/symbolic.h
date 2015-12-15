//
// Created by alex on 23/11/15.
//

#ifndef AUTODIFF_SYMBOLIC_H
#define AUTODIFF_SYMBOLIC_H


#include <stddef.h>
#include <exception>
#include <algorithm>
#include <bits/stringfwd.h>
#include <bits/stl_bvector.h>
#include "iostream"

namespace metadiff {
    namespace symbolic {

        class NonIntegerDivision : public std::exception {
            virtual const char *what() const throw() {
                return "The requested division leads to a non integer monomial/polynomial";
            }
        };

        class UnrecognisedSymbolicVariable : public std::exception {
            virtual const char *what() const throw() {
                return "Requested a symbolic variable >= N";
            }
        };

        class NonAConstant : public std::exception {
            virtual const char *what() const throw() {
                return "Trying to evaluate a non constant monomial/polynomial without specified values";
            }
        };

        template<const size_t N, typename T>
        class SymbolicMonomial {
            static_assert(std::numeric_limits<T>::is_integer, "X can be only instantiated with integer types");
            static_assert(not std::numeric_limits<T>::is_signed, "X can be only instansiated with unsigned types");
        public:
            std::array<T, N> powers;
            int coefficient;

            SymbolicMonomial() {
                for (int i = 0; i < N; i++) {
                    this->powers[i] = 0;
                }
                this->coefficient = 1;
            }

            SymbolicMonomial(const SymbolicMonomial<N, T> &monomial) {
                this->powers = monomial.powers;
                this->coefficient = monomial.coefficient;
            }

            SymbolicMonomial(const int value) : SymbolicMonomial() {
                this->coefficient = value;
            }

            static SymbolicMonomial variable(const size_t variable) {
                if (variable >= N) {
                    throw UnrecognisedSymbolicVariable();
                }
                auto result = SymbolicMonomial<N, T>();
                result.powers[variable] = 1;
                return result;
            }

            static SymbolicMonomial one() {
                return SymbolicMonomial(1);
            }

            static SymbolicMonomial zero() {
                return SymbolicMonomial(0);
            }

            bool is_constant() const {
                for (int i = 0; i < N; i++) {
                    if (this->powers[i] > 0) {
                        return false;
                    }
                }
                return true;
            }

            int eval(std::array<T, N> &values) {
                int value = 0;
                for (int i = 0; i < N; i++) {
                    value += pow(values[i], this->powers[i]);
                }
                return value * this->coefficient;
            }

            int eval() {
                if (not is_constant()) {
                    throw NonAConstant();
                }
                return coefficient;
            }

            std::string to_string() const {
                if (this->coefficient == 0) {
                    return "0";
                }
                std::string result;
                if (this->coefficient != 1) {
                    if (this->coefficient == -1) {
                        result += "-";
                    } else {
                        result += std::to_string(this->coefficient);
                    }
                }
                for (int i = 0; i < N; i++) {
                    if (powers[i] > 0) {
                        result += 'a' + i;
                        if (powers[i] > 1) {
                            auto n = powers[i];
                            std::string supercripts;
                            while (n > 0) {
                                auto reminder = n % 10;
                                n /= 10;
                                switch (reminder) {
                                    case 0:
                                        supercripts = "\u2070" + supercripts;
                                        break;
                                    case 1:
                                        supercripts = "\u00B9" + supercripts;
                                        break;
                                    case 2:
                                        supercripts = "\u00B2" + supercripts;
                                        break;
                                    case 3:
                                        supercripts = "\u00B3" + supercripts;
                                        break;
                                    case 4:
                                        supercripts = "\u2074" + supercripts;
                                        break;
                                    case 5:
                                        supercripts = "\u2075" + supercripts;
                                        break;
                                    case 6:
                                        supercripts = "\u2076" + supercripts;
                                        break;
                                    case 7:
                                        supercripts = "\u2077" + supercripts;
                                        break;
                                    case 8:
                                        supercripts = "\u2078" + supercripts;
                                        break;
                                    case 9:
                                        supercripts = "\u2079" + supercripts;
                                        break;
                                }
                            }
                            result += supercripts;
                        }
                    }
                }
                if (result == "") {
                    return "1";
                } else {
                    return result;
                }
            }
        };

        template<const size_t N, typename T>
        std::ostream &operator<<(std::ostream &f, const SymbolicMonomial<N, T> &value) {
            f << value.to_string();
            return f;
        }

        template<size_t N, typename T>
        bool operator==(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            if (lhs.coefficient != rhs.coefficient) {
                return false;
            }
            for (int i = 0; i < N; i++) {
                if (lhs.powers[i] != rhs.powers[i]) {
                    return false;
                }
            }
            return true;
        }

        template<size_t N, typename T>
        bool operator==(const SymbolicMonomial<N, T> &lhs, const int rhs) {
            return lhs.is_constant() and lhs.coefficient == rhs;
        }


        template<size_t N, typename T>
        bool operator==(const int lhs, const SymbolicMonomial<N, T> &rhs) {
            return (rhs == lhs);
        }

        template<size_t N, typename T>
        bool operator!=(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            return not (lhs == rhs);
        }

        template<size_t N, typename T>
        bool operator!=(const int lhs, const SymbolicMonomial<N, T> &rhs) {
            return not (lhs == rhs);
        }

        template<size_t N, typename T>
        bool operator!=(const SymbolicMonomial<N, T> &lhs, const int rhs) {
            return not (lhs == rhs);
        }

        template<size_t N, typename T>
        bool up_to_coefficient(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            for (int i = 0; i < N; i++) {
                if (lhs.powers[i] != rhs.powers[i]) {
                    return false;
                }
            }
            return true;
        }

        template<size_t N, typename T>
        bool up_to_coefficient(const int lhs, const SymbolicMonomial<N, T> &rhs) {
            return rhs.is_constant();
        }

        template<size_t N, typename T>
        bool up_to_coefficient(const SymbolicMonomial<N, T> &lhs, const int rhs) {
            return lhs.is_constant();
        }

        template<size_t N, typename T>
        bool less_than_comparator(const SymbolicMonomial<N, T> &monomial1, const SymbolicMonomial<N, T> &monomial2) {
            for (int i = 0; i < N; i++) {
                if (monomial1.powers[i] < monomial2.powers[i]) {
                    return false;
                } else if (monomial1.powers[i] > monomial2.powers[i]) {
                    return true;
                }
            }
            return false;
        }

        template<const size_t N, typename T>
        SymbolicMonomial<N, T> operator+(const SymbolicMonomial<N, T> &rhs) {
            return rhs;
        }

        template<const size_t N, typename T>
        SymbolicMonomial<N, T> operator-(const SymbolicMonomial<N, T> &rhs) {
            auto result = SymbolicMonomial<N, T>(rhs);
            result.coefficient = -result.coefficient;
            return result;
        }

        template<const size_t N, typename T>
        SymbolicMonomial<N, T> operator*(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            auto result = SymbolicMonomial<N, T>(lhs);
            for (int i = 0; i < N; i++) {
                result.powers[i] += rhs.powers[i];
            }
            result.coefficient *= rhs.coefficient;
            return result;
        }

        template<size_t N, typename T>
        SymbolicMonomial<N, T> operator*(const SymbolicMonomial<N, T> &lhs, const int rhs) {
            auto result = SymbolicMonomial<N, T>(lhs);
            result.coefficient *= rhs;
            return result;
        }

        template<size_t N, typename T>
        SymbolicMonomial<N, T> operator*(const int lhs, const SymbolicMonomial<N, T> &rhs) {
            return rhs * lhs;
        }

        template<size_t N, typename T>
        SymbolicMonomial<N, T> operator/(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            auto result = SymbolicMonomial<N, T>(lhs);
            for (int i = 0; i < N; i++) {
                result.powers[i] -= rhs.powers[i];
                if (result.powers[i] > lhs.powers[i]) {
                    throw NonIntegerDivision();
                }
            }
            if (rhs.coefficient == 0 or result.coefficient % rhs.coefficient != 0) {
                throw NonIntegerDivision();
            }
            result.coefficient = lhs.coefficient / rhs.coefficient;
            return result;
        }

        template<size_t N, typename T>
        SymbolicMonomial<N, T> operator/(const SymbolicMonomial<N, T> &lhs, int rhs) {
            if (rhs == 0 or lhs.coefficient % rhs != 0) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicMonomial<N, T>(lhs);
            result.coefficient /= rhs;
            return result;
        }

        template<size_t N, typename T>
        SymbolicMonomial<N, T> operator/(const int lhs, const SymbolicMonomial<N, T> &rhs) {
            if (not rhs.is_constant() or rhs.coefficient == 0 or lhs % rhs.coefficient != 0) {
                throw NonIntegerDivision();
            }
            return SymbolicMonomial<N, T>(lhs / rhs.coefficient);
        }


        template<size_t N, typename T>
        class SymbolicPolynomial {
        public:
            std::vector<SymbolicMonomial<N, T>> monomials;

            SymbolicPolynomial() { };

            SymbolicPolynomial(const SymbolicPolynomial<N, T> &polynomial) {
                this->monomials = polynomial.monomials;
            }

            SymbolicPolynomial(const SymbolicMonomial<N, T> &monomial) {
                this->monomials.push_back(monomial);
            }

            SymbolicPolynomial(const int value) {
                if (value != 0) {
                    this->monomials.push_back(SymbolicMonomial<N, T>(value));
                }
            }

            static SymbolicPolynomial variable(const size_t variable) {
                return SymbolicPolynomial(SymbolicMonomial<N, T>::variable(variable));
            }

            bool is_constant() const {
                if (this->monomials.size() > 1) {
                    return false;
                } else if (this->monomials.size() == 1) {
                    return this->monomials[0].is_constant();
                } else {
                    return true;
                }
            }

            void simplify() {
                std::sort(this->monomials.begin(), this->monomials.end(), less_than_comparator<N, T>);
                for (int i = 1; i < this->monomials.size(); i++) {
                    if (up_to_coefficient(this->monomials[i - 1], this->monomials[i])) {
                        this->monomials[i - 1].coefficient += this->monomials[i].coefficient;
                        this->monomials.erase(this->monomials.begin() + i);
                        i--;
                    }
                }
                for (int i = 0; i < this->monomials.size(); i++) {
                    if (this->monomials[i].coefficient == 0) {
                        this->monomials.erase(this->monomials.begin() + i);
                        i--;
                    }
                }
            }

            int eval(std::array<int, N> &values) {
                int value = 0;
                for (int i = 0; i < this->monomials.size(); i++) {
                    value += monomials[i].eval(values);
                }
                return value;
            }

            int eval() {
                if (not is_constant()) {
                    throw NonAConstant();
                }
                if (monomials.size() == 0) {
                    return 0;
                } else {
                    return monomials[0].coefficient;
                }
            }

            std::string to_string() const {
                if (this->monomials.size() == 0) {
                    return "0";
                }
                auto result = this->monomials[0].to_string();
                for (int i = 1; i < this->monomials.size(); i++) {
                    if (this->monomials[i].coefficient > 0) {
                        result += "+" + this->monomials[i].to_string();
                    } else {
                        result += this->monomials[i].to_string();
                    }
                }
                return result;
            }

            static SymbolicPolynomial one() {
                return SymbolicPolynomial(SymbolicMonomial<N, T>::one());
            };

            static SymbolicPolynomial zero() {
                return SymbolicPolynomial(SymbolicMonomial<N, T>::zero());
            };
        };

        template<const size_t N, typename T>
        std::ostream &operator<<(std::ostream &f, const SymbolicPolynomial<N, T> &polynomial) {
            f << polynomial.to_string();
            return f;
        }

        template<const size_t N, typename T>
        bool operator==(const SymbolicPolynomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            if (lhs.monomials.size() != rhs.monomials.size()) {
                return false;
            }
            for (int i = 0; i < lhs.monomials.size(); i++) {
                if (lhs.monomials[i] != rhs.monomials[i]) {
                    return false;
                }
            }
            return true;
        }

        template<const size_t N, typename T>
        bool operator==(const SymbolicPolynomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
        }

        template<const size_t N, typename T>
        bool operator==(const SymbolicMonomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            return rhs == lhs;
        }

        template<const size_t N, typename T>
        bool operator==(const SymbolicPolynomial<N, T> &lhs, const int rhs) {
            return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
        }

        template<const size_t N, typename T>
        bool operator==(const int lhs, const SymbolicPolynomial<N, T> &rhs) {
            return rhs == lhs;
        }

        template<const size_t N, typename T>
        bool operator!=(const SymbolicPolynomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            return not (lhs == rhs);
        }

        template<const size_t N, typename T>
        bool operator!=(const SymbolicPolynomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            return not (lhs == rhs);
        }

        template<const size_t N, typename T>
        bool operator!=(const SymbolicMonomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            return not (lhs == rhs);
        }

        template<const size_t N, typename T>
        bool operator!=(const SymbolicPolynomial<N, T> &lhs, const int rhs) {
            return not (lhs == rhs);
        }

        template<const size_t N, typename T>
        bool operator!=(const int lhs, const SymbolicPolynomial<N, T> &rhs) {
            return not (lhs == rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const SymbolicPolynomial<N, T> &rhs) {
            return rhs;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const SymbolicPolynomial<N, T> &rhs) {
            SymbolicPolynomial<N, T> result = SymbolicPolynomial<N, T>(rhs);
            for (int i = 0; i < rhs.monomials.size(); i++) {
                result.monomials[i].coefficient = -result.monomials[i].coefficient;
            }
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            auto result = SymbolicPolynomial<N, T>();
            result.monomials.push_back(lhs);
            result.monomials.push_back(rhs);
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const SymbolicMonomial<N, T> &lhs, const int rhs) {
            auto result = SymbolicPolynomial<N, T>();
            result.monomials.push_back(lhs);
            result.monomials.push_back(SymbolicMonomial<N, T>(rhs));
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const int lhs, const SymbolicMonomial<N, T> &rhs) {
            return rhs + lhs;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const SymbolicPolynomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            auto result = SymbolicPolynomial<N, T>(lhs);
            result.monomials.insert(result.monomials.end(), rhs.monomials.begin(), rhs.monomials.end());
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const SymbolicPolynomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            auto result = SymbolicPolynomial<N, T>(lhs);
            result.monomials.push_back(rhs);
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const SymbolicMonomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            return rhs + lhs;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const SymbolicPolynomial<N, T> &lhs, const int rhs) {
            auto result = SymbolicPolynomial<N, T>(lhs);
            result.monomials.push_back(SymbolicMonomial<N, T>(rhs));
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator+(const int lhs, const SymbolicPolynomial<N, T> &rhs) {
            return rhs + lhs;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const SymbolicMonomial<N, T> &lhs, const int rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const int lhs, const SymbolicMonomial<N, T> rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const SymbolicPolynomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const SymbolicPolynomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const SymbolicMonomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const SymbolicPolynomial<N, T> &lhs, const int rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator-(const int lhs, const SymbolicPolynomial<N, T> &rhs) {
            return lhs + (-rhs);
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator*(const SymbolicPolynomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            auto result = SymbolicPolynomial<N, T>();
            for (int i = 0; i < lhs.monomials.size(); i++) {
                for (int j = 0; j < rhs.monomials.size(); j++) {
                    result.monomials.push_back(lhs.monomials[i] * rhs.monomials[j]);
                }
            }
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator*(const SymbolicPolynomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            auto result = SymbolicPolynomial<N, T>();
            for (int i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] * rhs);
            }
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator*(const SymbolicMonomial<N, T> lhs, const SymbolicPolynomial<N, T> rhs) {
            return rhs * lhs;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator*(const SymbolicPolynomial<N, T> &lhs, int rhs) {
            auto result = SymbolicPolynomial<N, T>();
            for (int i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] * rhs);
            }
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator*(int lhs, const SymbolicPolynomial<N, T> rhs) {
            return rhs * lhs;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator/(const SymbolicPolynomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            auto result = SymbolicPolynomial<N, T>();
            auto reminder = SymbolicPolynomial<N, T>(lhs);
            while (not reminder.is_constant()) {
                result.monomials.push_back(reminder.monomials[0] / rhs.monomials[0]);
                auto s = rhs * result.monomials.back();
                reminder = reminder - s;
            }
            if (reminder != 0) {
                throw NonIntegerDivision();
            }
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator/(const SymbolicPolynomial<N, T> &lhs, const SymbolicMonomial<N, T> &rhs) {
            auto result = SymbolicPolynomial<N, T>();
            for (int i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] / rhs);
            }
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator/(const SymbolicMonomial<N, T> &lhs, const SymbolicPolynomial<N, T> &rhs) {
            if (rhs.monomials.size() != 1) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicPolynomial<N, T>();
            result.monomials.push_back(lhs / rhs.monomials[0]);
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator/(const SymbolicPolynomial<N, T> &lhs, const int rhs) {
            if (rhs == 0) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicPolynomial<N, T>();
            for (int i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] / rhs);
            }
            result.simplify();
            return result;
        }

        template<const size_t N, typename T>
        SymbolicPolynomial<N, T> operator/(const int lhs, const SymbolicPolynomial<N, T> rhs) {
            if (rhs.monomials.size() != 1) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicPolynomial<N, T>();
            result.monomials.push_back(lhs / rhs.monomials[0]);
            return result;
        }
    }
}
#endif //AUTODIFF_SYMBOLIC_H
