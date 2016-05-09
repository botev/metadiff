//
// Created by alex on 23/11/15.
//

#ifndef METADIFF_SYMBOLIC_H
#define METADIFF_SYMBOLIC_H


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


        template<typename I, typename P>
        class SymbolicMonomial {
            static_assert(std::numeric_limits<I>::is_integer, "I can be only instantiated with integer types");
            static_assert(not std::numeric_limits<I>::is_signed, "I can be only instantiated with unsigned types");
            static_assert(std::numeric_limits<P>::is_integer, "P can be only instantiated with unsigned types");
            static_assert(not std::numeric_limits<P>::is_signed, "P can be only instantiated with unsigned types");
        public:
//            std::array<T, N> powers;
            /** A power first argument is the id of the variable, the second is the actual power */
            std::vector<std::pair<I,P>> powers;
            /** The constant coefficient */
            long long int coefficient;

            SymbolicMonomial(): coefficient(1) {}

            SymbolicMonomial(const long long int value) : coefficient(value) {}

            SymbolicMonomial(const SymbolicMonomial<I, P> &monomial) {
                this->powers = monomial.powers;
                this->coefficient = monomial.coefficient;
            }

            static SymbolicMonomial variable(const I variable) {
                auto result = SymbolicMonomial<I, P>();
                result.powers.push_back(std::pair<I,P>{variable, 1});
                return result;
            }

            static SymbolicMonomial<I,P> zero;

            static SymbolicMonomial<I,P> one;

            bool is_constant() const {
                return powers.size() == 0;
            }

            template <typename T>
            long long int eval(std::vector<T> &values) {
                T value = 0;
                for (auto i = 0; i < powers.size(); i++) {
                    value += pow(values[powers[i].first], powers[i].second);
                }
                return value * this->coefficient;
            }

            long long int eval() {
                if (not is_constant()) {
                    throw NonAConstant();
                }
                return coefficient;
            }

            std::string to_string() const {
                if (coefficient == 0) {
                    return "0";
                }
                std::string result;
                if (coefficient != 1) {
                    if (coefficient == -1) {
                        result += "-";
                    } else {
                        result += std::to_string(coefficient);
                    }
                }
                for (auto i = 0; i < powers.size(); i++) {
                        result += ('a' + powers[i].first);
                        if (powers[i].second > 1) {
                            auto n = powers[i].second;
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
                if (result == "") {
                    return "1";
                } else {
                    return result;
                }
            }

            std::string to_string_with_star() const {
                if (coefficient == 0) {
                    return "0";
                }
                std::string result;
                bool first = true;
                if (coefficient != 1) {
                    if (coefficient == -1) {
                        result += "-";
                    } else {
                        result += std::to_string(coefficient);
                        first = false;
                    }
                }
                for (int i = 0; i < powers.size(); i++) {
                        char variable = ('a' + powers[i].first);
                        if(not first){
                            result += "*";
                        }
                        for(int j=0;j<powers[i].second;j++){
                            result += variable;
                            if(j < powers[i].second-1){
                                result += "*";
                            }
                        }
                        first = false;
                }
                if (result == "") {
                    return "1";
                } else {
                    return result;
                }
            }
        };

        template<typename I, typename P>
        SymbolicMonomial<I,P> SymbolicMonomial<I,P>::zero = SymbolicMonomial<I,P>();

        template<typename I, typename P>
        SymbolicMonomial<I,P> SymbolicMonomial<I,P>::one = SymbolicMonomial<I,P>(1);

        template<typename I, typename P>
        std::ostream &operator<<(std::ostream &f, const SymbolicMonomial<I, P> &value) {
            f << value.to_string();
            return f;
        }

        template<typename I, typename P>
        bool operator==(const SymbolicMonomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            if (lhs.coefficient != rhs.coefficient or lhs.powers.size() != rhs.powers.size()) {
                return false;
            }
            for(auto i = 0; i < lhs.powers.size(); i++){
                if(lhs.powers[i] != rhs.powers[i]){
                    return false;
                }
            }
            return true;
        }

        template<typename I, typename P>
        bool operator==(const SymbolicMonomial<I, P> &lhs, const long long int rhs) {
            return lhs.is_constant() and lhs.coefficient == rhs;
        }


        template<typename I, typename P>
        bool operator==(const long long int lhs, const SymbolicMonomial<I, P> &rhs) {
            return rhs.is_constant() and rhs.coefficient == lhs;
        }

        template<typename I, typename P>
        bool operator!=(const SymbolicMonomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        bool operator!=(const long long int lhs, const SymbolicMonomial<I, P> &rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        bool operator!=(const SymbolicMonomial<I, P> &lhs, const long long int rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        bool up_to_coefficient(const SymbolicMonomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            if(lhs.powers.size() != rhs.powers.size()){
                return false;
            }
            for(auto i = 0; i < lhs.powers.size(); i++){
                if(lhs.powers[i] != rhs.powers[i]){
                    return false;
                }
            }
            return true;
        }

        template<typename I, typename P>
        bool up_to_coefficient(const long long int lhs, const SymbolicMonomial<I, P> &rhs) {
            return rhs.is_constant();
        }

        template<typename I, typename P>
        bool up_to_coefficient(const SymbolicMonomial<I, P> &lhs, const long long int rhs) {
            return lhs.is_constant();
        }

        /**
         * An monomial m1 is compared to monomial m2 in the following order of precedence:
         * 1. Check if the lowest variable id in m1 and m2 are equal.
         *    - If they are not than whoever has the lowest is "before" the other.
         * 2.If they are equal compare the power of this variable.
         *    - Whoever has the higher is "before" the other.
         * 3. If the they are equal continue with next lowest variable by id.
         * 4. If all are equal compare coefficients.
         *
         * For instance x^2y^1 is "before" 100xy^300, since x^2 <-> x.
         */
        template<typename I, typename P>
        bool less_than_comparator(const SymbolicMonomial<I, P> &monomial1, const SymbolicMonomial<I, P> &monomial2) {
            auto max = monomial1.powers.size() > monomial2.powers.size() ? monomial2.powers.size() : monomial1.powers.size();
            for (auto i = 0; i < max; i++) {
                if(monomial1.powers[i].first < monomial2.powers[i].first){
                    return true;
                } else if(monomial1.powers[i].first > monomial2.powers[i].first){
                    return false;
                } else if (monomial1.powers[i].second > monomial2.powers[i].second) {
                    return true;
                } else if (monomial1.powers[i].second < monomial2.powers[i].second) {
                    return false;
                }
            }
            if(monomial1.powers.size() < monomial2.powers.size()){
                return false;
            } else if(monomial1.powers.size() > monomial2.powers.size()){
                return true;
            } else {
                return monomial1.coefficient > monomial2.coefficient;
            }
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator+(const SymbolicMonomial<I, P> &rhs) {
            return rhs;
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator-(const SymbolicMonomial<I, P> &rhs) {
            auto result = SymbolicMonomial<I, P>(rhs);
            result.coefficient = -result.coefficient;
            return result;
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator*(const SymbolicMonomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            auto result = SymbolicMonomial<I, P>(lhs.coefficient * rhs.coefficient);
            auto i1 = 0;
            auto i2 = 0;
            while(i1 < lhs.powers.size() and i2 < rhs.powers.size()){
                if(lhs.powers[i1].first < rhs.powers[i2].first){
                    result.powers.push_back(lhs.powers[i1]);
                    i1++;
                } else if(lhs.powers[i1].first > rhs.powers[i2].first){
                    result.powers.push_back(rhs.powers[i2]);
                    i2++;
                } else {
                    result.powers.push_back(std::pair<I,P>{lhs.powers[i1].first, lhs.powers[i1].second + rhs.powers[i2].second});
                    i1++;
                    i2++;
                }
            }
            while(i1 < lhs.powers.size()){
                result.powers.push_back(lhs.powers[i1]);
                i1++;
            }
            while(i2 < rhs.powers.size()){
                result.powers.push_back(rhs.powers[i2]);
                i2++;
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator*(const SymbolicMonomial<I, P> &lhs, const long long int rhs) {
            auto result = SymbolicMonomial<I, P>(lhs);
            result.coefficient *= rhs;
            return result;
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator*(const long long int lhs, const SymbolicMonomial<I, P> &rhs) {
            auto result = SymbolicMonomial<I, P>(rhs);
            result.coefficient *= lhs;
            return result;
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator/(const SymbolicMonomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            auto result = SymbolicMonomial<I, P>();
            if(lhs.coefficient < rhs.coefficient or lhs.coefficient % rhs.coefficient != 0){
                throw NonIntegerDivision();
            }
            result.coefficient = lhs.coefficient / rhs.coefficient;
            auto i1=0;
            auto i2=0;
            while(i1 < lhs.powers.size() and i2 < rhs.powers.size()){
                if(lhs.powers[i1].first < rhs.powers[i2].first){
                    result.powers.push_back(lhs.powers[i1]);
                    i1++;
                } else if(lhs.powers[i1].first > rhs.powers[i2].first){
                    throw NonIntegerDivision();
                } else {
                    if (lhs.powers[i1].second < rhs.powers[i2].second) {
                        throw NonIntegerDivision();
                    } else if (lhs.powers[i1].second > rhs.powers[i2].second) {
                        result.powers.push_back(
                                std::pair<I, P>{lhs.powers[i1].first, lhs.powers[i1].second - rhs.powers[i2].second});
                    } 
                    i1++;
                    i2++;
                }
            }
            if(i2 < rhs.powers.size()){
                throw NonIntegerDivision();
            }
            while(i1 < lhs.powers.size()){
                result.powers.push_back(lhs.powers[i1]);
                i1++;
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator/(const SymbolicMonomial<I, P> &lhs, const long long int rhs) {
            if (rhs == 0 or lhs.coefficient % rhs != 0) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicMonomial<I, P>(lhs);
            result.coefficient /= rhs;
            return result;
        }

        template<typename I, typename P>
        SymbolicMonomial<I, P> operator/(const long long int lhs, const SymbolicMonomial<I, P> &rhs) {
            if (not rhs.is_constant() or rhs.coefficient == 0 or lhs % rhs.coefficient != 0) {
                throw NonIntegerDivision();
            }
            return SymbolicMonomial<I, P>(lhs / rhs.coefficient);
        }

        template<typename I, typename P>
        class SymbolicPolynomial {
        public:
            /** The monomials vector should be kept sorted according to less_then_comparator at all times */
            std::vector<SymbolicMonomial<I, P>> monomials;

            SymbolicPolynomial() {};

            SymbolicPolynomial(const SymbolicPolynomial<I, P> &polynomial):
                    monomials(polynomial.monomials){};

            SymbolicPolynomial(const SymbolicMonomial<I, P> &monomial):
                    monomials({monomial}) {};

            SymbolicPolynomial(const long long int value) {
                if (value != 0) {
                    monomials.push_back(SymbolicMonomial<I, P>(value));
                }
            }

            static SymbolicPolynomial variable(const I variable) {
                return SymbolicPolynomial(SymbolicMonomial<I, P>::variable(variable));
            }

            static SymbolicPolynomial<I,P> zero;

            static SymbolicPolynomial<I,P> one;

            bool is_constant() const {
                if (monomials.size() > 1) {
                    return false;
                } else if (monomials.size() == 1) {
                    return monomials[0].is_constant();
                } else {
                    return true;
                }
            }

//            void simplify() {
//                std::sort(this->monomials.begin(), this->monomials.end(), less_than_comparator<I, P>);
//                for (int i = 1; i < this->monomials.size(); i++) {
//                    if (up_to_coefficient(this->monomials[i - 1], this->monomials[i])) {
//                        this->monomials[i - 1].coefficient += this->monomials[i].coefficient;
//                        this->monomials.erase(this->monomials.begin() + i);
//                        i--;
//                    }
//                }
//                for (int i = 0; i < this->monomials.size(); i++) {
//                    if (this->monomials[i].coefficient == 0) {
//                        this->monomials.erase(this->monomials.begin() + i);
//                        i--;
//                    }
//                }
//            }
            template <typename T>
            T eval(std::vector<T> &values) {
                T value = 0;
                for(auto i = 0; i < monomials.size(); i++){
                    value += monomials[i].eval<T>(values);
                }
                return value;
            }

            long long int eval() {
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
                if (monomials.size() == 0) {
                    return "0";
                }
                std::string result = monomials[0].to_string();
                for (auto i = 1; i < monomials.size(); i++) {
                    if (monomials[i].coefficient > 0) {
                        result += "+" + monomials[i].to_string();
                    } else {
                        result += monomials[i].to_string();
                    }
                }
                return result;
            }

            std::string to_string_with_star() const {
                if (monomials.size() == 0) {
                    return "0";
                }
                std::string result = monomials[0].to_string_with_star();
                for (auto i = 1; i < monomials.size(); i++) {
                    if (monomials[i].coefficient > 0) {
                        result += "+" + monomials[i].to_string_with_star();
                    } else {
                        result += monomials[i].to_string_with_star();
                    }
                }
                return result;
            }
        };

        template<typename I, typename P>
        SymbolicPolynomial<I,P> SymbolicPolynomial<I,P>::zero = SymbolicPolynomial<I,P>();

        template<typename I, typename P>
        SymbolicPolynomial<I,P> SymbolicPolynomial<I,P>::one = SymbolicPolynomial<I,P>(SymbolicMonomial<I,P>::one);

        template<typename I, typename P>
        std::ostream &operator<<(std::ostream &f, const SymbolicPolynomial<I, P> &polynomial) {
            f << polynomial.to_string();
            return f;
        }

        template<typename I, typename P>
        bool operator==(const SymbolicPolynomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            if (lhs.monomials.size() != rhs.monomials.size()) {
                return false;
            }
            for (auto i = 0; i < lhs.monomials.size(); i++) {
                if (lhs.monomials[i] != rhs.monomials[i]) {
                    return false;
                }
            }
            return true;
        }

        template<typename I, typename P>
        bool operator==(const SymbolicPolynomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
        }

        template<typename I, typename P>
        bool operator==(const SymbolicMonomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            return (rhs.monomials.size() == 0 and lhs == 0) or (rhs.monomials.size() == 1 and rhs.monomials[0] == lhs);
        }

        template<typename I, typename P>
        bool operator==(const SymbolicPolynomial<I, P> &lhs, const long long int rhs) {
            return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
        }

        template<typename I, typename P>
        bool operator==(const long long int lhs, const SymbolicPolynomial<I, P> &rhs) {
            return (rhs.monomials.size() == 0 and lhs == 0) or (rhs.monomials.size() == 1 and rhs.monomials[0] == lhs);
        }

        template<typename I, typename P>
        bool operator!=(const SymbolicPolynomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        bool operator!=(const SymbolicPolynomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        bool operator!=(const SymbolicMonomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        bool operator!=(const SymbolicPolynomial<I, P> &lhs, const long long int rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        bool operator!=(const long long int lhs, const SymbolicPolynomial<I, P> &rhs) {
            return not (lhs == rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const SymbolicPolynomial<I, P> &rhs) {
            return rhs;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const SymbolicPolynomial<I, P> &rhs) {
            SymbolicPolynomial<I, P> result = SymbolicPolynomial<I, P>(rhs);
            for (auto i = 0; i < rhs.monomials.size(); i++) {
                result.monomials[i].coefficient = -result.monomials[i].coefficient;
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const SymbolicMonomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            auto result = SymbolicPolynomial<I, P>();
            if(up_to_coefficient(lhs, rhs)){
                if(lhs.coefficient != - rhs.coefficient) {
                    result.monomials.push_back(SymbolicMonomial<I, P>(lhs));
                    result.monomials[0].coefficient += rhs.coefficient;
                }
            } else if(less_than_comparator(lhs, rhs)){
                result.monomials.push_back(lhs);
                result.monomials.push_back(rhs);
            } else {
                result.monomials.push_back(rhs);
                result.monomials.push_back(lhs);
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const SymbolicMonomial<I, P> &lhs, const long long int rhs) {
            auto result = SymbolicPolynomial<I, P>();
            if(lhs.is_constant()){
                if(lhs.coefficient != -rhs) {
                    result.monomials.push_back(SymbolicMonomial<I, P>(lhs));
                    result.monomials[0].coefficient += rhs;
                }
            } else {
                result.monomials.push_back(lhs);
                result.monomials.push_back(SymbolicMonomial<I,P>(rhs));
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const long long int lhs, const SymbolicMonomial<I, P> &rhs) {
            return rhs + lhs;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const SymbolicPolynomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            auto result = SymbolicPolynomial<I, P>();
            auto i1 = 0;
            auto i2 = 0;
            while(i1 < lhs.monomials.size() and i2 < rhs.monomials.size()){
                if(up_to_coefficient(lhs.monomials[i1], rhs.monomials[i2])){
                    if(lhs.monomials[i1].coefficient != - rhs.monomials[i2].coefficient) {
                        result.monomials.push_back(SymbolicMonomial<I, P>(lhs.monomials[i1]));
                        result.monomials.back().coefficient += rhs.monomials[i2].coefficient;
                    }
                    i1++;
                    i2++;
                } else if(less_than_comparator(lhs.monomials[i1], rhs.monomials[i2])){
                    result.monomials.push_back(lhs.monomials[i1]);
                    i1++;
                } else {
                    result.monomials.push_back(rhs.monomials[i2]);
                    i2++;
                }
            }
            while(i1 < lhs.monomials.size()){
                result.monomials.push_back(lhs.monomials[i1]);
                i1++;
            }
            while(i2 < rhs.monomials.size()){
                result.monomials.push_back(rhs.monomials[i2]);
                i2++;
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const SymbolicPolynomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            return lhs + SymbolicPolynomial<I,P>(rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const SymbolicMonomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            return rhs + lhs;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const SymbolicPolynomial<I, P> &lhs, const long long int rhs) {
            return lhs + SymbolicPolynomial<I,P>(rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator+(const long long int lhs, const SymbolicPolynomial<I, P> &rhs) {
            return rhs + lhs;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const SymbolicMonomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const SymbolicMonomial<I, P> &lhs, const long long int rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const long long int lhs, const SymbolicMonomial<I, P> rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const SymbolicPolynomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const SymbolicPolynomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const SymbolicMonomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const SymbolicPolynomial<I, P> &lhs, const long long int rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator-(const long long int lhs, const SymbolicPolynomial<I, P> &rhs) {
            return lhs + (-rhs);
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator*(const SymbolicPolynomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            auto result = SymbolicPolynomial<I, P>();
            for (auto i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] * rhs);
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator*(const SymbolicPolynomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            auto result = SymbolicPolynomial<I, P>();
            auto partial = SymbolicPolynomial<I, P>();
            for (int i = 0; i < lhs.monomials.size(); i++) {
                partial.monomials.clear();
                for (int j = 0; j < rhs.monomials.size(); j++) {
                    partial.monomials.push_back(lhs.monomials[i] * rhs.monomials[j]);
                }
                result = result + partial;
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator*(const SymbolicMonomial<I, P> lhs, const SymbolicPolynomial<I, P> rhs) {
            return rhs * lhs;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator*(const SymbolicPolynomial<I, P> &lhs, const long long int rhs) {
            auto result = SymbolicPolynomial<I, P>();
            for (int i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] * rhs);
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator*(const long long int lhs, const SymbolicPolynomial<I, P> rhs) {
            return rhs * lhs;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator/(const SymbolicPolynomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            auto result = SymbolicPolynomial<I, P>();
            auto reminder = SymbolicPolynomial<I, P>(lhs);
            SymbolicMonomial<I, P> next_monomial;
            while (not reminder.is_constant()) {
                next_monomial = (reminder.monomials[0] / rhs.monomials[0]);
                result = result + next_monomial;
                auto s = rhs * next_monomial;
                reminder = reminder - s;
            }
            if (reminder != 0) {
                throw NonIntegerDivision();
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator/(const SymbolicPolynomial<I, P> &lhs, const SymbolicMonomial<I, P> &rhs) {
            auto result = SymbolicPolynomial<I, P>();
            for (auto i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] / rhs);
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator/(const SymbolicMonomial<I, P> &lhs, const SymbolicPolynomial<I, P> &rhs) {
            if (rhs.monomials.size() != 1) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicPolynomial<I, P>();
            result.monomials.push_back(lhs / rhs.monomials[0]);
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator/(const SymbolicPolynomial<I, P> &lhs, const long long int rhs) {
            if (rhs == 0) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicPolynomial<I, P>();
            for (auto i = 0; i < lhs.monomials.size(); i++) {
                result.monomials.push_back(lhs.monomials[i] / rhs);
            }
            return result;
        }

        template<typename I, typename P>
        SymbolicPolynomial<I, P> operator/(const long long int lhs, const SymbolicPolynomial<I, P> rhs) {
            if (rhs.monomials.size() != 1) {
                throw NonIntegerDivision();
            }
            auto result = SymbolicPolynomial<I, P>();
            result.monomials.push_back(lhs / rhs.monomials[0]);
            return result;
        }
    }
}
#endif //METADIFF_SYMBOLIC_H
