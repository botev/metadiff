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


namespace symbolic {

    class impossible_division: public std::exception
    {
        virtual const char* what() const throw()
        {
            return "The requested division leads to a non integer monomial/polynomial";
        }
    };

    template<const size_t N>
    class SymbolicMonomial {
    public:
        int powers[N];
        int coefficient;

        SymbolicMonomial() {
            for (int i = 0; i < N; i++) {
                this->powers[i] = 0;
            }
            this->coefficient = 1;
        }

        SymbolicMonomial(const SymbolicMonomial& monomial){
            for (int i = 0; i < N; i++) {
                this->powers[i] = monomial.powers[i];
            }
            this->coefficient = monomial.coefficient;
        }

        SymbolicMonomial(const int variable): SymbolicMonomial(){
            this->powers[variable] = 1;
        }

        static SymbolicMonomial as_monomial(int value){
            auto result = SymbolicMonomial<N>();
            result.coefficient = value;
            return result;
        }

        bool is_constant() const{
            for(int i=0;i<N;i++){
                if(this->powers[i] > 0){
                    return false;
                }
            }
            return true;
        }
    };

    template<const size_t N>
    std::ostream &operator<<(std::ostream &out, const SymbolicMonomial<N>& value){
        out << value.coefficient << "(";
        for(int i=0;i<N-1;i++){
            out << value.powers[i] << ",";
        }
        out << value.powers[N-1] << ")";
        return out;
    }

    template<size_t N>
    bool operator==(const SymbolicMonomial<N> &lhs, const SymbolicMonomial<N> &rhs){
        if(lhs.coefficient != rhs.coefficient){
            return false;
        }
        for(int i=0;i<N;i++){
            if(lhs.powers[i] != rhs.powers[i]){
                return false;
            }
        }
        return true;
    }

    template<size_t N>
    bool operator==(const SymbolicMonomial<N> &lhs, const int rhs){
        return lhs.is_constant() and lhs.coefficient == rhs;
    }


    template<size_t N>
    bool operator==(const int lhs, const SymbolicMonomial<N> &rhs){
        return (rhs == lhs);
    }

    template<size_t N>
    bool operator!=(const SymbolicMonomial<N> &lhs, const SymbolicMonomial<N> &rhs){
        return not(lhs == rhs);
    }

    template<size_t N>
    bool operator!=(const int lhs, const SymbolicMonomial<N> &rhs){
        return not(lhs == rhs);
    }

    template<size_t N>
    bool operator!=(const SymbolicMonomial<N> &lhs, const int rhs){
        return not(lhs == rhs);
    }

    template<size_t N>
    bool up_to_coefficient(const SymbolicMonomial<N> &lhs, const SymbolicMonomial<N> &rhs){
        for(int i=0;i<N;i++){
            if(lhs.powers[i] != rhs.powers[i]){
                return false;
            }
        }
        return true;
    }

    template<size_t N>
    bool up_to_coefficient(const int lhs, const SymbolicMonomial<N> &rhs){
        return rhs.is_constant();
    }

    template<size_t N>
    bool up_to_coefficient(const SymbolicMonomial<N> &lhs, const int rhs){
        return lhs.is_constant();
    }

    template<size_t N>
    bool less_than_comparator(const SymbolicMonomial<N>& monomial1, const SymbolicMonomial<N>& monomial2){
        for(int i=0;i<N;i++){
            if(monomial1.powers[i] < monomial2.powers[i]){
                return false;
            } else if(monomial1.powers[i] > monomial2.powers[i]){
                return true;
            }
        }
        return false;
    }

    template<const size_t N>
    SymbolicMonomial<N> operator+(const SymbolicMonomial<N>& rhs) {
        return rhs;
    }

    template<const size_t N>
    SymbolicMonomial<N> operator-(const SymbolicMonomial<N>& rhs) {
        SymbolicMonomial<N> result = SymbolicMonomial<N>(rhs);
        result.coefficient = -result.coefficient;
        return result;
    }

    template<const size_t N>
    SymbolicMonomial<N> operator*(const SymbolicMonomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        SymbolicMonomial<N> result = SymbolicMonomial<N>(lhs);
        for (int i = 0; i < N; i++) {
            result.powers[i] += rhs.powers[i];
        }
        result.coefficient *= rhs.coefficient;
        return result;
    }

    template<size_t N>
    SymbolicMonomial<N> operator*(const SymbolicMonomial<N>& lhs, const int rhs) {
        auto result = SymbolicMonomial<N>(lhs);
        result.coefficient *= rhs;
        return result;
    }

    template<size_t N>
    SymbolicMonomial<N> operator*(const int lhs, const SymbolicMonomial<N>& rhs) {
        return rhs*lhs;
    }

    template<size_t N>
    SymbolicMonomial<N> operator/(const SymbolicMonomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        auto result = SymbolicMonomial<N>(lhs);
        for (int i = 0; i < N; i++) {
            result.powers[i] -= rhs.powers[i];
            if(result.powers[i] > lhs.powers[i]){
                throw impossible_division();
            }
        }
        if(rhs.coefficient == 0 or result.coefficient % rhs.coefficient != 0){
            throw impossible_division();
        }
        result.coefficient = lhs.coefficient / rhs.coefficient;
        return result;
    }

    template<size_t N>
    SymbolicMonomial<N> operator/(const SymbolicMonomial<N>& lhs, int rhs) {
        if(rhs == 0 or lhs.coefficient % rhs != 0){
            throw impossible_division();
        }
        auto result = SymbolicMonomial<N>(lhs);
        result.coefficient /= rhs;
        return result;
    }

    template<size_t N>
    SymbolicMonomial<N> operator/(int lhs, const SymbolicMonomial<N>& rhs) {
        if(not rhs.is_constant() or rhs.coefficient == 0 or lhs % rhs.coefficient != 0){
            throw impossible_division();
        }
        return as_monomial(lhs / rhs.coefficient);
    }


    template<size_t N>
    class SymbolicPolynomial{
    public:
        std::vector<SymbolicMonomial<N>> monomials;

        SymbolicPolynomial(){};

        SymbolicPolynomial(int variable){
            monomials.push_back(SymbolicMonomial<N>(variable));
        }

        SymbolicPolynomial(const SymbolicPolynomial& polynomial){
            this->monomials = polynomial.monomials;
        }

        static SymbolicPolynomial as_polynomial(const int value){
            auto result = SymbolicPolynomial<N>();
            result.monomials.push_back(SymbolicMonomial<N>::as_monomial(value));
            return result;
        }

        static SymbolicPolynomial as_polynomial(const SymbolicMonomial<N>& monomial){
            auto result = SymbolicPolynomial();
            result.monomials.push_back(SymbolicMonomial<N>(monomial));
            return result;
        }

        bool is_constant() const{
            if(this->monomials.size() > 1){
                return false;
            } else if(this->monomials.size() == 1){
                return this->monomials[0].is_constant();
            } else {
                return true;
            }
        }

        void simplify() {
            std::sort(this->monomials.begin(), this->monomials.end(), less_than_comparator<N>);
            for(int i=1;i<this->monomials.size();i++){
                if(up_to_coefficient(this->monomials[i-1],this->monomials[i])){
                    this->monomials[i-1].coefficient += this->monomials[i].coefficient;
                    this->monomials.erase(this->monomials.begin()+i);
                    i--;
                }
            }
            for(int i=0;i<this->monomials.size();i++){
                if(this->monomials[i].coefficient == 0){
                    this->monomials.erase(this->monomials.begin()+i);
                    i--;
                }
            }
        }
    };

    template<const size_t N>
    std::ostream &operator<<(std::ostream &out, const SymbolicPolynomial<N>& polynomial){
        if(polynomial.monomials.size() == 0){
            out << "0" << std::endl;
        } else {
            for(int i=0;i<polynomial.monomials.size();i++){
                if(i < polynomial.monomials.size()-1) {
                    out << polynomial.monomials[i] << "|";
                } else {
                    out << polynomial.monomials[i];
                }
            }
        }
        return out;
    }

    template<const size_t N>
    bool operator==(const SymbolicPolynomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        if(lhs.monomials.size() != rhs.monomials.size()){
            return false;
        }
        for(int i=0;i<lhs.monomials.size();i++){
            if(lhs.monomials[i] !=rhs.monomials[i]){
                return false;
            }
        }
        return true;
    }

    template<const size_t N>
    bool operator==(const SymbolicPolynomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
    }

    template<const size_t N>
    bool operator==(const SymbolicMonomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        return rhs == lhs;
    }

    template<const size_t N>
    bool operator==(const SymbolicPolynomial<N>& lhs, const int rhs) {
        return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
    }

    template<const size_t N>
    bool operator==(const int lhs, const SymbolicPolynomial<N>& rhs) {
        return rhs == lhs;
    }

    template<const size_t N>
    bool operator!=(const SymbolicPolynomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N>
    bool operator!=(const SymbolicPolynomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N>
    bool operator!=(const SymbolicMonomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N>
    bool operator!=(const SymbolicPolynomial<N>& lhs, const int rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N>
    bool operator!=(const int lhs, const SymbolicPolynomial<N>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const SymbolicPolynomial<N>& rhs) {
        return rhs;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const SymbolicPolynomial<N>& rhs) {
        SymbolicPolynomial<N> result = SymbolicPolynomial<N>(rhs);
        for(int i=0;i<rhs.monomials.size();i++){
            result.monomials[i].coefficient = - result.monomials[i].coefficient;
        }
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const SymbolicMonomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        auto result = SymbolicPolynomial<N>();
        result.monomials.push_back(lhs);
        result.monomials.push_back(rhs);
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const SymbolicMonomial<N>& lhs, const int rhs) {
        auto result = SymbolicPolynomial<N>();
        result.monomials.push_back(lhs);
        result.monomials.push_back(SymbolicMonomial<N>::as_monomial(rhs));
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const int lhs, const SymbolicMonomial<N>& rhs) {
        return rhs + lhs;
    }



    template<const size_t N>
    SymbolicPolynomial<N> operator+(const SymbolicPolynomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        auto result = SymbolicPolynomial<N>(lhs);
        result.monomials.insert(result.monomials.end(), rhs.monomials.begin(), rhs.monomials.end());
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const SymbolicPolynomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        auto result = SymbolicPolynomial<N>(lhs);
        result.monomials.push_back(rhs);
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const SymbolicMonomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        return rhs + lhs;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const SymbolicPolynomial<N>& lhs, const int rhs) {
        auto result = SymbolicPolynomial<N>(lhs);
        result.monomials.push_back(SymbolicMonomial<N>::as_monomial(rhs));
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator+(const int lhs, const SymbolicPolynomial<N>& rhs) {
        return rhs + lhs;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const SymbolicMonomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const SymbolicMonomial<N>& lhs, const int rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const int lhs, const SymbolicMonomial<N> rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const SymbolicPolynomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const SymbolicPolynomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const SymbolicMonomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const SymbolicPolynomial<N>& lhs, const int rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator-(const int lhs, const SymbolicPolynomial<N>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator*(const SymbolicPolynomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        auto result = SymbolicPolynomial<N>();
        for(int i=0;i<lhs.monomials.size();i++){
            for(int j=0;j<rhs.monomials.size();j++){
                result.monomials.push_back(lhs.monomials[i] * rhs.monomials[j]);
            }
        }
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator*(const SymbolicPolynomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        auto result = SymbolicPolynomial<N>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] * rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator*(const SymbolicMonomial<N> lhs, const SymbolicPolynomial<N> rhs) {
        return rhs * lhs;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator*(const SymbolicPolynomial<N>& lhs, int rhs) {
        auto result = SymbolicPolynomial<N>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] * rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator*(int lhs, const SymbolicPolynomial<N> rhs) {
        return rhs * lhs;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator/(const SymbolicPolynomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        auto result = SymbolicPolynomial<N>();
        auto reminder = SymbolicPolynomial<N>(lhs);
        while(not reminder.is_constant()){
//            std::cout<< "R: " << reminder << std::endl;
            result.monomials.push_back(reminder.monomials[0] / rhs.monomials[0]);
//            std::cout<< "M: " << result.monomials.back() << std::endl;
            auto s = rhs * result.monomials.back();
//            std::cout<< "S: " << s << std::endl;
            reminder =  reminder - s;
//            std::cout<< "R2: " << reminder << std::endl;
//            std::cout << (reminder == 0) << std::endl;
        }
        if(reminder != 0){
            throw impossible_division();
        }
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator/(const SymbolicPolynomial<N>& lhs, const SymbolicMonomial<N>& rhs) {
        auto result = SymbolicPolynomial<N>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] / rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator/(const SymbolicMonomial<N>& lhs, const SymbolicPolynomial<N>& rhs) {
        if(rhs.monomials.size() != 1){
            throw impossible_division();
        }
        auto result = SymbolicPolynomial<N>();
        result.monomials.push_back(lhs / rhs.monomials[0]);
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator/(const SymbolicPolynomial<N>& lhs, const int rhs) {
        if(rhs == 0){
            throw impossible_division();
        }
        auto result = SymbolicPolynomial<N>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] / rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N>
    SymbolicPolynomial<N> operator/(const int lhs, const SymbolicPolynomial<N> rhs) {
        if(rhs.monomials.size() != 1){
            throw impossible_division();
        }
        auto result = SymbolicPolynomial<N>();
        result.monomials.push_back(lhs / rhs.monomials[0]);
        return result;
    }

}

#endif //AUTODIFF_SYMBOLIC_H
