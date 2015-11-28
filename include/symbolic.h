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

    template<const size_t N, typename T>
    class SymbolicMonomial {
        static_assert(std::numeric_limits<T>::is_integer, "X can be only instantiated with integer types");
    public:
        T powers[N];
        int coefficient;

        SymbolicMonomial() {
            for (int i = 0; i < N; i++) {
                this->powers[i] = 0;
            }
            this->coefficient = 1;
        }

        SymbolicMonomial(const SymbolicMonomial<N,T>& monomial){
            for (int i = 0; i < N; i++) {
                this->powers[i] = monomial.powers[i];
            }
            this->coefficient = monomial.coefficient;
        }

        SymbolicMonomial(const size_t variable): SymbolicMonomial(){
            this->powers[variable] = 1;
        }

        static SymbolicMonomial as_monomial(int value){
            auto result = SymbolicMonomial<N,T>();
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

    template<const size_t N, typename T>
    std::ostream &operator<<(std::ostream &out, const SymbolicMonomial<N,T>& value){
        out << value.coefficient << "(";
        for(int i=0;i<N-1;i++){
            out << value.powers[i] << ",";
        }
        out << value.powers[N-1] << ")";
        return out;
    }

    template<size_t N, typename T>
    bool operator==(const SymbolicMonomial<N, T> &lhs, const SymbolicMonomial<N,T> &rhs){
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

    template<size_t N, typename T>
    bool operator==(const SymbolicMonomial<N,T> &lhs, const int rhs){
        return lhs.is_constant() and lhs.coefficient == rhs;
    }


    template<size_t N, typename T>
    bool operator==(const int lhs, const SymbolicMonomial<N,T> &rhs){
        return (rhs == lhs);
    }

    template<size_t N, typename T>
    bool operator!=(const SymbolicMonomial<N,T> &lhs, const SymbolicMonomial<N,T> &rhs){
        return not(lhs == rhs);
    }

    template<size_t N, typename T>
    bool operator!=(const int lhs, const SymbolicMonomial<N,T> &rhs){
        return not(lhs == rhs);
    }

    template<size_t N, typename T>
    bool operator!=(const SymbolicMonomial<N,T> &lhs, const int rhs){
        return not(lhs == rhs);
    }

    template<size_t N, typename T>
    bool up_to_coefficient(const SymbolicMonomial<N,T> &lhs, const SymbolicMonomial<N,T> &rhs){
        for(int i=0;i<N;i++){
            if(lhs.powers[i] != rhs.powers[i]){
                return false;
            }
        }
        return true;
    }

    template<size_t N, typename T>
    bool up_to_coefficient(const int lhs, const SymbolicMonomial<N,T> &rhs){
        return rhs.is_constant();
    }

    template<size_t N, typename T>
    bool up_to_coefficient(const SymbolicMonomial<N,T> &lhs, const int rhs){
        return lhs.is_constant();
    }

    template<size_t N, typename T>
    bool less_than_comparator(const SymbolicMonomial<N,T>& monomial1, const SymbolicMonomial<N,T>& monomial2){
        for(int i=0;i<N;i++){
            if(monomial1.powers[i] < monomial2.powers[i]){
                return false;
            } else if(monomial1.powers[i] > monomial2.powers[i]){
                return true;
            }
        }
        return false;
    }

    template<const size_t N, typename T>
    SymbolicMonomial<N,T> operator+(const SymbolicMonomial<N,T>& rhs) {
        return rhs;
    }

    template<const size_t N, typename T>
    SymbolicMonomial<N,T> operator-(const SymbolicMonomial<N,T>& rhs) {
        auto result = SymbolicMonomial<N,T>(rhs);
        result.coefficient = -result.coefficient;
        return result;
    }

    template<const size_t N, typename T>
    SymbolicMonomial<N,T> operator*(const SymbolicMonomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        auto result = SymbolicMonomial<N,T>(lhs);
        for (int i = 0; i < N; i++) {
            result.powers[i] += rhs.powers[i];
        }
        result.coefficient *= rhs.coefficient;
        return result;
    }

    template<size_t N, typename T>
    SymbolicMonomial<N,T> operator*(const SymbolicMonomial<N,T>& lhs, const int rhs) {
        auto result = SymbolicMonomial<N,T>(lhs);
        result.coefficient *= rhs;
        return result;
    }

    template<size_t N, typename T>
    SymbolicMonomial<N,T> operator*(const int lhs, const SymbolicMonomial<N,T>& rhs) {
        return rhs*lhs;
    }

    template<size_t N, typename T>
    SymbolicMonomial<N,T> operator/(const SymbolicMonomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        auto result = SymbolicMonomial<N,T>(lhs);
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

    template<size_t N, typename T>
    SymbolicMonomial<N,T> operator/(const SymbolicMonomial<N,T>& lhs, int rhs) {
        if(rhs == 0 or lhs.coefficient % rhs != 0){
            throw impossible_division();
        }
        auto result = SymbolicMonomial<N,T>(lhs);
        result.coefficient /= rhs;
        return result;
    }

    template<size_t N, typename T>
    SymbolicMonomial<N,T> operator/(int lhs, const SymbolicMonomial<N,T>& rhs) {
        if(not rhs.is_constant() or rhs.coefficient == 0 or lhs % rhs.coefficient != 0){
            throw impossible_division();
        }
        return as_monomial(lhs / rhs.coefficient);
    }


    template<size_t N, typename T>
    class SymbolicPolynomial{
    public:
        std::vector<SymbolicMonomial<N,T>> monomials;

        SymbolicPolynomial(){};

        SymbolicPolynomial(const size_t variable){
            monomials.push_back(SymbolicMonomial<N,T>(variable));
        }

        SymbolicPolynomial(const SymbolicPolynomial<N,T>& polynomial){
            this->monomials = polynomial.monomials;
        }

        static SymbolicPolynomial as_polynomial(const int value){
            auto result = SymbolicPolynomial<N,T>();
            result.monomials.push_back(SymbolicMonomial<N,T>::as_monomial(value));
            return result;
        }

        static SymbolicPolynomial as_polynomial(const SymbolicMonomial<N,T>& monomial){
            auto result = SymbolicPolynomial<N,T>();
            result.monomials.push_back(SymbolicMonomial<N,T>(monomial));
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
            std::sort(this->monomials.begin(), this->monomials.end(), less_than_comparator<N,T>);
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

    template<const size_t N, typename T>
    std::ostream &operator<<(std::ostream &out, const SymbolicPolynomial<N,T>& polynomial){
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

    template<const size_t N, typename T>
    bool operator==(const SymbolicPolynomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
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

    template<const size_t N, typename T>
    bool operator==(const SymbolicPolynomial<N,T>& lhs, const SymbolicMonomial<N, T>& rhs) {
        return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
    }

    template<const size_t N, typename T>
    bool operator==(const SymbolicMonomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        return rhs == lhs;
    }

    template<const size_t N, typename T>
    bool operator==(const SymbolicPolynomial<N,T>& lhs, const int rhs) {
        return (lhs.monomials.size() == 0 and rhs == 0) or (lhs.monomials.size() == 1 and lhs.monomials[0] == rhs);
    }

    template<const size_t N, typename T>
    bool operator==(const int lhs, const SymbolicPolynomial<N,T>& rhs) {
        return rhs == lhs;
    }

    template<const size_t N, typename T>
    bool operator!=(const SymbolicPolynomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N, typename T>
    bool operator!=(const SymbolicPolynomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N, typename T>
    bool operator!=(const SymbolicMonomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N, typename T>
    bool operator!=(const SymbolicPolynomial<N, T>& lhs, const int rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N, typename T>
    bool operator!=(const int lhs, const SymbolicPolynomial<N, T>& rhs) {
        return not(lhs == rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const SymbolicPolynomial<N,T>& rhs) {
        return rhs;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const SymbolicPolynomial<N,T>& rhs) {
        SymbolicPolynomial<N,T> result = SymbolicPolynomial<N,T>(rhs);
        for(int i=0;i<rhs.monomials.size();i++){
            result.monomials[i].coefficient = - result.monomials[i].coefficient;
        }
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const SymbolicMonomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        auto result = SymbolicPolynomial<N,T>();
        result.monomials.push_back(lhs);
        result.monomials.push_back(rhs);
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const SymbolicMonomial<N,T>& lhs, const int rhs) {
        auto result = SymbolicPolynomial<N,T>();
        result.monomials.push_back(lhs);
        result.monomials.push_back(SymbolicMonomial<N,T>::as_monomial(rhs));
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const int lhs, const SymbolicMonomial<N,T>& rhs) {
        return rhs + lhs;
    }



    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const SymbolicPolynomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        auto result = SymbolicPolynomial<N,T>(lhs);
        result.monomials.insert(result.monomials.end(), rhs.monomials.begin(), rhs.monomials.end());
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const SymbolicPolynomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        auto result = SymbolicPolynomial<N,T>(lhs);
        result.monomials.push_back(rhs);
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const SymbolicMonomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        return rhs + lhs;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const SymbolicPolynomial<N,T>& lhs, const int rhs) {
        auto result = SymbolicPolynomial<N,T>(lhs);
        result.monomials.push_back(SymbolicMonomial<N,T>::as_monomial(rhs));
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator+(const int lhs, const SymbolicPolynomial<N,T>& rhs) {
        return rhs + lhs;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const SymbolicMonomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const SymbolicMonomial<N,T>& lhs, const int rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const int lhs, const SymbolicMonomial<N,T> rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const SymbolicPolynomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const SymbolicPolynomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const SymbolicMonomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const SymbolicPolynomial<N,T>& lhs, const int rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator-(const int lhs, const SymbolicPolynomial<N,T>& rhs) {
        return lhs + (-rhs);
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator*(const SymbolicPolynomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        auto result = SymbolicPolynomial<N,T>();
        for(int i=0;i<lhs.monomials.size();i++){
            for(int j=0;j<rhs.monomials.size();j++){
                result.monomials.push_back(lhs.monomials[i] * rhs.monomials[j]);
            }
        }
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator*(const SymbolicPolynomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        auto result = SymbolicPolynomial<N,T>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] * rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator*(const SymbolicMonomial<N,T> lhs, const SymbolicPolynomial<N,T> rhs) {
        return rhs * lhs;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator*(const SymbolicPolynomial<N,T>& lhs, int rhs) {
        auto result = SymbolicPolynomial<N,T>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] * rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator*(int lhs, const SymbolicPolynomial<N,T> rhs) {
        return rhs * lhs;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator/(const SymbolicPolynomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        auto result = SymbolicPolynomial<N,T>();
        auto reminder = SymbolicPolynomial<N,T>(lhs);
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

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator/(const SymbolicPolynomial<N,T>& lhs, const SymbolicMonomial<N,T>& rhs) {
        auto result = SymbolicPolynomial<N,T>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] / rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator/(const SymbolicMonomial<N,T>& lhs, const SymbolicPolynomial<N,T>& rhs) {
        if(rhs.monomials.size() != 1){
            throw impossible_division();
        }
        auto result = SymbolicPolynomial<N,T>();
        result.monomials.push_back(lhs / rhs.monomials[0]);
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator/(const SymbolicPolynomial<N,T>& lhs, const int rhs) {
        if(rhs == 0){
            throw impossible_division();
        }
        auto result = SymbolicPolynomial<N,T>();
        for(int i=0;i<lhs.monomials.size();i++){
            result.monomials.push_back(lhs.monomials[i] / rhs);
        }
        result.simplify();
        return result;
    }

    template<const size_t N, typename T>
    SymbolicPolynomial<N,T> operator/(const int lhs, const SymbolicPolynomial<N,T> rhs) {
        if(rhs.monomials.size() != 1){
            throw impossible_division();
        }
        auto result = SymbolicPolynomial<N,T>();
        result.monomials.push_back(lhs / rhs.monomials[0]);
        return result;
    }

}

#endif //AUTODIFF_SYMBOLIC_H
