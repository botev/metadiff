//
// Created by alex on 17/01/16.
//

#include "regex"
#include "iostream"

int main(int argc, char **argv){
    std::regex ids("\\s[[:digit:]]+(-|\\|)");
    std::string s = "Grad msg 52 -> 48|Grad of 48|Grad msg 48 -> 46|Grad msg 48 -> 47|Grad of 47|Grad of 46|";
    for(std::sregex_iterator i = std::sregex_iterator(s.begin(), s.end(), ids);
        i != std::sregex_iterator();
        ++i )
    {
        std::smatch m = *i;
        std::cout << m.str() << " at position " << m.position() << '\n';
    }
}