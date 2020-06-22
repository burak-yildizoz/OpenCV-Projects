#ifndef GENERAL_HPP
#define GENERAL_HPP

#include <opencv2/core.hpp>

#include <iostream>
#include <csignal>
#include <map>

#define DEBUG(cc) \
if(true) \
{ \
    std::cout << #cc << " = " << std::endl; \
    std::cout << cc << std::endl; \
}

#define CHECK(cc) \
if(!(cc)) \
{ \
    std::cerr << "Error! " << #cc << std::endl; \
    std::cerr << "Line: " << __LINE__ << std::endl; \
    std::cerr << "In file: " << __FILE__ << std::endl; \
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); \
    std::cin.get(); \
    raise(SIGTERM); \
}

// print vector container elements
template <class T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
    for (unsigned i = 0; i<v.size(); i++)
    {
        os << v[i];
        if (i != v.size() - 1)
            os << " ";
    }
    return os;
}

// print matrix container elements
template <class T>
std::ostream& operator << (std::ostream& os, const std::vector< std::vector<T> >& M)
{
    for (unsigned i = 0; i<M.size(); i++)
        os << M[i] << "\n";
    return os;
}

// vector addition
template <class T>
std::vector<T> operator + (const std::vector<T>& v, const T& x)
{
    std::vector<T> res(v);
    for (size_t i = 0; i < v.size(); i++)
        res[i] += x;
    return res;
}

// find the most frequent element in any container
// {mostFrequentElement, maxFrequency}
template <typename T>
std::pair<typename T::value_type, int> most_frequent_element(T const& v)
{
    // Precondition: v is not empty
    std::map<typename T::value_type, int> frequencyMap;
    int maxFrequency = 0;
    typename T::value_type mostFrequentElement{};
    for (auto&& x : v)
    {
        int f = ++frequencyMap[x];
        if (f > maxFrequency)
        {
            maxFrequency = f;
            mostFrequentElement = x;
        }
    }
    return std::make_pair(mostFrequentElement, maxFrequency);
}

#endif	// GENERAL_HPP
