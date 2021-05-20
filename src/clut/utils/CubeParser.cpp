//
// Created by denn nevera on 08/06/2020.
//

#include "dehancer/gpu/clut/utils/CubeParser.h"
#include "dehancer/Common.h"
#include <string>
#include <iostream>
#include <sstream>

namespace dehancer {

    static inline std::string right_trim(std::string str)
    {
        const auto it = std::find_if(str.rbegin(), str.rend(), [](char ch) { return !std::isspace(ch); });
        str.erase(it.base(), str.end());
        return str;
    }


    static inline std::string left_trim(std::string str)
    {
        const auto it = std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace(ch); });
        str.erase(str.begin(), it);
        return str;
    }

    static inline std::string trim(const std::string& str)
    {
        return left_trim(right_trim(str));
    }

    static inline bool nextline(std::istream &istream, std::string &line)
    {
        while ( istream.good() )
        {
            std::getline(istream, line);
            if(!line.empty() && line[line.size() - 1] == '\r')
            {
                line.resize(line.size() - 1);
            }

            line = trim(line);

            if(!line.empty())
            {
                return true;
            }
        }

        line = "";
        return false;
    }

    static inline std::string to_lower(std::string str)
    {
        std::transform(
                str.begin(),
                str.end(),
                str.begin(),
                [](unsigned char c){ return std::tolower(c); });
        return str;
    }

    static inline bool starts_with(const std::string & str, const std::string & prefix)
    {
        return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
    }

    static inline std::vector<std::string> split(const std::string & str)
    {
        std::stringstream stream(str);
        return std::vector<std::string>(std::istream_iterator<std::string>(stream),
                                        std::istream_iterator<std::string>());
    }

    static inline bool to_int(size_t& ival, const std::string& str)
    {
        if(str.empty()) return false;
        std::istringstream i(str);
        bool ret = !(i >> ival);
        return !ret;
    }

     static inline bool to_float(float& fval, const std::string& str)
    {
        if(str.empty()) return false;

        std::istringstream input_stringstream(str);
        float x = 0;

        if(!(input_stringstream >> x)) return false;

        fval = x;
        return true;
    }

    std::istream &operator>>(std::istream &is, CubeParser &dt) {

        dt.buffer_.clear();

        int line_number = 0;
        std::string line;

        while(nextline(is, line))
        {
            if(starts_with(line,"#")) continue;

            auto words = split(line);

            if (words.empty()) continue;

            if(to_lower(words[0]) == "title")
            {
                // skip
            }
            else if(to_lower(words[0]) == "lut_1d_size")
            {
                throw Error(dehancer::CommonError::NOT_SUPPORTED, "1D Cube format is not supported");
            }
            else if(to_lower(words[0]) == "lut_2d_size")
            {
                throw Error(dehancer::CommonError::NOT_SUPPORTED, "2D Cube format is not supported");
            }

            else if(to_lower(words[0]) == "lut_3d_size")
            {
                if (words.size()!=2 || !to_int(dt.lut_size_, words[1])) {
                    throw Error(
                            dehancer::CommonError::PARSE_ERROR,
                            error_string("3D lut size is not correct at line: %i", line_number));
                }

                dt.buffer_.reserve(dt.lut_size_*dt.lut_size_*dt.lut_size_* dt.get_channels());
            }

            else if(to_lower(words[0]) == "domain_min")
            {
                if(words.size() != 4 ||
                   !to_float( dt.domain_min_[0], words[1]) ||
                   !to_float( dt.domain_min_[1], words[2]) ||
                   !to_float( dt.domain_min_[2], words[3]))
                {
                    throw Error(
                            dehancer::CommonError::PARSE_ERROR,
                            error_string("3D lut domain min is not correct at line: %i", line_number));
                }
            }

            else if(to_lower(words[0]) == "domain_max")
            {
                if(words.size() != 4 ||
                   !to_float( dt.domain_max_[0], words[1]) ||
                   !to_float( dt.domain_max_[1], words[2]) ||
                   !to_float( dt.domain_max_[2], words[3]))
                {
                    throw Error(
                            dehancer::CommonError::PARSE_ERROR,
                            error_string("3D lut domain max is not correct at line: %i", line_number));
                }
            }

            else {
                // data
                float data[] = {0,0,0};
                if(words.size() != 3 ||
                   !to_float( data[0], words[0]) ||
                   !to_float( data[1], words[1]) ||
                   !to_float( data[2], words[2]))
                {
                    throw Error(
                            dehancer::CommonError::PARSE_ERROR,
                            error_string("3D lut data is not correct at line: %i", line_number));
                }

                for (int i=0;i<3;++i) dt.buffer_.push_back(data[i]);
                dt.buffer_.push_back(1.0f);
            }

            ++line_number;
        }

        if (dt.lut_size_==0)
            throw Error(
                    dehancer::CommonError::PARSE_ERROR,
                    error_string("3D lut has 0 size..."));

        return is;
    }
}