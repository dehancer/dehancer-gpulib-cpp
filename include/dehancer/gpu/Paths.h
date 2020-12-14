//
// Created by denn nevera on 16/11/2020.
//

#pragma once

#include <string>

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    extern std::string get_lib_path();

    /**
     * CAN BE defined in certain plugin module
     * @param source user defined library source of kernels
     * @return source hash
     */
    extern std::size_t get_lib_source(std::string& source);

    /**
     * Must be defined in certain plugin
     * @return string
     */
    extern std::string get_installation_path();
}
