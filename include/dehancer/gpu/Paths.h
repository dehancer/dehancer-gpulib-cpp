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

    /**
     * CAN Be defined in certain plugin
     * @return string
     */
     extern std::string get_opencl_cache_path();

     /**
      * CAN Be defined in certain plugin
      * @param name file name (in opencl_cache_path)
      * @param size size in bytes
      * @param lib_binary precompiled lib bynary pointer
      * @return true if success
      */
     extern bool get_opencl_lib_binary(const std::string &name, const unsigned char **lib_binary, const size_t **size);
}
