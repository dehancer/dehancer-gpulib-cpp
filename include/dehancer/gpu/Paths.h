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
     * Must be defined in certain plugin
     * @return string
     */
    extern std::string get_installation_path();
}
