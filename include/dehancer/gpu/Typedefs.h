//
// Created by denn nevera on 12/11/2020.
//

#ifndef DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H
#define DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H

#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include "aoBenchKernel.h"

using namespace metal;

#elif CL_VERSION_1_2

#else

#include "dehancer/gpu/kernels/types.h"
#include "dehancer/math.hpp"

namespace dehancer {
    typedef dehancer::math::float2 float2;
    typedef dehancer::math::float3 float3;
    typedef dehancer::math::float4 float4;
    typedef dehancer::math::float2x2 float2x2;
    typedef dehancer::math::float3x3 float3x3;
    typedef dehancer::math::float4x4 float4x4;

    //namespace arma {
    typedef arma::Col<uint> uint_vec;
    typedef arma::Col<int> int_vec;
    typedef arma::Col<uint> bool_vec;
    //}
    
    namespace math {
    
        template<size_t N>
        class uint_vector: public uint_vec::fixed<N> {
            using armN = uint_vec::fixed<N>;
        public:
            using armN::armN;
            uint_vector& operator=(const observable::Aproxy<uint_vector> &a) {
              (*this = a.get_data());
              return *this;
            }
        };
    
        template<size_t N>
        class int_vector: public int_vec::fixed<N> {
            using armN = int_vec::fixed<N>;
        public:
            using armN::armN;
            int_vector& operator=(const observable::Aproxy<int_vector> &a) {
              (*this = a.get_data());
              return *this;
            }
        };
    
        template<size_t N>
        class bool_vector: public bool_vec::fixed<N> {
            using armN = bool_vec::fixed<N>;
        public:
            using armN::armN;
            bool_vector& operator=(const observable::Aproxy<bool_vector> &a) {
              (*this = a.get_data());
              return *this;
            }
        };
    
        
        /***
         * uintN
         */
        
        class uint2: public uint_vector<2> {
        public:
            using uint_vector::uint_vector;
        
            uint& x() { return (*this)[0]; };
            [[nodiscard]] const uint& x() const { return (*this)[0]; };
    
            uint& y() { return (*this)[1]; };
            [[nodiscard]] const uint& y() const { return (*this)[1]; };
    
            explicit uint2(const observable::Aproxy<uint2> &a):uint2(a.get_data()){}
        };
    
        class uint3: public uint_vector<3> {
        public:
            using uint_vector::uint_vector;
        
            uint& x() { return (*this)[0]; };
            [[nodiscard]] const uint& x() const { return (*this)[0]; };
        
            uint& y() { return (*this)[1]; };
            [[nodiscard]] const uint& y() const { return (*this)[1]; };
    
            uint& z() { return (*this)[2]; };
            [[nodiscard]] const uint& z() const { return (*this)[2]; };
    
            explicit uint3(const observable::Aproxy<uint3> &a):uint3(a.get_data()){}
        };
    
        class uint4: public uint_vector<4> {
        public:
            using uint_vector::uint_vector;
        
            uint& x() { return (*this)[0]; };
            [[nodiscard]] const uint& x() const { return (*this)[0]; };
        
            uint& y() { return (*this)[1]; };
            [[nodiscard]] const uint& y() const { return (*this)[1]; };
    
            uint& z() { return (*this)[2]; };
            [[nodiscard]] const uint& z() const { return (*this)[2]; };
    
            uint& w() { return (*this)[3]; };
            [[nodiscard]] const uint& w() const { return (*this)[3]; };
    
            explicit uint4(const observable::Aproxy<uint4> &a):uint4(a.get_data()){}
        };
    
        /***
         * intN
         */
    
        class int2: public int_vector<2> {
        public:
            using int_vector::int_vector;
    
            int& x() { return (*this)[0]; };
            [[nodiscard]] const int& x() const { return (*this)[0]; };
    
            int& y() { return (*this)[1]; };
            [[nodiscard]] const int& y() const { return (*this)[1]; };
        
            explicit int2(const observable::Aproxy<int2> &a):int2(a.get_data()){}
        };
    
        class int3: public int_vector<3> {
        public:
            using int_vector::int_vector;
    
            int& x() { return (*this)[0]; };
            [[nodiscard]] const int& x() const { return (*this)[0]; };
    
            int& y() { return (*this)[1]; };
            [[nodiscard]] const int& y() const { return (*this)[1]; };
    
            int& z() { return (*this)[2]; };
            [[nodiscard]] const int& z() const { return (*this)[2]; };
        
            explicit int3(const observable::Aproxy<int3> &a):int3(a.get_data()){}
        };
    
        class int4: public int_vector<4> {
        public:
            using int_vector::int_vector;
    
            int& x() { return (*this)[0]; };
            [[nodiscard]] const int& x() const { return (*this)[0]; };
    
            int& y() { return (*this)[1]; };
            [[nodiscard]] const int& y() const { return (*this)[1]; };
    
            int& z() { return (*this)[2]; };
            [[nodiscard]] const int& z() const { return (*this)[2]; };
    
            int& w() { return (*this)[3]; };
            [[nodiscard]] const int& w() const { return (*this)[3]; };
        
            explicit int4(const observable::Aproxy<int4> &a):int4(a.get_data()){}
        };
    
        /***
         * boolN
         */
    
        class bool2: public bool_vector<2> {
        public:
            using bool_vector::bool_vector;
        
            uint& x() { return (*this)[0]; };
            [[nodiscard]] bool x() const { return (*this)[0]; };
    
            uint& y() { return (*this)[1]; };
            [[nodiscard]] bool y() const { return (*this)[1]; };
        
            explicit bool2(const observable::Aproxy<bool2> &a):bool2(a.get_data()){}
        };
    
        class bool3: public bool_vector<3> {
        public:
            using bool_vector::bool_vector;
    
            uint& x() { return (*this)[0]; };
            [[nodiscard]] bool x() const { return (*this)[0]; };
    
            uint& y() { return (*this)[1]; };
            [[nodiscard]] bool y() const { return (*this)[1]; };
    
            uint& z() { return (*this)[2]; };
            [[nodiscard]] bool z() const { return (*this)[2]; };
        
            explicit bool3(const observable::Aproxy<bool3> &a):bool3(a.get_data()){}
        };
    
        class bool4: public bool_vector<4> {
        public:
            using bool_vector::bool_vector;
    
            uint& x() { return (*this)[0]; };
            [[nodiscard]] bool x() const { return (*this)[0]; };
    
            uint& y() { return (*this)[1]; };
            [[nodiscard]] bool y() const { return (*this)[1]; };
    
            uint& z() { return (*this)[2]; };
            [[nodiscard]] bool z() const { return (*this)[2]; };
    
            uint& w() { return (*this)[3]; };
            [[nodiscard]] bool w() const { return (*this)[3]; };
        
            explicit bool4(const observable::Aproxy<bool4> &a):bool4(a.get_data()){}
        };

//        struct uint2 {
//            [[nodiscard]] inline const uint& x() const { return x_; }
//            [[nodiscard]] inline const uint& y() const { return y_; }
//            inline uint& x() { return x_; }
//            inline uint& y() { return y_; }
//        private:
//            uint x_,y_;
//        };
////
//        struct uint3 {
//            [[nodiscard]] inline const uint& x() const { return x_; }
//            [[nodiscard]] inline const uint& y() const { return y_; }
//            [[nodiscard]] inline const uint& z() const { return z_; }
//            inline uint& x() { return x_; }
//            inline uint& y() { return y_; }
//            inline uint& z() { return z_; }
//        private:
//            uint x_,y_,z_;
//        };
//        struct uint4 {
//        private:
//            uint x,y,z,w;
//        };
//
//        struct int2 {
//        private:
//            int x,y;
//        };
//        struct int3 {
//        private:
//            int x,y,z;
//        };
//        struct int4 {
//        private:
//            int x,y,z,w;
//        };
//
//        struct bool2 {
//        private:
//            bool x,y;
//        };
//        struct bool3 {
//        private:
//            bool x,y,z;
//        };
//        struct bool4 {
//        private:
//            bool x,y,z,w;
//        };
      
    }
  
}
#endif

#endif //DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H
