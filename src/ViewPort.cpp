//
//  ViewPort.cpp
//  dehancer_gpulib_metal_iphoneos
//
//  Created by Dmitry Sukhorukov on 15.11.2023.
//

#include "dehancer/gpu/ViewPort.h"
#include <algorithm>

namespace dehancer {
ViewPortHolder::ViewPortHolder(const viewport::Origin &origin
                               , const viewport::Size &size
                               , const viewport::Size &source_size
                               , const viewport::Size &original_size)
    : _origin(origin), _size(size), _source_size(source_size), _original_size(original_size){
        _origin.x = std::clamp(_origin.x, 0.f, 1.f);
        _origin.y = std::clamp(_origin.y, 0.f, 1.f);

        if(_origin.x + _size.width > 1.f) _size.width = 1.f - _origin.x;
        if(_origin.y + _size.height > 1.f) _size.height = 1.f - _origin.y;
    }

ViewPort ViewPortHolder::Make(const viewport::Origin &origin
                              , const viewport::Size &size
                              , const viewport::Size &source_size
                              , const viewport::Size &original_size)
    {
        return std::make_shared<ViewPortHolder>(origin, size, source_size, original_size);
    }

    const viewport::Origin& ViewPortHolder::get_origin() const {
        return _origin;
    }

    const viewport::Size& ViewPortHolder::get_size() const {
        return _size;
    }

    const viewport::Size& ViewPortHolder::get_source_size() const {
        return _source_size;
    }

    const viewport::Size& ViewPortHolder::get_original_size() const {
        return _original_size;
    }
}
