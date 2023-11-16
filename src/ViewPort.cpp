//
//  ViewPort.cpp
//  dehancer_gpulib_metal_iphoneos
//
//  Created by Dmitry Sukhorukov on 15.11.2023.
//

#include "dehancer/gpu/ViewPort.h"

namespace dehancer {
    ViewPortHolder::ViewPortHolder(const viewport::Origin &origin, const viewport::Size &size)
    : _origin(origin), _size(size) {
        
    }

    ViewPort ViewPortHolder::Make(const viewport::Origin &origin, const viewport::Size &size)
    {
        return std::make_shared<ViewPortHolder>(origin, size);
    }

    const viewport::Origin& ViewPortHolder::get_origin() const {
        return _origin;
    }

    const viewport::Size& ViewPortHolder::get_size() const {
        return _size;
    }
}
