#pragma once

#include <memory>

namespace dehancer {

    class ViewPortHolder;

    using ViewPort = std::shared_ptr<ViewPortHolder>;

    namespace viewport {
        struct Origin {
            static Origin Make(const float x, const float y) {
                return { .x = x, .y = y};
            }
            float x;
            float y;
        };

        struct Size {
            static Size Make(const float width, const float height) {
                return { .width = width, .height = height};
            }
            float width;
            float height;
        };
    }

    class ViewPortHolder {
    public:
        
        ViewPortHolder() = delete;
        
        explicit ViewPortHolder(const viewport::Origin &origin, const viewport::Size &size);
        
        static ViewPort Make(const viewport::Origin &origin, const viewport::Size &size);
        
        [[nodiscard]] const viewport::Origin& get_origin() const;
        
        [[nodiscard]] const viewport::Size& get_size() const;
        
    private:
        viewport::Origin _origin;
        viewport::Size _size;
    };    
}
