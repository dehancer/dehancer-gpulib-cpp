#pragma once

#include <cinttypes>
#include <vector>

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/TextureInput.h"
#include "dehancer/gpu/clut/CLut.h"

namespace dehancer {

class CLutCubeInput : public CLut, protected TextureInput {
public:
	explicit CLutCubeInput(const void* command_queue);

	const Texture& get_texture() override;
	[[nodiscard]] const Texture& get_texture() const override;

	[[nodiscard]] size_t get_lut_size() const override {
		return lut_size_;
	};

	[[nodiscard]] Type get_lut_type() const override {
		return Type::lut_3d;
	};

	Error load_from_data(float* buffer, size_t lut_size) override;
	Error load_from_data(const std::vector<float>& buffer, size_t lut_size) override;

	friend std::istream& operator>>(std::istream& is, CLutCubeInput& dt);

	~CLutCubeInput() override;

private:
	// std::shared_ptr<TextureInput> texture_input_;
	size_t lut_size_;
};
}  // namespace dehancer
