#pragma once

#include "dehancer/CameraLutXmp.h"
#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/StreamSpace.h"
#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/clut/CLutTransform.h"

namespace dehancer {

class CameraProfile : public Command {
public:
	explicit CameraProfile(const void* command_queue,
	                       const StreamSpace& space = stream_space_identity(),
	                       StreamSpaceDirection direction = StreamSpaceDirection::DHCR_None,
	                       bool wait_until_completed = WAIT_UNTIL_COMPLETED);

	Error load(const CameraLutXmp& xmp);

	const std::shared_ptr<CLut>& get() const;

public:
	StreamSpace space_;
	StreamSpaceDirection direction_;
	std::shared_ptr<CLut> clut_;
};
}  // namespace dehancer
