//
// Created by denn on 31.12.2020.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

#include <opencv2/opencv.hpp>

int capture_video(int dev_num,
                  const void* command_queue,
                  const std::string& platform,
                  const std::string& file,
                  const std::string& name,
                  bool reverse,
                  int index = 0
) {
  std::cout << file << std::endl;
  
  auto video_stream = dehancer::VideoStream::Open(command_queue, file);
  
  if (!video_stream) {
    std::cout << "Error: " << video_stream.error() << std::endl;
    return -1;
  }
  
  std::cout
          << "Video        WxH:" << video_stream->get_desc().keyframe.size.width << "x" << video_stream->get_desc().keyframe.size.height << std::endl
          << "Video   channels:" << video_stream->get_desc().keyframe.channels << std::endl
          << "Video      depth:" << video_stream->get_desc().keyframe.channel_depth << std::endl
          << "Video      total:" << video_stream->get_desc().keyframe.count << std::endl
          << "Video   duration:" << video_stream->get_desc().keyframe.duration << std::endl
          << "Video        fps:" << video_stream->get_desc().fps <<std::endl
          << "Video    bitrate:" << video_stream->get_desc().bitrate << std::endl
          << "Video       time:" << video_stream->get_desc().time <<std::endl
          << "Video       type:" << video_stream->get_desc().type <<std::endl
          ;
  
  if (reverse) {
    if (index>0){
      video_stream->seek_at_time(video_stream->get_desc().time - (float)index*video_stream->get_desc().keyframe.duration);
    }
    if (index<0){
      video_stream->seek_at_time(- (float)index*video_stream->get_desc().keyframe.duration);
    }
    else
      video_stream->skip_forward();
  }
  else {
    if (index<0){
      video_stream->seek_at_time(video_stream->get_desc().time + (float)index*video_stream->get_desc().keyframe.duration);
    }
    else if (index>0){
      video_stream->seek_at_time((float)index*video_stream->get_desc().keyframe.duration);
    }
    else
      video_stream->skip_backward();
  }
  
  for (int i = 0; i<16; i++){
    
    dehancer::Texture texture;
    
    if (reverse)
      texture = video_stream->previous_texture();
    else
      texture = video_stream->next_texture();
    
    std::cout
            << "next["<< video_stream->current_keyframe_position()
            << "] frame at: "
            << video_stream->current_keyframe_time()
            << std::endl;
  
    if (!texture) break;
  
    {
      std::stringstream ss;
      
      std::string fn = name;
      
      std::transform(fn.begin(), fn.end(), fn.begin(), [](char x) {
          if (x=='.') return '_';
          return x;
      });
      
      ss << "dev_" << dev_num << "_" << platform << "_" << fn << "_frame_" << video_stream->current_keyframe_position() << ".jpg";
      std::ofstream os(ss.str(), std::ostream::binary | std::ostream::trunc);
      
      os << dehancer::TextureOutput(command_queue, texture, {
              .type = dehancer::TextureIO::Options::Type::jpeg,
              .compression = 0.1
      });
    }
    
    
    if (reverse) {
      if (video_stream->current_keyframe_position() <= 0) break;
    }
    else
    if (video_stream->current_keyframe_position() >= video_stream->get_desc().keyframe.count - 1) break;
  }
  
  return 0 ;
}

auto io_texture_test_forward = [] (int dev_num,
                                   const void* command_queue,
                                   const std::string& platform) {
    
    for (auto file: VIDEO_FILES) {
      std::string vfile = IMAGES_DIR; vfile +="/"; vfile+= file;
      if(capture_video(dev_num, command_queue, platform, vfile, file, false)<0) return -1;
    }
    
    return 0;
};

auto io_texture_test_reverse = [] (int dev_num,
                                   const void* command_queue,
                                   const std::string& platform) {
    
    for (auto file: VIDEO_FILES) {
      std::string vfile = IMAGES_DIR; vfile +="/"; vfile+= file;
      if(capture_video(dev_num, command_queue, platform, vfile, file, true)<0) return -1;
    }
    
    return 0;
};

auto io_texture_test_last = [] (int dev_num,
                                   const void* command_queue,
                                   const std::string& platform) {
    
    for (auto file: VIDEO_FILES) {
      std::string vfile = IMAGES_DIR; vfile +="/"; vfile+= file;
      if(capture_video(dev_num, command_queue, platform, vfile, file, false, -2)<0) return -1;
    }
    
    return 0;
};