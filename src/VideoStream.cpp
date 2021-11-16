//
// Created by denn on 15.11.2021.
//

#include "dehancer/gpu/Lib.h"
#include <opencv2/opencv.hpp>
#include <utility>

namespace dehancer {
    
    namespace impl {
        struct VideoStream {
            
            ~VideoStream () {
              m_cap->release();
            }
            
            explicit VideoStream (const void *command_queue, const std::string &file_path) :
                    m_command_queue(command_queue),
                    m_file_path(file_path),
                    m_cap(std::make_unique<cv::VideoCapture>(m_file_path)) {
              m_cap->setExceptionMode(false);
              
              if (!m_cap->isOpened()) {
                std::stringstream ss;
                ss << "Error opening video stream or file: " << file_path;
                throw std::runtime_error(ss.str());
              }
              
              m_desc.fps = static_cast<float>(m_cap->get(cv::CAP_PROP_FPS));
              m_desc.frame.count = static_cast<int>(m_cap->get(cv::CAP_PROP_FRAME_COUNT));
              m_desc.frame.duration = 1000.0f / static_cast<float>(m_desc.fps);
              m_desc.frame.size.width = static_cast<size_t>(m_cap->get(cv::CAP_PROP_FRAME_WIDTH));
              m_desc.frame.size.height = static_cast<size_t>(m_cap->get(cv::CAP_PROP_FRAME_HEIGHT));
              
              m_cap->set(cv::CAP_PROP_POS_FRAMES, m_desc.frame.count-1);
              m_desc.time = static_cast<float>(m_cap->get(cv::CAP_PROP_POS_MSEC))+m_desc.frame.duration;
              m_cap->set(cv::CAP_PROP_POS_FRAMES, 0);
              m_desc.bitrate = static_cast<float>(m_cap->get(cv::CAP_PROP_BITRATE));
              
              cv::Mat frame;
              *m_cap >> frame;
              
              m_desc.type = CV_MAT_TYPE(frame.type());
              
              m_desc.frame.channels = static_cast<int>(frame.channels());
              m_desc.frame.channel_depth = frame.depth();
              m_cap->set(cv::CAP_PROP_POS_AVI_RATIO, 0);
              
              switch (m_desc.frame.channel_depth) {
                case CV_8S:
                case CV_8U:
                  m_scale = 1.0f / (256.0f-1);
                  break;
                case CV_16U:
                  m_scale = 1.0f / (65536.0f-1);
                  break;
                case CV_32S:
                  m_scale = 1.0f / (16777216.0f-1);
                  break;
                case CV_16F:
                case CV_32F:
                case CV_64F:
                  m_scale = 1.0f;
                  break;
                default:
                  throw std::runtime_error("Image pixel depth is not supported");
              }
  
              m_convert_function = std::make_shared<dehancer::Function>(m_command_queue,"kernel_bgr8_to_texture");
              m_frame_texture = m_convert_function->make_texture(m_desc.frame.size.width,m_desc.frame.size.height);
            }
        
            dehancer::Texture convert(const cv::Mat& frame) {
  
              auto mem = dehancer::MemoryHolder::Make(
                      m_command_queue,
                      frame.data,
                      frame.total() * frame.channels()
              );
              
              m_convert_function->execute([this,&mem](dehancer::CommandEncoder& command_encoder){
      
                  command_encoder.set(mem, 0);
                  command_encoder.set(m_scale, 1);
                  command_encoder.set(m_frame_texture, 2);
      
                  return dehancer::CommandEncoder::Size::From(m_frame_texture);
              });
              
              return m_frame_texture;
            }
            
        public:
            const void *m_command_queue;
            std::string m_file_path;
            VideoDesc m_desc;
            std::unique_ptr<cv::VideoCapture> m_cap;
            int m_video_index = 0;
            float m_video_time = 0.0f;
            float m_scale = 1.0f;
            std::shared_ptr<dehancer::Function> m_convert_function;
            Texture m_frame_texture;
        };
    }
    
    VideoStream::VideoStream (const void *command_queue, const std::string &file_path) :
            impl_(std::make_shared<impl::VideoStream>(command_queue, file_path)) {
    }
    
    dehancer::expected<VideoStream, Error> VideoStream::Open (const void *command_queue, const std::string &file_path) {
      try {
        return VideoStream(command_queue, file_path);
      }
      catch (const std::runtime_error &e) {
        return dehancer::make_unexpected(Error(CommonError::NOT_SUPPORTED, e.what()));
      }
    }
    
    const VideoDesc &VideoStream::get_desc () const {
      return impl_->m_desc;
    }
    
    Texture VideoStream::next_texture () const {
      
      cv::Mat frame; impl_->m_cap->read(frame);
  
      impl_->m_video_time = static_cast<float>(impl_->m_cap->get(cv::CAP_PROP_POS_MSEC));
      impl_->m_video_index = static_cast<int>(impl_->m_cap->get(cv::CAP_PROP_POS_FRAMES));
  
      if (frame.empty()) {
        return nullptr;
      }
  
      return impl_->convert(frame);
    }
    
    Texture VideoStream::previous_texture () const {
      return get_texture_at_index(impl_->m_video_index - 1);
    }
    
    Texture VideoStream::get_texture_at_time (float time) const {
      
      impl_->m_video_time = time;
      
      if (impl_->m_video_time >= impl_->m_desc.time) {
        impl_->m_video_index = impl_->m_desc.frame.count;
        impl_->m_video_time = impl_->m_desc.time - impl_->m_desc.frame.duration;
        return nullptr;
      }
      else if (impl_->m_video_time<0) {
        impl_->m_video_index = 0;
        impl_->m_video_time = 0;
        return nullptr;
      }
      
      impl_->m_cap->set(cv::CAP_PROP_POS_MSEC, impl_->m_video_time);
  
      impl_->m_video_index = static_cast<int>(impl_->m_cap->get(cv::CAP_PROP_POS_FRAMES));
  
      cv::Mat frame; impl_->m_cap->retrieve(frame);
      
      if (frame.empty()) {
        return nullptr;
      }
      
      return impl_->convert(frame);
    }
    
    Texture VideoStream::get_texture_at_index (int index) const {
      
      impl_->m_video_index = index;
      
      if (index > impl_->m_desc.frame.count) {
        impl_->m_video_index = impl_->m_desc.frame.count-1;
        impl_->m_video_time = impl_->m_desc.time-impl_->m_desc.frame.duration;
        return nullptr;
      }
      else if (impl_->m_video_index < 0) {
        impl_->m_video_index = 0;
        impl_->m_video_time = 0;
        return nullptr;
      }
      
      impl_->m_cap->set(cv::CAP_PROP_POS_FRAMES, index);
  
      impl_->m_video_time = static_cast<float>(impl_->m_cap->get(cv::CAP_PROP_POS_MSEC));
  
      cv::Mat frame; impl_->m_cap->retrieve(frame);
  
      if (frame.empty()) {
        return nullptr;
      }
  
      return impl_->convert(frame);
    }
    
    int VideoStream::get_frame_index () const {
      return impl_->m_video_index;
    }
    
    float VideoStream::get_frame_time () const {
      return impl_->m_video_time;
    }
    
    void VideoStream::seek_begin () {
      impl_->m_cap->set(cv::CAP_PROP_POS_AVI_RATIO, 0);
      impl_->m_cap->set(cv::CAP_PROP_POS_MSEC, 0);
      impl_->m_video_time = -impl_->m_desc.frame.duration;
      impl_->m_video_index = -1;
    }
    
    void VideoStream::seek_end () {
      impl_->m_cap->set(cv::CAP_PROP_POS_AVI_RATIO, 1);
      impl_->m_video_time = static_cast<float>(impl_->m_cap->get(cv::CAP_PROP_POS_MSEC));
      impl_->m_video_index = impl_->m_desc.frame.count;
      impl_->m_cap->set(cv::CAP_PROP_POS_FRAMES, impl_->m_video_index-1);
    }
  
}