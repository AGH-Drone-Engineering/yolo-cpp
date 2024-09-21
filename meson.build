project('yolo-cpp', 'cpp', default_options : ['cpp_std=c++14'])

opencv_dep = dependency('opencv4')

yolocpp_lib = shared_library('yolocpp', 'yolocpp.cpp', dependencies : opencv_dep)

executable('main', 'main.cpp', dependencies : opencv_dep, link_args : '-lyolocpp')
