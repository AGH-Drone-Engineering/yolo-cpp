# YOLO C++

## Use on Raspberry Pi

### Export model

```bash
yolo export model=yolov8n.pt format=onnx simplify=True
```

### Install library

```bash
sudo cp libyolocpp.so /usr/lib
sudo cp yolocpp.hpp /usr/include
```

### Use

```cpp
#include "yolocpp.hpp"

int main() {
    YOLOCPP inf("yolov8n.onnx", 640, 640, {"classes", "..."});
}
```

## Build library (optional)

### Build static OpenCV

```bash
# download opencv-4.10.0
cd opencv-build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=$HOME/opencv-install -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_SHARED_LIBS=OFF -DOPENCV_GENERATE_PKGCONFIG=YES -DWITH_TBB=OFF -DWITH_IPP=OFF -DWITH_GSTREAMER=OFF -DWITH_FFMPEG=OFF -DWITH_VTK=OFF -DWITH_OPENCL=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_LIST=core,dnn -DWITH_1394=OFF -DBUILD_ZLIB=ON -DWITH_V4L=OFF -DWITH_CAROTENE=OFF -DWITH_ITT=OFF ../opencv-4.10.0
ninja
ninja install
```

### Compile yolocpp

```bash
export PKG_CONFIG_PATH=$HOME/opencv-install/lib/pkgconfig
meson setup -Doptimization=2 -Ddebug=false -Dstrip=true -Dprefer_static=true builddir
ninja -C builddir libyolocpp.so
```
