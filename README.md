# YOLO C++

## Export model

```bash
yolo export model=yolov8n.pt format=onnx simplify=True
```

## Compile

```bash
meson setup builddir
ninja -C builddir
```

## Run

```bash
./builddir/main
```
