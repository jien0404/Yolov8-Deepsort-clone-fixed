# Mở file:
```code
D:\zalo_project\YOLOv8-DeepSORT-Object-Tracking\ultralytics\nn\tasks.py
```

# Tìm đoạn gọi torch.load(attempt_download(w), map_location='cpu') (dòng tương ứng theo traceback).

# Thay bằng đoạn sau (ví dụ sửa quanh chỗ gọi torch.load):
```code
import ultralytics.nn.tasks as _ul_tasks
torch.serialization.add_safe_globals([
            _ul_tasks.DetectionModel,
            nn.modules.container.Sequential,
            _ul_modules.Conv,
            nn.modules.conv.Conv2d,
            nn.modules.batchnorm.BatchNorm2d,
            nn.modules.activation.SiLU,
            _ul_modules.C2f,
            nn.modules.container.ModuleList,
            _ul_modules.Bottleneck,
            _ul_modules.SPPF,
            nn.modules.pooling.MaxPool2d,
            nn.modules.upsampling.Upsample,
            _ul_modules.Concat,
            _ul_modules.Detect,
            _ul_modules.DFL
        ])
ckpt = torch.load(attempt_download(w), map_location='cpu')


1. Sửa code trong thư viện bạn đang dùng

Vì lỗi xảy ra trong file …/deep_sort_pytorch/deep_sort/…/detection.py có dòng:

self.tlwh = np.asarray(tlwh, dtype=np.float)


Bạn có thể thay dtype=np.float bằng dtype=float (hoặc dtype=np.float64) ở dòng đó. Ví dụ:

self.tlwh = np.asarray(tlwh, dtype=float)