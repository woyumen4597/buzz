# 视频高光候选

Buzz 主窗口现在包含“视频高光”工作区。打开 Buzz 后，切换到该 Tab，选择视频文件，可选选择 SRT，然后点击“生成高光合集”即可一键处理。默认目标时长为原视频的三分之一：1 小时视频约生成 20 分钟高光，2 小时视频约生成 40 分钟高光。窗口长度、步长和候选上限是内部分析策略，不需要在日常使用中调整。生成过程在后台执行，完成后点击“打开结果”即可查看成片和复核页面。

该页面是轻量入口，候选卡片、缩略图和预览仍在生成的静态 HTML 中查看，不会改变现有转写工作区。候选分数是用于排序的启发式分数：结合场景变化、字幕密度、画面运动和窗口时长，不是模型对“高光”的确定性判断；没有字幕或运动信号时，同类候选出现相同分数是正常的。

Buzz 也提供一个不依赖 Qt 主窗口的独立命令，用于生成供人工排查的视频候选片段。

```bash
uv run python -m buzz.highlights.cli video.mp4 \
  --srt video.srt \
  --output-dir video_highlights
```

也可以在安装后使用 `buzz-highlights` 命令。默认会检查 FFmpeg、生成固定时间窗口，并尝试进行画面场景变化扫描。场景扫描失败时会记录 warning 并自动退回固定窗口；使用 `--no-scene-detection` 可以主动关闭场景扫描。要直接自动选片并输出合集，使用 `--auto-edit`。

```bash
uv run buzz-highlights video.mp4 --srt video.srt \
  --auto-edit --output-dir video_highlights
```

自动模式默认按源视频三分之一计算目标时长，片段数量不限，按评分和时间区间做非重叠选择，主体片段总时长尽量填满目标时长但不会超过目标。也可以用 `--target-duration 600` 指定 10 分钟；用 `--max-auto-clips 6` 才会限制片段数量。输出目录会生成 `highlights.mp4` 和 `auto-selection.json`。这是可解释的自动初剪，不是对内容高光的确定性判断。

常用选项：

- `--keep-static-scenes`：不自动将画面基本静止的候选标为“忽略”。默认会进行低帧率画面运动分析；静止候选不会删除，仍可在结果页手动改为“保留”。
- `--target-duration 600`：指定 10 分钟目标成片；不指定时按源视频三分之一计算。
- `--max-auto-clips 6`：可选地限制自动成片中的片段数量，默认不限。
- `--no-previews`：不生成预览文件，只生成成片和时间信息。
- `--gif --gif-limit 20`：为排名靠前的候选额外生成 GIF。
- `--keep-existing`：复用输出目录中已有的缩略图和预览，并读取 `.highlight-progress.json` 继续未完成的候选。
- `--open`：生成后尝试打开 `index.html`。

GUI 页面会显示素材生成百分比，并提供“取消”按钮。点击“取消”会停止后续候选处理，已经完成的缩略图/预览和 `.highlight-progress.json` 会保留；下次使用相同视频、相同主要参数并保持“复用已有文件”时，会从已完成位置继续。GUI 默认开启复用已有文件。

输出目录包含 `index.html`、`candidates.json`、`manifest.json`、`selected.json`、`clips.txt`、`thumbnails/` 和 `previews/`。HTML 页面可以按分数、时间和状态排序，按字幕关键词筛选，并将保留的候选状态保存到浏览器 `localStorage`。导出按钮会下载已选 JSON 和 FFmpeg 命令清单。通过 Buzz GUI 打开的结果页还提供“生成最终视频”：它会按时间顺序裁剪所有“保留”片段并自动拼接为输出目录中的 `highlights.mp4`。生成过程不会删除原视频或被忽略的候选。

页面可直接用浏览器打开。如果浏览器限制 `file://` 页面的视频加载，可在输出目录执行：

```bash
python -m http.server 8000
```

然后访问 `http://127.0.0.1:8000/`。
