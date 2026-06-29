import argparse
import os
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont
from PIL import ImageTk


# =========================
# Direct-run configuration
# =========================
# Set RUN_FROM_CONFIG = True, then edit the paths below and run:
#   python E:\桌面\a2\scripts\make_zoom_comparison.py
RUN_FROM_CONFIG = False

# Folder containing ordered method images. Images are processed by filename order.
CONFIG_INPUT_DIR = r"E:\path\to\method_images"

# Output folder. Crops are saved to CONFIG_OUTPUT_DIR\crops.
CONFIG_OUTPUT_DIR = r"E:\path\to\zoom_outputs"

# Use manual mouse selection on the first image.
CONFIG_SELECT_BOX = True

# Used only when CONFIG_SELECT_BOX = False. Format: (x, y, width, height)
CONFIG_BOX = (180, 220, 120, 120)

# Crop enlargement factor.
CONFIG_ZOOM_SCALE = 3.0

# Also save full images resized to the first image size.
CONFIG_SAVE_RESIZED = True

# Also save full images with the red crop box drawn.
CONFIG_SAVE_MARKED = True


def parse_method(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Method must use the format Label=path/to/image.png"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path.strip().strip('"'))
    if not label:
        raise argparse.ArgumentTypeError("Method label cannot be empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Image does not exist: {path}")
    return label, path


def parse_box(value):
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Box must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Box width and height must be positive")
    return x, y, w, h


def load_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_image(img, target_w, target_h, fill=(255, 255, 255)):
    img = img.convert("RGB")
    scale = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def draw_label(draw, xy, text, font, text_fill=(20, 20, 20), bg_fill=(255, 255, 255)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x = 8
    pad_y = 4
    rect = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    )
    draw.rounded_rectangle(rect, radius=3, fill=bg_fill)
    draw.text((x, y), text, fill=text_fill, font=font)


def draw_rect(draw, box, color, width):
    x, y, w, h = box
    for i in range(width):
        draw.rectangle((x - i, y - i, x + w + i, y + h + i), outline=color)


def clamp_box(box, img_w, img_h):
    x, y, w, h = box
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def image_files(input_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [
        path
        for path in Path(input_dir).iterdir()
        if path.is_file() and path.suffix.lower() in exts
    ]
    return sorted(files, key=lambda p: p.name.lower())


def select_box_interactively(image_path):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    root = tk.Tk()
    root.title(
        "Drag to select zoom box, press Enter to confirm, Esc to cancel"
    )

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.85)
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    view_w = max(1, int(round(img_w * scale)))
    view_h = max(1, int(round(img_h * scale)))
    view = img.resize((view_w, view_h), Image.Resampling.LANCZOS)

    photo = ImageTk.PhotoImage(view)
    canvas = tk.Canvas(root, width=view_w, height=view_h, cursor="crosshair")
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)

    state = {
        "start": None,
        "rect": None,
        "box": None,
    }

    def on_press(event):
        state["start"] = (event.x, event.y)
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="red",
            width=3,
        )

    def on_drag(event):
        if state["start"] is None or state["rect"] is None:
            return
        x0, y0 = state["start"]
        x1 = max(0, min(event.x, view_w - 1))
        y1 = max(0, min(event.y, view_h - 1))
        canvas.coords(state["rect"], x0, y0, x1, y1)

    def on_release(event):
        if state["start"] is None:
            return
        x0, y0 = state["start"]
        x1 = max(0, min(event.x, view_w - 1))
        y1 = max(0, min(event.y, view_h - 1))
        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)
        if right - left < 2 or bottom - top < 2:
            state["box"] = None
            return
        ox = int(round(left / scale))
        oy = int(round(top / scale))
        ow = int(round((right - left) / scale))
        oh = int(round((bottom - top) / scale))
        state["box"] = clamp_box((ox, oy, ow, oh), img_w, img_h)

    def confirm(_event=None):
        if state["box"] is None:
            print("No valid box selected yet. Drag a rectangle first.")
            return
        root.destroy()

    def cancel(_event=None):
        state["box"] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", confirm)
    root.bind("<Escape>", cancel)

    root.mainloop()
    if state["box"] is None:
        raise RuntimeError("Box selection cancelled or no valid box selected.")
    return state["box"]


def save_zoom_crops_from_dir(
    input_dir,
    output_dir,
    box,
    zoom_scale=2.0,
    save_resized=False,
    save_marked=False,
):
    files = image_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    resized_dir = output_dir / "resized"
    marked_dir = output_dir / "marked"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if save_resized:
        resized_dir.mkdir(parents=True, exist_ok=True)
    if save_marked:
        marked_dir.mkdir(parents=True, exist_ok=True)

    first = Image.open(files[0]).convert("RGB")
    ref_w, ref_h = first.size
    box = clamp_box(box, ref_w, ref_h)
    x, y, w, h = box
    zoom_w = max(1, int(round(w * zoom_scale)))
    zoom_h = max(1, int(round(h * zoom_scale)))

    print(f"Reference image: {files[0].name} ({ref_w}x{ref_h})")
    print(f"Crop box on reference size: x={x}, y={y}, w={w}, h={h}")
    print(f"Processing {len(files)} images...")

    for index, path in enumerate(files, start=1):
        img = Image.open(path).convert("RGB")
        if img.size != (ref_w, ref_h):
            img = img.resize((ref_w, ref_h), Image.Resampling.LANCZOS)

        if save_resized:
            img.save(resized_dir / path.name)

        crop = img.crop((x, y, x + w, y + h))
        if zoom_scale != 1.0:
            crop = crop.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)
        crop.save(crops_dir / path.name)

        if save_marked:
            marked = img.copy()
            marked_draw = ImageDraw.Draw(marked)
            draw_rect(marked_draw, box, (220, 30, 30), 4)
            marked.save(marked_dir / path.name)

        print(f"[{index:02d}/{len(files):02d}] {path.name}")

    print(f"Saved zoom crops: {crops_dir}")
    if save_resized:
        print(f"Saved resized images: {resized_dir}")
    if save_marked:
        print(f"Saved marked images: {marked_dir}")


def make_comparison(methods, box, output, crops_dir=None, cell_width=260, zoom_scale=2.0):
    images = []
    for label, path in methods:
        img = Image.open(path).convert("RGB")
        images.append((label, path, img))

    ref_w, ref_h = images[0][2].size
    box = clamp_box(box, ref_w, ref_h)
    x, y, w, h = box

    title_h = 34
    gap = 10
    margin = 16
    border_color = (220, 30, 30)
    border_w = 4

    cell_h = int(round(cell_width * ref_h / ref_w))
    zoom_h = max(80, int(round(h * zoom_scale)))
    zoom_w = max(80, int(round(w * zoom_scale)))
    zoom_w = min(zoom_w, cell_width)

    font = load_font(18)
    small_font = load_font(15)

    cols = len(images)
    canvas_w = margin * 2 + cols * cell_width + (cols - 1) * gap
    canvas_h = margin * 2 + title_h + cell_h + gap + zoom_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if crops_dir:
        crops_dir = Path(crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    for idx, (label, path, img) in enumerate(images):
        col_x = margin + idx * (cell_width + gap)
        label_y = margin

        draw_label(draw, (col_x + 6, label_y + 4), label, font)

        fitted, scale, ox, oy = fit_image(img, cell_width, cell_h)
        rect_on_cell = (
            int(round(x * scale + ox)),
            int(round(y * scale + oy)),
            int(round(w * scale)),
            int(round(h * scale)),
        )
        fitted_draw = ImageDraw.Draw(fitted)
        draw_rect(fitted_draw, rect_on_cell, border_color, border_w)

        image_y = margin + title_h
        canvas.paste(fitted, (col_x, image_y))

        crop_box = clamp_box(box, img.width, img.height)
        cx, cy, cw, ch = crop_box
        crop = img.crop((cx, cy, cx + cw, cy + ch))
        if crops_dir:
            safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
            crop.save(crops_dir / f"{safe_label}_crop.png")

        zoom = crop.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)
        zoom_canvas = Image.new("RGB", (cell_width, zoom_h), (255, 255, 255))
        zx = (cell_width - zoom_w) // 2
        zoom_canvas.paste(zoom, (zx, 0))
        zoom_draw = ImageDraw.Draw(zoom_canvas)
        draw_rect(zoom_draw, (zx, 0, zoom_w - 1, zoom_h - 1), border_color, border_w)

        zoom_y = margin + title_h + cell_h + gap
        canvas.paste(zoom_canvas, (col_x, zoom_y))

    note = f"Zoom box: x={x}, y={y}, w={w}, h={h}"
    note_bbox = draw.textbbox((0, 0), note, font=small_font)
    draw.text(
        (canvas_w - margin - (note_bbox[2] - note_bbox[0]), canvas_h - margin + 2),
        note,
        fill=(120, 120, 120),
        font=small_font,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Saved comparison: {output}")
    if crops_dir:
        print(f"Saved crops: {crops_dir}")


def main():
    if RUN_FROM_CONFIG:
        box = CONFIG_BOX
        if CONFIG_SELECT_BOX:
            files = image_files(CONFIG_INPUT_DIR)
            if not files:
                raise FileNotFoundError(f"No images found in {CONFIG_INPUT_DIR}")
            box = select_box_interactively(files[0])
            print(f"Selected box: {box[0]},{box[1]},{box[2]},{box[3]}")
        save_zoom_crops_from_dir(
            input_dir=CONFIG_INPUT_DIR,
            output_dir=CONFIG_OUTPUT_DIR,
            box=box,
            zoom_scale=CONFIG_ZOOM_SCALE,
            save_resized=CONFIG_SAVE_RESIZED,
            save_marked=CONFIG_SAVE_MARKED,
        )
        return

    parser = argparse.ArgumentParser(
        description="Create paper-style comparison figures or batch zoom crops."
    )
    parser.add_argument(
        "--method",
        action="append",
        type=parse_method,
        help='Method image in the format "Label=path". Repeat for each method.',
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Folder of ordered method images. Images are sorted by filename.",
    )
    parser.add_argument(
        "--box",
        type=parse_box,
        default=None,
        help="Crop box on the first image, format x,y,w,h.",
    )
    parser.add_argument("--output", help="Output comparison image path.")
    parser.add_argument("--output-dir", help="Output folder for batch crops.")
    parser.add_argument("--crops-dir", default=None, help="Optional directory for individual crops.")
    parser.add_argument("--cell-width", type=int, default=260, help="Width of each method panel.")
    parser.add_argument("--zoom-scale", type=float, default=2.0, help="Scale factor for the crop.")
    parser.add_argument(
        "--save-resized",
        action="store_true",
        help="In batch mode, also save images resized to the first image size.",
    )
    parser.add_argument(
        "--save-marked",
        action="store_true",
        help="In batch mode, also save full images with the crop box drawn.",
    )
    parser.add_argument(
        "--select-box",
        action="store_true",
        help="In batch mode, manually select the crop box on the first image.",
    )
    args = parser.parse_args()

    if args.input_dir:
        if not args.output_dir:
            parser.error("--output-dir is required when using --input-dir")
        box = args.box
        if args.select_box:
            files = image_files(args.input_dir)
            if not files:
                parser.error(f"No images found in {args.input_dir}")
            box = select_box_interactively(files[0])
            print(f"Selected box: {box[0]},{box[1]},{box[2]},{box[3]}")
        if box is None:
            parser.error("--box is required unless --select-box is used")
        save_zoom_crops_from_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            box=box,
            zoom_scale=args.zoom_scale,
            save_resized=args.save_resized,
            save_marked=args.save_marked,
        )
    else:
        if args.box is None:
            parser.error("--box is required in comparison mode")
        if not args.method:
            parser.error("--method is required unless --input-dir is used")
        if not args.output:
            parser.error("--output is required unless --input-dir is used")
        make_comparison(
            methods=args.method,
            box=args.box,
            output=args.output,
            crops_dir=args.crops_dir,
            cell_width=args.cell_width,
            zoom_scale=args.zoom_scale,
        )


if __name__ == "__main__":
    main()
