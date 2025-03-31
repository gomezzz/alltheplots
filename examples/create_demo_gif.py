import os
import logging
import numpy as np
from moviepy.editor import VideoClip, TextClip, ImageClip, concatenate_videoclips, ColorClip
from moviepy.config import change_settings

try:
    from PIL import Image

    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except Exception as e:
    print("Error patching PIL.Image.ANTIALIAS:", e)


import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alltheplots import plot

# ---------- Logging Configuration ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gif_script")

# ---------- Set ImageMagick Binary ----------
magick_path = r"C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"
if "IMAGEMAGICK_BINARY" not in os.environ:
    os.environ["IMAGEMAGICK_BINARY"] = magick_path
    logger.info("Set IMAGEMAGICK_BINARY environment variable.")
else:
    logger.info("IMAGEMAGICK_BINARY already set.")
change_settings({"IMAGEMAGICK_BINARY": magick_path})
logger.info(f"IMAGEMAGICK_BINARY is set to: {magick_path}")

# ---------- Generate Plot Files if They Don't Exist ----------

if not os.path.exists("plot_1d.png"):
    logger.info("Generating plot_1d.png")
    my_1D_tensor = np.linspace(-10, 10, 300) ** 2 + np.random.normal(0, 10, 300)
    plot(my_1D_tensor, filename="plot_1d.png", dpi=100, show=False)

if not os.path.exists("plot_2d.png"):
    logger.info("Generating plot_2d.png")
    my_2D_tensor = np.sin(
        np.linspace(0, 4 * np.pi, 100).reshape(10, 10)
        + np.cos(np.linspace(0, 4 * np.pi, 100).reshape(10, 10))
    )
    plot(my_2D_tensor, filename="plot_2d.png", dpi=100, show=False)

if not os.path.exists("plot_3d.png"):
    logger.info("Generating plot_3d.png")

    def gaussian_3d_mod(
        resolution=30, sigma=(0.5, 0.7, 0.9), offset=(0.2, -0.1, 0.3), size=2, noise_level=0.05
    ):
        x = np.linspace(-size, size, resolution) + offset[0]
        y = np.linspace(-size, size, resolution) + offset[1]
        z = np.linspace(-size, size, resolution) + offset[2]
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        gauss = np.exp(
            -(
                (xx**2) / (2 * sigma[0] ** 2)
                + (yy**2) / (2 * sigma[1] ** 2)
                + (zz**2) / (2 * sigma[2] ** 2)
            )
        )
        noise = np.random.normal(scale=noise_level, size=gauss.shape)
        return gauss + noise

    my_3d_tensor = gaussian_3d_mod()
    plot(my_3d_tensor, filename="plot_3d.png", dpi=100, show=False)

if not os.path.exists("plot_nd.png"):
    logger.info("Generating plot_nd.png")

    def complex_topology_4d(resolution=48, noise_level=0.1):
        x = np.linspace(-3, 3, resolution)
        y = np.linspace(-3, 3, resolution)
        z = np.linspace(-3, 3, resolution)
        w = np.linspace(-3, 3, resolution)
        xx, yy, zz, ww = np.meshgrid(x, y, z, w, indexing="ij")
        g1 = np.exp(-(((xx + 1) ** 2 + (yy + 1) ** 2 + (zz + 1) ** 2 + (ww + 1) ** 2) / 2.0))
        g2 = np.exp(-(((xx - 1) ** 2 + (yy - 1) ** 2 + (zz - 1) ** 2 + (ww - 1) ** 2) / 2.0))
        sine_component = np.sin(2 * xx) * np.cos(2 * yy) * np.sin(2 * zz) * np.cos(2 * ww)
        composite = g1 + g2 + 0.5 * sine_component
        composite += np.random.normal(scale=noise_level, size=composite.shape)
        return composite

    my_nd_tensor = complex_topology_4d()
    plot(my_nd_tensor, filename="plot_nd.png", dpi=100, show=False)

# ---------- Helper Functions for Video Generation ----------


def typewriter_clip(
    text, duration=4, fps=24, fontsize=24, color="lime", bg_color="black", font="Courier-New"
):
    logger.info("Creating typewriter clip.")

    def make_frame(t):
        total_chars = len(text)
        current_chars = int(np.clip(total_chars * (t / duration), 0, total_chars))
        txt = text[:current_chars]
        if txt.strip() == "":
            txt = " "  # Ensure non-empty text
        clip = TextClip(
            txt, fontsize=fontsize, color=color, bg_color=bg_color, font=font, method="caption"
        )
        return clip.get_frame(0)

    return VideoClip(make_frame, duration=duration).set_fps(fps)


def loading_clip(duration=1.5, fontsize=28, color="white", bg_color="black", font="Courier-New"):
    logger.info("Creating loading clip.")

    def make_frame(t):
        num_dots = int((t * 3) % 4)
        txt = "Loading" + "." * num_dots
        if txt.strip() == "":
            txt = " "
        clip = TextClip(
            txt, fontsize=fontsize, color=color, bg_color=bg_color, font=font, method="caption"
        )
        return clip.get_frame(0)

    return VideoClip(make_frame, duration=duration).set_fps(24)


def pan_image_clip(image_path, duration=3, zoom_factor=1.05):
    logger.info(f"Creating pan image clip for {image_path}.")
    clip = ImageClip(image_path).set_duration(duration)
    # Custom zoom effect: scale factor goes from 1 to zoom_factor over the duration
    clip = clip.resize(lambda t: 1 + (zoom_factor - 1) * (t / duration))
    return clip


# ---------- Code Snippets for Each Example ----------

code_1d = """\
from alltheplots import plot
my_1D_tensor = np.linspace(-10, 10, 300)**2 + np.random.normal(0, 10, 300)
plot(my_1D_tensor)
"""

code_2d = """\
from alltheplots import plot
my_2D_tensor = np.sin(np.linspace(0, 4*np.pi, 100).reshape(10, 10) +
                      np.cos(np.linspace(0, 4*np.pi, 100).reshape(10, 10))
)
plot(my_2D_tensor)
"""

code_3d = """\
from alltheplots import plot

def gaussian_3d_mod(resolution=30, sigma=(0.5, 0.7, 0.9), 
                    offset=(0.2, -0.1, 0.3), size=2, noise_level=0.05):
    \"\"\"Create a 3D Gaussian density field with noise.\"\"\"
    x = np.linspace(-size, size, resolution) + offset[0]
    y = np.linspace(-size, size, resolution) + offset[1]
    z = np.linspace(-size, size, resolution) + offset[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    gauss = np.exp(-((xx**2)/(2*sigma[0]**2) + 
                     (yy**2)/(2*sigma[1]**2) + 
                     (zz**2)/(2*sigma[2]**2)))
    noise = np.random.normal(scale=noise_level, size=gauss.shape)
    return gauss + noise

my_3d_tensor = gaussian_3d_mod()
plot(my_3d_tensor)
"""

code_4d = """\
from alltheplots import plot

def complex_topology_4d(resolution=48, noise_level=0.1):
    \"\"\"Generate a 4D composite array with non-uniform topology.\"\"\"
    x = np.linspace(-3, 3, resolution)
    y = np.linspace(-3, 3, resolution)
    z = np.linspace(-3, 3, resolution)
    w = np.linspace(-3, 3, resolution)
    xx, yy, zz, ww = np.meshgrid(x, y, z, w, indexing='ij')
    g1 = np.exp(-(((xx + 1)**2 + (yy + 1)**2 + (zz + 1)**2 + (ww + 1)**2)/2.0))
    g2 = np.exp(-(((xx - 1)**2 + (yy - 1)**2 + (zz - 1)**2 + (ww - 1)**2)/2.0))
    sine_component = np.sin(2 * xx) * np.cos(2 * yy) * np.sin(2 * zz) * np.cos(2 * ww)
    composite = g1 + g2 + 0.5 * sine_component
    composite += np.random.normal(scale=noise_level, size=composite.shape)
    return composite

my_nd_tensor = complex_topology_4d()
plot(my_nd_tensor)
"""

# List of tuples: (code snippet, corresponding output image file)
segments = [
    (code_1d, "plot_1d.png"),
    (code_2d, "plot_2d.png"),
    (code_3d, "plot_3d.png"),
    (code_4d, "plot_nd.png"),
]

clips = []

# ---------- Process Each Segment to Create the Final Video ----------
for code_text, image_file in segments:
    logger.info(f"Processing segment with output image: {image_file}")
    tclip = typewriter_clip(
        code_text, duration=4, fontsize=22, color="lime", bg_color="black", font="Courier-New"
    )
    lclip = loading_clip(
        duration=1.5, fontsize=28, color="white", bg_color="black", font="Courier-New"
    )
    iplot = pan_image_clip(image_file, duration=3, zoom_factor=1.05)
    pause = ColorClip(size=(iplot.w, iplot.h), color=(0, 0, 0), duration=1)
    seg_clip = concatenate_videoclips([tclip, lclip, iplot, pause])
    clips.append(seg_clip)

final_clip = concatenate_videoclips(clips, method="compose")
logger.info("Writing final video file demo_video.mp4")
final_clip.write_videofile("demo_video.mp4", fps=24)
logger.info("Finished writing video file.")
# To export as GIF instead, uncomment the following line:
# final_clip.write_gif("demo_video.gif", fps=12)
