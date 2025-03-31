import os
import logging
import numpy as np
from moviepy.editor import (
    VideoClip,
    TextClip,
    ImageClip,
    concatenate_videoclips,
    ColorClip,
    CompositeVideoClip,
)
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

# ---------- Configuration Options ----------
# Set to True to generate a short version with fewer plots for quick iteration
SHORT_VERSION = True
# Select which plots to include (only used if SHORT_VERSION is True)
SELECTED_PLOTS = ["1d", "2d"]  # Choose from: "1d", "2d", "3d", "nd"

# High-resolution settings
HIGH_RES_SIZE = (800, 800)  # Square format for high resolution
LOW_RES_SIZE = (500, 500)  # Square format for low resolution
OUTPUT_FPS = 24

# Font configuration
# Use system fonts with backup options
CODE_FONT = None  # Let TextClip use default monospace
TEXT_FONT = None  # Let TextClip use default font
FONT_SIZES = {"intro_title": 50, "intro_text": 40, "code_title": 45, "code": 22, "outro": 60}

# Syntax highlighting colors (based on VSCode Dark+)
CODE_COLORS = {
    "keyword": "#569CD6",  # blue
    "function": "#DCDCAA",  # yellow
    "comment": "#6A9955",  # green
    "string": "#CE9178",  # orange
    "number": "#B5CEA8",  # light green
    "variable": "#9CDCFE",  # light blue
    "normal": "#D4D4D4",  # light gray (default text)
}

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


def create_intro_clip(duration=4.0, fps=24, bg_color="white"):
    """Create an intro animation with the slogan text appearing line by line."""
    logger.info("Creating intro clip.")

    # First create a simple clip with white background to get dimensions
    frame_size = HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE
    base_clip = ColorClip(size=frame_size, color=(255, 255, 255), duration=duration)
    base_clip = base_clip.set_fps(fps)

    # The three lines of text to display one by one
    lines = ["Plot any tensor", "one command", "0 extra parameters"]

    # Calculate timing for each line
    line_duration = duration / len(lines)

    # Create a clip for each line with proper timing
    line_clips = []
    for i, line in enumerate(lines):
        # Create a TextClip for this line
        line_clip = TextClip(
            line,
            fontsize=FONT_SIZES["intro_text"],
            color="black",
            method="caption",  # Use simpler caption method
            align="center",
            size=frame_size,
        )

        # Set position to center but with appropriate vertical offset
        vertical_position = (
            frame_size[1] // 2
            - FONT_SIZES["intro_text"] * (len(lines) - 1)
            + i * FONT_SIZES["intro_text"] * 2
        )
        line_clip = line_clip.set_position(("center", vertical_position))

        # Set start time and duration with fade-in effect
        start_time = i * line_duration
        line_clip = line_clip.set_start(start_time)
        line_clip = line_clip.set_duration(duration - start_time)

        # Add fade-in effect
        line_clip = line_clip.crossfadein(line_duration * 0.3)

        line_clips.append(line_clip)

    # Composite all line clips over the base white clip
    final_clip = CompositeVideoClip([base_clip] + line_clips)
    return final_clip


def create_outro_clip(duration=2.5, fps=24, bg_color="white"):
    """Create an outro animation with the package name in italic style."""
    logger.info("Creating outro clip.")

    # Create a white background clip first
    frame_size = HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE
    base_clip = ColorClip(size=frame_size, color=(255, 255, 255), duration=duration)
    base_clip = base_clip.set_fps(fps)

    # Create the outro text - just use plain text instead of HTML markup
    # We'll use a slant effect with positioning to simulate italic
    text_clip = TextClip(
        "alltheplots",
        fontsize=FONT_SIZES["outro"],
        color="black",
        method="caption",  # Use simpler caption method
        align="center",
        size=frame_size,
    )

    # Center the text
    text_clip = text_clip.set_position("center")

    # Set duration and add fade-in effect
    text_clip = text_clip.set_duration(duration)
    text_clip = text_clip.crossfadein(duration * 0.4)

    # Composite the clips
    final_clip = CompositeVideoClip([base_clip, text_clip])
    return final_clip


def create_title_clip(title_text, duration=1.5, fps=24):
    """Create a title clip for each code segment."""
    logger.info(f"Creating title clip: {title_text}")

    # Format the title text
    formatted_title = f"{title_text.upper()} Plotting"

    # Create a white background clip first
    frame_size = HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE
    base_clip = ColorClip(size=frame_size, color=(255, 255, 255), duration=duration)
    base_clip = base_clip.set_fps(fps)

    # Create the title text as a separate clip
    title_clip = TextClip(
        formatted_title,
        fontsize=FONT_SIZES["code_title"],
        color="black",
        method="caption",
        align="center",
        size=frame_size,
    )

    # Center the title
    title_clip = title_clip.set_position("center")

    # Set duration and add fade effects
    title_clip = title_clip.set_duration(duration)
    title_clip = title_clip.crossfadein(duration * 0.3)
    title_clip = title_clip.crossfadeout(duration * 0.3)

    # Composite the clips
    final_clip = CompositeVideoClip([base_clip, title_clip])
    return final_clip


def code_display_clip_with_highlighting(code_text, duration=4.0, fps=24, segment_type=""):
    """Create an animated code display with line-by-line animation using PIL for rendering."""
    logger.info(f"Creating code display clip for {segment_type}")

    # Create a white background clip first
    frame_size = HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE
    base_clip = ColorClip(size=frame_size, color=(255, 255, 255), duration=duration)
    base_clip = base_clip.set_fps(fps)

    # Remove 'import numpy as np' to save time as requested
    code_lines = code_text.split("\n")
    filtered_code_lines = []
    for line in code_lines:
        if not line.strip().startswith("import numpy as np"):
            filtered_code_lines.append(line)

    code_lines = filtered_code_lines

    # Calculate timing for each line
    line_time = duration / len(code_lines)

    # We'll use PIL to render text directly
    def make_frame(t):
        from PIL import Image, ImageDraw, ImageFont

        # Create a blank white image
        img = Image.new("RGB", frame_size, color="white")
        draw = ImageDraw.Draw(img)

        # Try to load the font, with fallback to default
        try:
            font = ImageFont.truetype("Courier", FONT_SIZES["code"])
        except:
            try:
                font = ImageFont.truetype("Arial", FONT_SIZES["code"])
            except:
                font = ImageFont.load_default()

        # Determine which lines should be visible at time t
        for i, line in enumerate(code_lines):
            line_start_time = i * line_time

            if t >= line_start_time:
                # Calculate opacity for this line
                time_since_appear = t - line_start_time
                if time_since_appear < line_time * 0.3:  # Fade in during first 30% of line time
                    opacity = int(255 * time_since_appear / (line_time * 0.3))
                else:
                    opacity = 255

                # Draw this line of code with calculated opacity
                text_color = (0, 0, 0, opacity)  # Black with opacity
                y_position = 50 + i * int(FONT_SIZES["code"] * 1.2)  # Add spacing between lines
                draw.text((50, y_position), line, fill=text_color, font=font)

        # Convert PIL Image to numpy array
        return np.array(img)

    # Create the clip
    clip = VideoClip(make_frame, duration=duration)
    clip = clip.set_fps(fps)

    return clip


def loading_animation_clip(duration=1.0, fps=24):
    """Create a simple loading animation using direct frame generation."""
    logger.info("Creating loading animation clip.")

    # Create a white background clip
    frame_size = HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE

    # Use the same approach as the code display - generate frames directly
    def make_frame(t):
        from PIL import Image, ImageDraw, ImageFont

        # Create a blank white image
        img = Image.new("RGB", frame_size, color="white")
        draw = ImageDraw.Draw(img)

        # Try to load the font, with fallback to default
        try:
            font = ImageFont.truetype("Courier", FONT_SIZES["code"])
        except:
            try:
                font = ImageFont.truetype("Arial", FONT_SIZES["code"])
            except:
                font = ImageFont.load_default()

        # Calculate the number of dots to show (0-3)
        num_dots = int((t / duration) * 4) % 4
        dots = "." * num_dots

        # Create the loading text with variable dots
        loading_text = f">>> Loading{dots}"

        # Draw the text
        draw.text((50, 50), loading_text, fill="black", font=font)

        # Convert PIL Image to numpy array
        return np.array(img)

    # Create the clip
    clip = VideoClip(make_frame, duration=duration)
    clip = clip.set_fps(fps)

    return clip


def smooth_plot_display(image_path, duration=5.0, fps=24):
    """Display plot smoothly: first static, then pan across subplots in a 3x3 grid."""
    logger.info(f"Creating smooth plot display for {image_path}.")

    # Load the original image
    img_clip = ImageClip(image_path)

    # Get dimensions
    w, h = img_clip.size

    # Calculate the duration for each phase
    static_time = duration * 0.35  # Show full plot for 35% of the time
    pan_time = duration - static_time  # The rest is for panning

    # Create a white background frame
    frame_size = HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE

    def make_frame(t):
        # Create white background
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255

        # First phase: static display of the full image
        if t < static_time:
            # Resize image to fill frame while maintaining aspect ratio
            img = img_clip.get_frame(0)
            img_pil = Image.fromarray(img)

            # Calculate dimensions to fill the frame
            img_aspect = w / h
            frame_aspect = frame_size[0] / frame_size[1]

            if img_aspect > frame_aspect:
                # Image is wider than frame
                new_width = frame_size[0]
                new_height = int(new_width / img_aspect)
            else:
                # Image is taller than frame
                new_height = frame_size[1]
                new_width = int(new_height * img_aspect)

            # Resize the image
            img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Calculate position to center in frame
            x_offset = (frame_size[0] - new_width) // 2
            y_offset = (frame_size[1] - new_height) // 2

            # Create a frame with the image centered
            frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
            img_array = np.array(img_resized)

            # Place the image in the frame
            frame[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = img_array

            return frame

        # Second phase: systematic panning across the image
        # Calculate pan position using a smooth path
        pan_t = (t - static_time) / pan_time

        # Create a smooth path through the 3x3 grid
        # Define the 9 grid positions (normalized 0-1)
        grid_positions = [
            (0.25, 0.25),  # Top-left
            (0.5, 0.25),  # Top-center
            (0.75, 0.25),  # Top-right
            (0.75, 0.5),  # Middle-right
            (0.75, 0.75),  # Bottom-right
            (0.5, 0.75),  # Bottom-center
            (0.25, 0.75),  # Bottom-left
            (0.25, 0.5),  # Middle-left
            (0.5, 0.5),  # Center
        ]

        # Find the two grid positions to interpolate between
        num_positions = len(grid_positions)
        idx = min(int(pan_t * num_positions), num_positions - 1)

        # For the last position, just stay put
        if idx == num_positions - 1:
            pos = grid_positions[idx]
        else:
            # Interpolate between the current and next position
            t_sub = pan_t * num_positions - idx
            pos1 = grid_positions[idx]
            pos2 = grid_positions[(idx + 1) % num_positions]
            pos = (pos1[0] + t_sub * (pos2[0] - pos1[0]), pos1[1] + t_sub * (pos2[1] - pos1[1]))

        # Calculate the slice of the image to show
        # We'll show a zoomed portion (about 40% of the image)
        zoom_factor = 0.4

        # Calculate the slice position based on the current position
        x_center = int(w * pos[0])
        y_center = int(h * pos[1])

        # Calculate slice dimensions
        slice_w = int(w * zoom_factor)
        slice_h = int(h * zoom_factor)

        # Calculate slice boundaries with bounds checking
        left = max(0, x_center - slice_w // 2)
        right = min(w, x_center + slice_w // 2)
        top = max(0, y_center - slice_h // 2)
        bottom = min(h, y_center + slice_h // 2)

        # Extract the slice
        img_frame = img_clip.get_frame(0)
        slice_frame = img_frame[top:bottom, left:right]

        # Resize to fill the frame while maintaining aspect ratio
        slice_pil = Image.fromarray(slice_frame)
        slice_aspect = slice_w / slice_h
        frame_aspect = frame_size[0] / frame_size[1]

        if slice_aspect > frame_aspect:
            # Slice is wider than frame
            new_width = frame_size[0]
            new_height = int(new_width / slice_aspect)
        else:
            # Slice is taller than frame
            new_height = frame_size[1]
            new_width = int(new_height * slice_aspect)

        # Resize the slice
        slice_resized = slice_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate position to center in frame
        x_offset = (frame_size[0] - new_width) // 2
        y_offset = (frame_size[1] - new_height) // 2

        # Place the image in the frame
        slice_array = np.array(slice_resized)
        frame[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = slice_array

        return frame

    # Create the clip
    clip = VideoClip(make_frame, duration=duration)

    # Set the fps
    clip = clip.set_fps(fps)

    return clip


# ---------- Code Snippets for Each Example ----------

code_1d = """\
import numpy as np
from alltheplots import plot

# Create a 1D array with parabola and noise
my_1D_tensor = np.linspace(-10, 10, 300)**2 + np.random.normal(0, 10, 300)

# Plot with a single function call
plot(my_1D_tensor)"""

code_2d = """\
import numpy as np
from alltheplots import plot

# Create a 2D sine wave pattern
my_2D_tensor = np.sin(np.linspace(0, 4*np.pi, 100).reshape(10, 10) +
                      np.cos(np.linspace(0, 4*np.pi, 100).reshape(10, 10)))

# Plot with a single function call
plot(my_2D_tensor)"""

code_3d = """\
import numpy as np
from alltheplots import plot

def gaussian_3d_mod(resolution=30, sigma=(0.5, 0.7, 0.9), 
                    offset=(0.2, -0.1, 0.3), size=2, noise_level=0.05):
    # Create a 3D Gaussian density field with noise
    x = np.linspace(-size, size, resolution) + offset[0]
    y = np.linspace(-size, size, resolution) + offset[1]
    z = np.linspace(-size, size, resolution) + offset[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    gauss = np.exp(-((xx**2)/(2*sigma[0]**2) + 
                     (yy**2)/(2*sigma[1]**2) + 
                     (zz**2)/(2*sigma[2]**2)))
    noise = np.random.normal(scale=noise_level, size=gauss.shape)
    return gauss + noise

# Create a 3D tensor
my_3d_tensor = gaussian_3d_mod()

# Plot with a single function call
plot(my_3d_tensor)"""

code_4d = """\
import numpy as np
from alltheplots import plot

def complex_topology_4d(resolution=48, noise_level=0.1):
    # Generate a 4D composite array with non-uniform topology
    x = np.linspace(-3, 3, resolution)
    y = np.linspace(-3, 3, resolution)
    z = np.linspace(-3, 3, resolution)
    w = np.linspace(-3, 3, resolution)
    xx, yy, zz, ww = np.meshgrid(x, y, z, w, indexing='ij')
    g1 = np.exp(-(((xx+1)**2 + (yy+1)**2 + (zz+1)**2 + (ww+1)**2)/2.0))
    g2 = np.exp(-(((xx-1)**2 + (yy-1)**2 + (zz-1)**2 + (ww-1)**2)/2.0))
    sine_component = np.sin(2*xx) * np.cos(2*yy) * np.sin(2*zz) * np.cos(2*ww)
    composite = g1 + g2 + 0.5 * sine_component
    composite += np.random.normal(scale=noise_level, size=composite.shape)
    return composite

# Create a 4D tensor
my_nd_tensor = complex_topology_4d()

# Plot with a single function call
plot(my_nd_tensor)"""

# List of tuples: (code snippet, corresponding output image file, duration for code display)
segments = [
    ("1d", code_1d, "plot_1d.png", 3.5),  # Simple, so shorter duration
    ("2d", code_2d, "plot_2d.png", 3.5),  # Simple, so shorter duration
    ("3d", code_3d, "plot_3d.png", 5.0),  # More complex, so longer duration
    ("nd", code_4d, "plot_nd.png", 5.5),  # Most complex, so longest duration
]

# Filter segments if using SHORT_VERSION
if SHORT_VERSION:
    segments = [seg for seg in segments if seg[0] in SELECTED_PLOTS]

# Create clips for each segment
clips = []

# Add intro clip (longer duration to allow all three lines to appear)
intro_clip = create_intro_clip(duration=4.0)
clips.append(intro_clip)

# Process each segment
for plot_type, code_text, image_file, code_duration in segments:
    logger.info(f"Processing segment {plot_type} with output image: {image_file}")

    # Title clip for this segment
    title_clip = create_title_clip(plot_type, duration=1.5)

    # Code display with syntax highlighting and line-by-line animation
    code_clip = code_display_clip_with_highlighting(
        code_text, duration=code_duration, fps=OUTPUT_FPS, segment_type=plot_type
    )

    # Loading animation
    loading_clip = loading_animation_clip(duration=1.0)

    # Plot display with pan effect
    plot_clip = smooth_plot_display(image_file, duration=5.0)

    # Short pause between segments (white background)
    pause = ColorClip(
        size=HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE,
        color=(255, 255, 255),
        duration=0.8,
    )

    # Combine clips for this segment
    segment_clips = [title_clip, code_clip, loading_clip, plot_clip, pause]
    segment_clip = concatenate_videoclips(segment_clips, method="compose")
    clips.append(segment_clip)

# Add outro clip with italic text
outro_clip = create_outro_clip(duration=2.5)
clips.append(outro_clip)

# Combine all clips
final_clip = concatenate_videoclips(clips, method="compose")

# Ensure the final clip has the correct size and is properly padded with white
# (Ensuring no black borders anywhere)
final_size = HIGH_RES_SIZE if not SHORT_VERSION else LOW_RES_SIZE
final_clip = final_clip.resize(width=final_size[0], height=final_size[1])
final_clip = final_clip.on_color(size=final_size, color=(255, 255, 255))

# Output settings
output_fps = OUTPUT_FPS
output_name = "demo" if not SHORT_VERSION else "demo_short"

# Write GIF file
output_path = f"../resources/{output_name}.gif"
logger.info(f"Writing GIF file {output_path}")
final_clip.write_gif(
    output_path,
    fps=12,
    program="ffmpeg",
    opt="nq",
    fuzz=0,  # Reduces dithering which can cause dark borders
)
logger.info("Finished writing GIF file.")
