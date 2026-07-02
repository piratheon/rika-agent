# skill: youtube
description: Download YouTube video transcripts, captions, and playlist metadata using yt-dlp.
tools: run_shell_command, run_python, write_file

## When to use
- Summarising YouTube videos without watching them
- Downloading captions/subtitles for a video or playlist
- Extracting metadata (title, description, chapters)

## Usage pattern
```python
# Download auto-generated captions for a single video
run_shell_command("yt-dlp --write-auto-subs --skip-download --sub-format vtt -o '%(title)s.%(ext)s' 'VIDEO_URL'")

# Download captions for an entire playlist
run_shell_command("yt-dlp --write-auto-subs --skip-download --sub-format vtt -o '%(playlist_index)s-%(title)s.%(ext)s' 'PLAYLIST_URL'")

# Extract plain text from a .vtt file
run_python(code="""
import re, pathlib
vtt = pathlib.Path('captions.vtt').read_text()
lines = [l for l in vtt.splitlines() if l and not re.match(r'^(WEBVTT|NOTE|\d+:\d+|<\d)', l)]
print('\n'.join(dict.fromkeys(lines)))  # deduplicate consecutive identical lines
""")
```

## Notes
- Requires yt-dlp installed: `pip install yt-dlp`
- VTT is the most portable subtitle format; use --sub-lang en for English
- For audio download: add `-f bestaudio` and `--extract-audio`
